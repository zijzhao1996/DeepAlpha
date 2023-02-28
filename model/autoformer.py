import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .informer import TockenEmbedding, Encoder
from .dlinear import SeriesDecomp


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self,
                 mask_flag=True,
                 factor=1,
                 scale=None,
                 attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)],
                              dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(
                1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[...,
                                                          i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(),
                               dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(),
                corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(),
                corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):

    def __init__(self,
                 correlation,
                 d_model,
                 n_heads,
                 d_keys=None,
                 d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TimeFeatureEmbedding(nn.Module):

    def __init__(self, d_model, freq='h'):
        super().__init__()
        freq_map = {'h': 6}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)


def forward(self, x):
    return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """

    def __init__(self,
                 attention,
                 d_model,
                 d_ff=None,
                 moving_avg=25,
                 dropout=0.1,
                 activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff,
                               kernel_size=1,
                               bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff,
                               out_channels=d_model,
                               kernel_size=1,
                               bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """

    def __init__(self,
                 self_attention,
                 cross_attention,
                 d_model,
                 c_out,
                 d_ff=None,
                 moving_avg=25,
                 dropout=0.1,
                 activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model,
                               out_channels=d_ff,
                               kernel_size=1,
                               bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff,
                               out_channels=d_model,
                               kernel_size=1,
                               bias=False)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model,
                                    out_channels=c_out,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    padding_mode='circular',
                                    bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(
            self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2,
                                                                1)).transpose(
                                                                    1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x,
                                      cross,
                                      x_mask=x_mask,
                                      cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self,
                 enc_in=7,
                 dec_in=7,
                 label_len=8,
                 d_model=512,
                 n_heads=8,
                 factor=5,
                 e_layers=3,
                 d_layers=3,
                 d_ff=256,
                 dropout=0.0,
                 activation='gelu',
                 output_attention=False,
                 kernel_size=5,
                 moving_avg=9,
                 decode_method='linear'):
        super().__init__()
        self.output_attention = output_attention
        self.decode_method = decode_method
        self.label_len = label_len
        self.decode_method = decode_method

        # Decomp
        self.decomp = SeriesDecomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder([
            EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, factor, dropout, output_attention),
                    d_model, n_heads), d_model, d_ff, moving_avg, dropout,
                activation) for l in range(e_layers)
        ],
                               norm_layer=my_Layernorm(d_model))

        # Decoder
        self.decoder = Decoder([
            DecoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(
                        True, factor, dropout, output_attention=False),
                    d_model, n_heads),
                AutoCorrelationLayer(
                    AutoCorrelation(
                        False, factor, dropout, output_attention=False),
                    d_model, n_heads),
                d_model,
                1,
                d_ff,
                moving_avg,
                dropout,
                activation,
            ) for l in range(d_layers)
        ],
                               norm_layer=my_Layernorm(d_model),
                               projection=nn.Linear(d_model, 1, bias=1))

        # Projection
        self.projection1 = nn.Linear(enc_in, 1)
        self.projection2 = nn.Linear(d_model, 1)

    def forward(self, x):
        # [batch_size, seq_len, features]
        x_enc = x[:, :, :-6]
        x_mark_enc = x[:, :, -6:]
        x_dec = x[:, -self.label_len - 1:, :-6]
        x_mark_dec = x[:, -self.label_len - 1:, -6:]
        if self.decode_method == 'linear':
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            # delete RNN here
            enc_out, _ = self.encoder(enc_out)
            output = self.projection1(enc_out[:, -1, :])
        else:
            # decomp init for everyfeatures
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, 1, 1)
            zeros = torch.zeros([x_dec.shape[0], 1, x_dec.shape[2]],
                                device=x_enc.device)
            seasonal_init, trend_init = self.decomp(
                x_enc)  # [batch_size, seq_len, features]
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean],
                                   dim=1)
            seasonal_init = torch.cat(
                [seasonal_init[:, -self.label_len:, :], zeros], dim=1)

            # enc
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)

            # deq
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
            seasonal_part, trend_part = self.decoder
            dec_out, enc_out, trend = trend_init

            # final
            dec_out = trend_part + seasonal_part
            output = self.projection2(dec_out[:, -1, :]).squeeze()

        return output