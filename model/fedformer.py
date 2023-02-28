import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dlinear import SeriesDecomp
from .autoformer import DataEmbedding_wo_pos, Encoder, EncoderLayer, AutoCorrelation, AutoCorrelationLayer, my_Layernorm, Decoder, DecoderLayer


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class FourierBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 seq_len,
                 modes=0,
                 mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len,
                                         modes=modes,
                                         mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale *
                                     torch.rand(8,
                                                in_channels // 8,
                                                out_channels // 8,
                                                len(self.index),
                                                dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B,
                             H,
                             E,
                             L // 2 + 1,
                             device=x.device,
                             dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i],
                                                   self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


class FourierCrossAttention(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 seq_len_q,
                 seq_len_kv,
                 modes=64,
                 mode_select_method='random',
                 activation='tanh',
                 policy=0):
        super(FourierCrossAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        # get modes for queries and keys (& values) on frequency domain
        self.index_q = get_frequency_modes(
            seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(
            seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv),
                                                self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale *
                                     torch.rand(8,
                                                in_channels // 8,
                                                out_channels // 8,
                                                len(self.index_q),
                                                dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B,
                             H,
                             E,
                             len(self.index_q),
                             device=xq.device,
                             dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B,
                             H,
                             E,
                             len(self.index_kv),
                             device=xq.device,
                             dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        # perform attention mechanism on frequency domain
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(
                self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B,
                             H,
                             E,
                             L // 2 + 1,
                             device=xq.device,
                             dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels,
                              n=xq.size(-1))
        return (out, None)


class Fedformer(nn.Module):
    """
    Fedformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self,
                 enc_in=7,
                 dec_in=7,
                 modes=32,
                 mode_select_method='random',
                 d_model=512,
                 n_heads=8,
                 e_layers=1,
                 d_layers=1,
                 d_ff=256,
                 label_len=22,
                 dropout=0.0,
                 activation='gelu',
                 seq_len=45,
                 kernel_size=5,
                 moving_avg=9,
                 decode_method='linear'):
        super.__init__()
        # Decomp
        self.decomp = SeriesDecomp(kernel_size)
        self.decode_method = decode_method
        self.label_len = label_len
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos
        (enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding_wo_pos
        (dec_in, d_model, dropout)
        # Encoder
        encoder_self_att = FourierBlock(in_channels=d_model,
                                        out_channels=d_model,
                                        seq_len=seq_len,
                                        modes=modes,
                                        mode_select_method=mode_select_method)
        self.encoder = Encoder([
            EncoderLayer(
                AutoCorrelationLayer(encoder_self_att, d_model, n_heads),
                d_model, d_ff, moving_avg, dropout, activation)
            for l in range(e_layers)
        ],
                               norm_layer=my_Layernorm(d_model))
        # Decoder
        decoder_self_att = FourierBlock(in_channels=d_model,
                                        out_channels=d_model,
                                        seq_len=seq_len // 2 + 1,
                                        modes=modes,
                                        mode_select_method=mode_select_method)
        decoder_cross_att = FourierCrossAttention(
            in_channels=d_model,
            out_channels=d_model,
            seq_len_q=seq_len // 2 + 1,
            seq_len_kv=seq_len,
            modes=modes,
            mode_select_method=mode_select_method)
        self.decoder = Decoder([
            DecoderLayer(
                AutoCorrelationLayer(decoder_self_att, d_model, n_heads),
                AutoCorrelationLayer(decoder_cross_att, d_model, n_heads),
                d_model,
                1,
                d_ff,
                moving_avg,
                dropout,
                activation,
            ) for l in range(d_layers)
        ],
                               norm_layer=my_Layernorm(d_model),
                               projection=nn.Linear(d_model, 1, bias=True))

        self.rnn = nn.GRU(input_size=d_model,
                          hidden_size=d_model,
                          num_layers=1,
                          batch_first=True,
                          dropout=dropout)

        # Projection
        self.projection = nn.Linear(d_model, 1)
        self.projection2 = nn.Linear(enc_in, 1)

    def forward(self, x):
        # [batch_size, seq_len, features]
        x_enc = x[:, :, :-6]
        x_mark_enc = x[:, :, -6:]
        x_dec = x[:, -self.label_len - 1:, :-6]
        x_mark_dec = x[:, -self.label_len - 1:, -6:]
        if self.decode_method == 'linear':
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, _ = self.encoder(enc_out)
            enc_out, _ = self.rnn(enc_out)
            output = self.projection(enc_out[:, -1, :])
        else:
            # decomp init for every features
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, 1, 1)
            zeros = torch.zeros([x_dec.shape[0], 1, x_dec.shape[2]],
                                device=x_enc.device)
            seasonal_init, trend_init = self.decomp(
                x_enc)  # [batch_size, seq_1en, features]
            trend_init = torch.cat
            [trend_init[:, -self.label_len:, :], mean], dim = 1
            seasonal_init = torch.cat
            [seasonal_init[:, -self.label_len:, :], zeros], dim = 1

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out,
                                                 enc_out,
                                                 trend=trend_init)

        # final
        dec_out = trend_part + seasonal_part
        output = self.projection2(dec_out[:, -1, :]).squeeze()
        return output.squeeze()