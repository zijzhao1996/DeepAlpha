import math
import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1),
                       y.size(-1))  # (timesteps, samples, output_size)

        return y


class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_state_size,
                 output_size,
                 dropout,
                 hidden_context_size=None,
                 batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size = hidden_state_size
        self.dropout = dropout

        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(
                nn.Linear(self.input_size, self.output_size))

        self.fc1 = TimeDistributed(nn.Linear(self.input_size,
                                             self.hidden_state_size),
                                   batch_first=batch_first)
        self.elu1 = nn.ELU()

        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(self.hidden_context_size,
                                                     self.hidden_state_size),
                                           batch_first=batch_first)

        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size,
                                             self.output_size),
                                   batch_first=batch_first)
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(self.dropout)
        self.bn = TimeDistributed(nn.BatchNorm1d(self.output_size),
                                  batch_first=batch_first)
        self.gate = TimeDistributed(GLU(self.output_size),
                                    batch_first=batch_first)

    def forward(self, x, context=None):

        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.bn(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[:x.size(0), :]


class GRNformer(nn.Module):

    def __init__(self,
                 d_feat=6,
                 d_model=8,
                 seq_length=45,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=1024,
                 dropout=0.5,
                 device=None):
        super().__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                             num_layers=num_layers)
        self.grn = GatedResidualNetwork(seq_length * d_model, d_model, 32,
                                        dropout)
        self.decoder = nn.Linear(32, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        src = self.feature_layer(src)  # linear transform
        srC = src.transpose(1, 0)
        mask = None
        src = self.pos_encoder(src)
        output = self.encoder(src, mask)
        # add a GRN layer before entering decoder
        output = self.grn(output.transpose(1, 0))
        output = self.decoder(output)
        return output.squeeze()