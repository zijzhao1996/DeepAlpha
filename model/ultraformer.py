import math
import torch
import copy
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from .grnformer import GatedResidualNetwork

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
    super().__init__
Compute the positionalencodingsonce in log space
pe = torch.zeros(max_len,  d_model).float(
pe.requires_grad = False
position = torch.arange(0,  max_len).float().unsqueeze(1)
div_term =(torch.arange(0, d_model, 2).float()
- -(math.log(10000.0) / d_model)).exp(
pe[:, 0::2] = torch.sin(position * div_term
pe[:, 1::2] = torch.cos(position * div_term
pe = pe.unsqueeze(0)
self.register_buffer('pe', pe
def forward(self, x):
return x + self.pe[:, :x.size(1)
class TokenEmbedding(nn.Module)
def __init__(self, c_in, d_model):
super().init__(
padding = 1 if torch.__version__ >= '1.5.@' else 2
self.tokenConv = nn.Conv1d(in_channels=d_model,
out_channels=d_model,
kernel_size=3,
padding=padding,
padding_mode='circular'
for m in self.modules():
if isinstance(m, nn.Conv1d):
nn.init kaiming_normal_
m.weight, mode='fan_in', nonlinearity='leaky_relu')
def forward(self, x):
x= self.tokenConv(x.transpose(1,  0).transpose(2, 1)
.transpose2,  1).transpose(1,0
return x
class DataEmbedding(nn.Module):
def __init__(self, c_in, d_model, dropout=0.1):
super).init__()
self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model
self.positional_embedding = PositionalEmbedding(d model=d_model
self.dropout = nn.Dropout(p=dropout
def forward(self, x):
x = self.value_embedding(x) + self.positional_embedding(x
return self.dropout(x
def _get_clones(module, N):
return ModuleList([copy.deepcopy(module) for i in range(N)]
class  LocalformerEncoder(nn.Module):
constants_^"norm"'
def _init__(self, encoder_layer, num_layers, d_model):
super(LocalformerEncoder, self)._init_()
self.layers = _get_clones(encoder_layer,num_layers
self.conv = _get_clones
nn.Conv1d(d_model, d_model, 3, 1, 1), num_layers
self.num_layers = num_layers
def forward(self, src, mask}:
output = src
out = src
for i, mod in enumerate(self.layers}:
[T, N, F]--> [N, T, F]--> [N, F, T]
out = output.transpose(1, 0).transpose(2, 1)
out = self.conv[i](out).transpose(2, 1).transpose(1,0
output = mod(output + out, src_mask=mask
return output + out
class Ultraformer(nn.Module):
def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dim_feedforward=1024, dropout=0.5, device=None):
super(._init__
here iwe use bi-GRU
self.rnn = nn.GRU(
input_size=d_model,
hidden_size=d_model,
num_layers=num_layers
batch_first=False
self.grn = GatedResidualNetwork
45*d_model, d_model, d_model, dropout
self.feature_layer = nn.Linear(d_feat, d_model
seIf.enc_embedding = DataEmbedding(d_feat, d_model, dropout
self. encoder_layer = nn.TransformerEncoderLayer
d_model=d_model,nhead=nhead,dim feedforward=dim feedforward,dropout=dropout
self.encoder = LocalformerEncoder
self encoder_layer, num_layers=num_layers, d_model=d_model
self.decoder = nn.Linear(d_ model, 1)
self.device = device
self.d_feat = d_feat
def forward(self, src):
src = self.feature_layer(src)  # [batch, 45, 64]
src = :src.transpose(1,0# [45, batch, 64]
mask = None
src = self.enc_embedding(src))  # [45, batch, 64]
output = self.encoder(src,  mask) # [45, batch, 64]
oatpst= self.rnn(output) # [45, batch, 64]
output=self.grn(output.transpose(1, 0))
output = self.decoder(output)  # [batch, 1]
return output.squeeze# [batch]