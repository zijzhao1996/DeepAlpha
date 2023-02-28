class Mixformer(nn.Module):
def __init__(self, enc_in=7,
d_feat=512, d_model=512,  n_heads=8, factor=5,
e_layers=3,  d_layers=3,  d_ff=256,
dropout=0.5,  activation='gelu',
output_attention=False,kernel_size=5, moving_avg=9,
decode_method='linear'):
super(.init
self.output_attention = output_attention
self.decode_method = decode_method
self.d_feat = d_feat
self.rnn = nn.GRU(
input_size=d model,
hidden_size=d_model,
num_layers=1,
batch_first=True,
dropout=dropout
LOCAL CONV TOWER
self.feature_layer = nn.Linear(d_feat, d_model
self.pos_encoder =PositionalEmbedding(d_model)
self.encoder_layer = nn.TransformerEncoderLayer
d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout
self.local_encoder = LocalformerEncoder
self encoder_layer, num_layers=e_layers, d_model=d_model
AUTO CORR TOWER
self.decomp = SeriesDecomp(kernel_size
self.enc_embedding = DataEmbedding_wo_pos
enc_in, d_model, dropout
self.auto_encoder = Encoder
EncoderLayer(
AutoCrrelationLayer
AutoCorrelation(False, factor,  dropoutj
output_attention),
d_model, n_heads),
d_model,
d_ff,
moving_avg,
dropout,
activation
) for l in range(e_layers)
norm_layer=my_Layernorm(d_model)
self.attn_decoder = Decoder
DecoderLayer(
AutoCrrelationLayer
AutoCorrelation(True,  factor, dropout,
output_attention=False)
d_model, n_heads),
AutocCrrelationLayer
AutoCorrelation(False, factor, dropout,
output_attention=False
d_model, n_heads),
d_model,
1,
d_ff,
moving_avg,
dropout,
activation,
) for l in range(d_layers)
norm_1ayer=my_Layernorm(d_model
self.decoder = nn.Linear(2*d_model, 1, bias=True
def forward(self,  src):
src [batch, seq_len, features]
LOCAL CONV TOWER
local_output = self.feature_layer(src) # [batch, seq_len, d_model]
local_output =self.pos_encoder(local_output# [seq_len, batch, d_model]
local_output = self.local_encoder(local_output) # [seq_length, batch, d_model]
Informer TOWER
only do value embedding,apply Conv1d on the seq_length dimensions
auto_output = self.enc_embedding(src) # [batch, seq_len, features]
auto_output, _= self.auto_encoder(auto_output.transpose(1, 0), attn_mask=None)
both tower output are  [seq_len, batch, d_model]
output = torch.cat((local_output, auto_output), dim=-1)
auto_output, _= self.rnn(auto_output.transpose(1,0))  # [batch, seq_len, d_model]
local_output,_=self.rnnlocal_output.transpose(1,0)  # [batch, seq_len, d_model]
output = torch.cat((local_output, auto_output), dim=-1
output = self.decoder(output[:, -1, :]) # [batch, 1]
return output.squeeze() # [batch]