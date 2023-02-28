import torch.nn as nn


class Linear(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim=1,
                 layers=(256, ),
                 act="LeakyReLU"):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, 1))
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        a=0.1,
                                        mode="fan_in",
                                        nonlinearity="leaky_relu")

    def forward(self, x):
        cur_output = x
        cur_output = self.layers(cur_output)
        cur_output = cur_output.squeeze(1)
        return cur_output