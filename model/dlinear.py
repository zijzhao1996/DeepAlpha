import torch
import torch.nn as nn


class MovingAvg(nn.Module):

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride,
                                padding=0)

    def forward(self, x):
        """
        padding on both ends of the time series
        """
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):

    def __init__(self, input_dim, seq_length, kernel_size):
        super()._init_()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.kernel_size = kernel_size
        self.linear_seasonal = nn.ModuleList()
        self.linear_trend = nn.ModuleList()
        for i in range(self.input_dim):
            self.linear_seasonal.append(nn.Linear(self.seq_length, 1))
            self.linear_seasonal[i].weight = nn.Parameter(
                (1 / self.seq_length) * torch.ones([1, self.seq_length]))
            self.linear_trend.append(nn.Linear(self.seq_length, 1))
            self.linear_trend[i].weight = nn.Parameter(
                (1 / self.seq_length) * torch.ones([1, self.seq_length]))

        self.decomposition = SeriesDecomp(self.kernel_size)
        self.decoder = nn.Linear(self.input_dim, 1)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2,
            1), trend_init.permute(0, 2,
                                   1)  # [batch_size, input_dim, seq_length]

        seasonal_output = torch.zeros(
            [seasonal_init.size(0), self.input_dim, 1],
            dtype=seasonal_init.dtype).to(seasonal_init.device)
        trend_output = torch.zeros([trend_init.size(0), self.input_dim, 1],
                                   dtype=trend_init.dtype).to(
                                       trend_init.device)
        for i in range(self.input_dim):
            seasonal_output[:, i, :] = self.linear_seasonal[i](
                seasonal_init[:, i, :])
            trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1).squeeze()  # [batch_size, 1, input_dim]
        output = self.decoder(x)
        return output.squeeze