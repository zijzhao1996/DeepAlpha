import torch.nn as nn


class MTL(nn.Module):

    def __init__(self, input_dim, shared_hidden_size, intra_tower_hidden_size,
                 cross_tower_hidden_size, output_dim):
        super().__init__()

        self.sharedbottom = nn.Sequential(
            nn.Linear(input_dim, shared_hidden_size), nn.ReLU(), nn.Dropout())

        self.intra_tower = nn.Sequential(
            nn.Linear(shared_hidden_size, intra_tower_hidden_size[0]),
            nn.ReLU(), nn.Dropout(),
            nn.Linear(intra_tower_hidden_size[0], intra_tower_hidden_size[1]),
            nn.ReLU(), nn.Dropout(),
            nn.Linear(intra_tower_hidden_size[1], output_dim))

        self.cross_tower = nn.Sequential(
            nn.Linear(shared_hidden_size, cross_tower_hidden_size[0]),
            nn.ReLU(), nn.Dropout(),
            nn.Linear(cross_tower_hidden_size[0], cross_tower_hidden_size[1]),
            nn.ReLU(), nn.Dropout(),
            nn.Linear(cross_tower_hidden_size[1], output_dim))

    def forward(self, x):
        h_shared = self.sharedbottom(x)
        cross_out = self.cross_tower(h_shared)
        intra_out = self.intra_tower(h_shared)
        return cross_out.squeeze(), intra_out.squeeze()