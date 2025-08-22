
import torch.nn as nn
class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.val = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_act),
        )
    def forward(self, x):
        z = self.feat(x)
        v = self.val(z)
        a = self.adv(z)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q