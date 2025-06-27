import torch
import torch.nn as nn
from scipy.integrate import lebedev_rule
from torch_efa.erope import ERoPE
from e3nn import o3

def get_rope_freqs(K):
    pass

class EuclideanFastAttention(nn.Module):
    def __init__(self, in_dim, out_dim, max_degree, num_lebedev):
        super().__init__()

        # TODO: define qkv kernel matrices

        self.L_max = max_degree
        self.num_lebedev = num_lebedev

        self.erope = ERoPE()

    def forward(self, x, pos):
        """
        x : node features of different irreps of size [B, N_atoms, P, (L+1)^2, H]
        pos : node coordinates of size [B, N_atoms, 3]
        """

        q = None
        k = None
        v = None

        grid_pts, grid_weights = lebedev_rule(n=self.num_lebedev)
        grid_pts = torch.from_numpy(grid_pts).T.float().unsqueeze(0)
        grid_weights = torch.from_numpy(grid_weights).float().unsqueeze(0)

        freqs = get_rope_freqs()

        Bu = self.erope(q, k, v)

        sph_harm = o3.spherical_harmonics(self.L_max, Irreps=) # [B, N_atoms, (L_max+1)^2]