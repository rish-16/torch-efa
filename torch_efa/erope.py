"""
PyTorch code inspired by the Jax implementation in
https://github.com/thorben-frank/euclidean_fast_attention/blob/main/euclidean_fast_attention/rope.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LEBEDEV_FREQUENCY_LOOKUP = {
    50: np.pi,
    86: 2 * np.pi,
    110: 2.5 * np.pi,
    146: 3 * np.pi,
    194: 4 * np.pi,
    230: 4.5 * np.pi,
    266: 5 * np.pi,
    302: 5.5 * np.pi,
    350: 6.5 * np.pi,
    434: 7.5 * np.pi,
    590: 9 * np.pi,
    770: 11 * np.pi,
    974: 12.5 * np.pi,
    6000: 35 * np.pi,
}

def apply_rotary_position_embedding(x, sin, cos):
    """
    x : torch.Tensor of shape [B, N_nodes, dim]
    sin : torch.Tensor of shape [B, N_nodes, dim]
    cos : torch.Tensor of shape [B, N_nodes, dim]
    """

    assert x.shape[-1] % 2 == 0, "x must have even dims"
    assert x.shape[-1] == sin.shape[-1] == cos.shape[-1], "x, sin, cos must have matching feature dims"

    x = x.unsqueeze(0) # [1, B, N, dim]
    sin = sin.unsqueeze(-3).unsqueeze(-2) # include parity dim
    cos = cos.unsqueeze(-3).unsqueeze(-2) # include parity dim
    
    # stack over feature dimension
    y = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape) # [1, B, N, dim]
    return x*cos + y*sin # [1, B, N, dim]

def calculate_rotary_position_embedding(x, theta):
    """
    x : torch.Tensor of shape [B, N] corresponding to dot-products of u * r
    theta : torch.Tensor of frequencies of shape (M)

    returns:
        torch.Tensor of shape [B, N, 2*M]
    """
    # NOTE: theta = [w1, w2, ..., wK] -> each freq is applied to pairs of elements along dim_qk
    angle = x[..., :, None] * theta[None, :] # [B, N, 1] * [1, M] -> [B, N, M]
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    sin = torch.repeat_interleave(x, 2, dim=-1) # [B, N, 2*M]
    cos = torch.repeat_interleave(x, 2, dim=-1) # [B, N, 2*M]

    assert sin.shape[-1] == 2 * theta.shape[-1] and cos.shape[-1] == 2 * theta.shape[-1]

    return sin, cos

class ERoPE(nn.Module):
    def __init__(self):
        """
        Efficient PyTorch implementation of Euclidean RoPE, 
        used inside Euclidean Fast Attention to encode spatial
        position into scalar node features.
        """
        super().__init__()

    def forward(self, q, k, v, pos, theta=None):
        """
        q : query node features [B, N, 1 or 2, (max_degree_qk+1)**2, dim_qk]
        k : key node features [B, N, 1 or 2, (max_degree_qk+1)**2, dim_qk]
        v : value node features [B, N, 1 or 2, (max_degree_v+1)**2, dim_v]
        pos : node coordinates [B, N, 3]
        theta : [M]
        """

        grid_u = None
        pos_proj = torch.einsum("nd,md->nm", pos, grid_u) # [B, N]
        sin, cos = calculate_rotary_position_embedding(pos_proj, theta) # [B, N, M, dim_qk]

        q = apply_rotary_position_embedding(q, sin, cos)  # (N, M, P, L, num_features_qk)
        k = apply_rotary_position_embedding(k, sin, cos)  # (N, M, P, L, num_features_qk)

        q = q.reshape(*q.shape[:-3], -1) # (B, N, M, dim_qk)
        k = k.reshape(*k.shape[:-3], -1) # (B, N, M, dim_qk)
        v = v.reshape(*v.shape[:-3], -1) # (B, N, dim_v)

        q = q / torch.sqrt(q.shape[-1])