"""
PyTorch code inspired by the Jax implementation in
https://github.com/thorben-frank/euclidean_fast_attention/blob/main/euclidean_fast_attention/rope.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_efa.utils as utils

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

def apply_erope(x, pos, grid_u, theta):
    """
    x : node features of size [B, N_atoms, P, (L_qk+1)**2, H]
    pos : node coordinates of size [B, N_atoms, 3]
    grid_u : grid points of size [B, N_omega, 3]
    theta : frequencies of size [K]

    returns:
        x_tilde : node features with rotational information of size [B, N_omega, N_atoms, P, L(L_qk + 1)**2, H]

    notes:
        This operation adds an additional integration dimension N_omega over the Lebedev grid points on the 2-sphere
    """

    proj_pos = torch.einsum("bnd,bmd->bmn", pos, grid_u) # dot products of size [B, N_omega, N_atoms]

    repeated_theta = torch.repeat_interleave(theta, 2, dim=-1) # repeated frequencies of size [2K=H]
    inside_angles = proj_pos[..., None] @ repeated_theta[None, ...] # angles for cos/sin of size [B, N_omega, N_atoms, 2K=H]

    sin = torch.sin(inside_angles) # [B, N_omega, N_atoms, 2K=H]
    cos = torch.cos(inside_angles) # [B, N_omega, N_atoms, 2K=H]

    swapped_x = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape) # [B, N_atoms, P, (L_qk+1)**2, H=2K]

    out = torch.einsum("bNPLH,bMNH->bMNPLH", x, cos) + torch.einsum("bNPLH,bMNH->bMNPLH", swapped_x, sin) # [B, N_omega, N, P, (L_qk+1)**2, H=2K]

    return out

class ERoPE(nn.Module):
    def __init__(self):
        """
        Efficient PyTorch implementation of Euclidean RoPE, 
        used inside Euclidean Fast Attention to encode spatial
        position into scalar node features.
        """
        super().__init__()

    def forward(self, q, k, v, pos, grid_u, grid_w, theta=None, include_pseudotensors_qk=False, include_pseudotensors_v=False):
        """
        q : query node features [B, N, P, (L_qk + 1)**2, dim_qk]
        k : key node features [B, N, P, (L_qk + 1)**2, dim_qk]
        v : value node features [B, N, P, (L_v + 1)**2, dim_v]
        pos : node coordinates [B, N, 3]
        grid_u : grid points on unit sphere for Lebedev quadrature [B, N_omega, 3]
        grid_w : corresponding weights for Lebedev quadrature [B, N_omega]
        theta : rotation frequencies for RoPE [K]
        include_pseudotensors_qk: Include pseudotensors from query and key. 
        include_pseudotensors_v: Include pseudotensors from value.
        """

        q = apply_erope(q, pos, grid_u, theta) # (B, N_omega, N_atoms, P, (L_qk + 1)**2, d_qk)
        k = apply_erope(k, pos, grid_u, theta) # (B, N_omega, N_atoms, P, (L_qk + 1)**2, d_qk)

        B, N_omega, N_atoms, P, L_max, d_qk = q.shape
        B, N_atoms, P, L_max, d_v = v.shape

        q = q.reshape(B, N_omega, N_atoms, -1) # (N_omega, N_atoms, D_qk) with D_qk = P * (L_qk+1)^2 * d_qk
        k = k.reshape(B, N_omega, N_atoms, -1) # (N_omega, N_atoms, D_qk) with D_qk = P * (L_qk+1)^2 * d_qk
        v = v.reshape(B, N_atoms, -1) # (N_atoms, D_v) with D_v = P * (L_v+1)^2 * d_v

        q = q / ((q.shape[-1]) ** 0.5) # rescaling [B, N_omega, N_atoms, D_qk]

        kv = torch.einsum("bMNt,bNq->bMNtq", k, v) # KV outer product of size [B, N_omega, N_atoms, D_qk, D_v]

        # NOTE: check about segment_sum and how to apply to `kv`

        # multiply each grid weight with respective grid point on sphere
        y = torch.einsum(
            "bMNt,bMNtq,bM->bMNq",
            q,
            kv,
            grid_w
        ) # [B, N_omega, N_atoms, D_v]

        qkv = y.reshape(B, N_omega, N_atoms, P, L_max, d_v) # [B, N_omega, N_atoms, P, (L_v+1)**2, d_v]

        return qkv