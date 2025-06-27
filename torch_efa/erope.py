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

def calculate_rotary_position_embedding(dotp, theta):
    """
    dotp : torch.Tensor of shape [B, N, M] corresponding to dot-products of u * r
    theta : torch.Tensor of frequencies of shape (K)

    returns:
        torch.Tensor of shape [B, N, 2*K, dim]
    """
    # NOTE: theta = [w1, w2, ..., wK] -> each freq is applied to pairs of elements along dim_qk
    angle = dotp[..., :, None] * theta[None, :] # [B, N, M, 1] * [1, M] -> [B, N, M]
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    sin = torch.repeat_interleave(sin, 2, dim=-1) # [B, N, 2*M]
    cos = torch.repeat_interleave(cos, 2, dim=-1) # [B, N, 2*M]

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

    """
    def forward(self, q, k, v, pos, grid_u, grid_w, theta=None, max_degree_qk=None, max_degree_v=None, include_pseudotensors_qk=False, include_pseudotensors_v=False):
        '''
        q : query node features [B, N, 1 or 2, (max_degree_qk+1)**2, dim_qk]
        k : key node features [B, N, 1 or 2, (max_degree_qk+1)**2, dim_qk]
        v : value node features [B, N, 1 or 2, (max_degree_v+1)**2, dim_v]
        pos : node coordinates [B, N, 3]
        grid_u : grid points on unit sphere for Lebedev quadrature [N_pts, 3]
        grid_w : corresponding weights for Lebedev quadrature [N_pts]
        include_pseudotensors_qk: Include pseudotensors from query and key. 
        include_pseudotensors_v: Include pseudotensors from value.
        theta : rotation frequencies for RoPE [M]
        '''

        num_parity_qk_in, num_degrees_qk_in = q.shape[-3], q.shape[-2]
        num_parity_v_in, num_degrees_v_in = v.shape[-3], v.shape[-2]
        num_features_v = v.shape[-1]

        max_degree_v_present = int(np.rint(np.sqrt(num_degrees_v_in) - 1).item())
        if max_degree_v is not None:
            if max_degree_v > max_degree_v_present:
                raise ValueError(
                    f"`max_degree_v = {max_degree_v}` must not be larger than maximal degree present in the value "
                    f"vector = {max_degree_v_present}. "
                )

        max_degree_qk_present = int(np.rint(np.sqrt(num_degrees_qk_in) - 1).item())
        if max_degree_qk is not None:
            if max_degree_qk > max_degree_qk_present:
                raise ValueError(
                    f"`max_degree_qk = {max_degree_qk}` must not be larger than maximal degree present in the value "
                    f"vector = {max_degree_qk_present}. "
                )

        # is max_degree_qk is not present, default to maximal degree of query and key
        if max_degree_qk is None:
            max_degree_qk = max_degree_qk_present

        # is max_degree_v is not present, default to maximal degree of value
        if max_degree_v is None:
            max_degree_v = max_degree_v_present

        # check if query, key and value have pseudotensors
        pseudotensors_qk_present = num_parity_qk_in == 2
        pseudotensors_v_present = num_parity_v_in == 2

        q = utils.change_max_degree_or_type(
            q,
            include_pseudotensors=include_pseudotensors_qk,
            max_degree=max_degree_qk
        )

        k = utils.change_max_degree_or_type(
            k,
            include_pseudotensors=include_pseudotensors_qk,
            max_degree=max_degree_qk
        )

        v = utils.change_max_degree_or_type(
            v,
            include_pseudotensors=include_pseudotensors_v,
            max_degree=max_degree_v
        )

        print ("fresh qkv", q.shape, k.shape, v.shape)

        print (pos.shape, grid_u.shape)

        pos_proj = torch.einsum("bnd,bmd->bnm", pos, grid_u) # [B, N, N_pts]
        print ("pos_proj", pos_proj.shape)
        sin, cos = calculate_rotary_position_embedding(pos_proj, theta) # [B, N, N_pts, dim_qk]

        q = apply_rotary_position_embedding(q, sin, cos)  # (N, M, P, L, num_features_qk)
        k = apply_rotary_position_embedding(k, sin, cos)  # (N, M, P, L, num_features_qk)
        
        print ("qkv bef", q.shape, k.shape)

        q = q.reshape(*q.shape[:-3], -1) # (B, N, M, dim_qk)
        k = k.reshape(*k.shape[:-3], -1) # (B, N, M, dim_qk)
        v = v.reshape(*v.shape[:-3], -1) # (B, N, dim_v)

        q = q / torch.sqrt(torch.tensor(q.shape[-1]))

        num_parity_v, num_degrees_v = v.shape[-3], v.shape[-2]
        num_features_v = v.shape[-1]

        print ("qkv aft", q.shape, k.shape, v.shape)
        print ("misc shape:", num_parity_v, num_degrees_v)

        kv = torch.einsum(
            "bnmk,bnv->bnmkv",
            k,
            v
        )

        print (kv.shape)

        y = torch.einsum(
                'bnmd,bnmdv,bm->bnv',
                q,
                kv,
                grid_w
            )
        
        print (y.shape)

        y = torch.reshape(y, (*y.shape[:-1], num_parity_v, num_degrees_v, num_features_v))
        
        return None
        """

    def forward(self, q, k, v, pos, grid_u, grid_w, theta=None, max_degree_qk=None, max_degree_v=None, include_pseudotensors_qk=False, include_pseudotensors_v=False):
        """
        q : query node features [B, N, P, (L_qk + 1)**2, dim_qk]
        k : key node features [B, N, P, (L_qk + 1)**2, dim_qk]
        v : value node features [B, N, P, (L_v + 1)**2, dim_v]
        pos : node coordinates [B, N, 3]
        grid_u : grid points on unit sphere for Lebedev quadrature [B, N_omega, 3]
        grid_w : corresponding weights for Lebedev quadrature [B, N_omega]
        include_pseudotensors_qk: Include pseudotensors from query and key. 
        include_pseudotensors_v: Include pseudotensors from value.
        theta : rotation frequencies for RoPE [K]
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