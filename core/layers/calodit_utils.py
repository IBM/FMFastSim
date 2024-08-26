#
# Copyright (c) 2024 by Contributors for FMFastSim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(
            self,
            img_size: list = [45,16,9],
            patch_size: list = [2,2,2],
            embed_dim: int = 1024,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])   
        self.left_over = tuple([((g + 1) * p - s) % p for p, s, g in zip(self.patch_size, self.img_size, self.grid_size)]) 
        self.grid_size = tuple([(s + l ) // p for p, l, s in zip(self.patch_size, self.left_over, self.img_size)]) 
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2] #3D

        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        B, C, Z, PHI, R = x.shape
        x = F.pad(x, (0, self.left_over[2], 0, self.left_over[1], 0, self.left_over[0])) #add padding if needed (padding is inversed)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NC Z PHI R -> NLC (N, 704, 1024)
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0 #half goes to cos half goes to sin
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0 # each third goes to one direction X,Y,Z

    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_phi = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_r = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_z, emb_phi, emb_r], axis=1) # (X*Y*Z, D)
    return emb


def get_3d_sincos_pos_embed(embed_dim = 1024, grid_size = [22,8,4]):
    """
    grid_size: list of the grid after patchify (convolution)
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] 
    """
    # print('grid_size:', grid_size)

    grid_z = np.arange(grid_size[0], dtype=np.float32)
    grid_phi = np.arange(grid_size[1], dtype=np.float32)
    grid_r = np.arange(grid_size[2], dtype=np.float32)

    grid = np.meshgrid(grid_z, grid_phi, grid_r, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed


class AttentionDiT(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, time_emb_dim=None, **block_kwargs):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionDiT(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(self.modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = [bias, bias]
        drop_probs = [drop, drop]
        # linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
