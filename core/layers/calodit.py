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

import torch
import torch.nn as nn

from core.layers.calodit_utils import PatchEmbed, get_3d_sincos_pos_embed, DiTBlock

class CaloDiT(nn.Module):
    """
    Class template for models.

    1. Together with the neural network parameters, layer class should have "variational" (bool).
       When "variational==True", the latent space of the model should be a normal distribution.
       When "variational==False", the model has a deterministic latent variable

    2. __init__ should define an array "self.decoder_input". 
       "decoder_input" is a tensor, whose size is torch.zeros(1,size of input variable of decoder)
    """
    def __init__(self, variation=False,
                       dim_r = 18,         
                       dim_a = 50,          
                       dim_v = 45,          
                       dim_c = 128,  # time and conditions (not valid for variational)
                       mlp_ratio = 4,
                       num_heads = 8,
                       embed_dim = 144,
                       num_enc_layers = 2,
                       num_dec_layers = 2,
                       patch_size = [3, 2, 3],
                       variational = False
                       ):
        super().__init__()

        self._name = 'CaloDiT'

        self.dim_r = dim_r
        self.dim_a = dim_a
        self.dim_v = dim_v

        self.dim_c = dim_c

        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.patch_size = patch_size

        self.variational = variational

        self.encoder = nn.ModuleList([DiTBlock(self.embed_dim,
                                               self.num_heads,
                                               mlp_ratio=self.mlp_ratio,
                                               time_emb_dim=self.dim_c) 
                                     for _ in range(self.num_enc_layers)])

        self.decoder = nn.ModuleList([DiTBlock(self.embed_dim,
                                               self.num_heads,
                                               mlp_ratio=self.mlp_ratio,
                                               time_emb_dim=self.dim_c) 
                                     for _ in range(self.num_dec_layers)])

        self.final_layer = nn.Sequential(
            nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(self.embed_dim, self.embed_dim, bias=True),
            nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6),
            nn.Linear(self.embed_dim, self.patch_size[0] * self.patch_size[1] * self.patch_size[2], bias=True),
        )

        if self.variational:
            raise NotImplementedError

        self.decoder_input = torch.zeros(1,3)  # recheck

        self.patch_embedder = PatchEmbed([self.dim_r, self.dim_a, self.dim_v],
                                         self.patch_size, self.embed_dim)
        
        # positional embeddings for patches of shower
        num_patches = self.patch_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim), requires_grad=False)
        pos_embed = get_3d_sincos_pos_embed(self.embed_dim, self.patch_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))



    """
    encoding function takes an input variable (x_in,c_in) and returns the latent variable.
    If (self.variational == True) and (sampling == False), encoding returns mean and variance 
    If (self.variational == True) and (sampling == True ), encoding returns a sample from mean and variance 
    If self.varional == False, encoding returns a deterministic latent variable
    """
    def encoding(self,x_in,c_in=None,sampling=True):

        patches = self.patch_embedder(x_in)
        z0 = patches + self.pos_embed

        for block in self.encoder:
            z0 = block(z0,c_in)

        if self.variational:
            z_mean = self.exp_net(z0)
            z_var  = self.var_net(z0).exp()

            if sampling:
                return z_mean + torch.randn_like(z_var)*z_var.sqrt()
            else:
                return z_mean, z_var
        else:
            return z0

    """
    decoding function takes a latent variable (z_in,c_in) and returns the output variable.
    """
    def decoding(self,z_in,c_in=None):
        for block in self.decoder:
            z_in = block(z_in,c_in)

        out = self.final_layer(z_in)
        out = self.unpatchify(out)
        return out

    """
    forward function combines encoding and decoding.
    If self.variational == True, forward function is based on the sampling
    """
    def forward(self,x_in,c_in=None):
        z_in = self.encoding(x_in,c_in,sampling=True)
        out  = self.decoding(z_in,c_in)
        return out
    
    def unpatchify(self, x):
        """
        input: (N, T, patch_size[0] * patch_size[1] * patch_size[2])    (N, 704, 2*2*2)
        voxels: (N, Z, PHI, R)          (N, 45, 16, 9)
        """
        p_r = self.patch_embedder.patch_size[0]
        p_phi = self.patch_embedder.patch_size[1]
        p_z = self.patch_embedder.patch_size[2]
        r_lo = self.patch_embedder.left_over[0]
        phi_lo = self.patch_embedder.left_over[1]
        z_lo = self.patch_embedder.left_over[2]
        r = self.patch_embedder.grid_size[0]
        phi = self.patch_embedder.grid_size[1]
        z = self.patch_embedder.grid_size[2]
        assert r * phi * z == x.shape[1]

        x = x.reshape(shape=(x.shape[0], r, phi, z, p_r, p_phi, p_z))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], r * p_r, phi * p_phi, z  * p_z))

        imgs = imgs[:,:r * p_r - r_lo, :phi * p_phi - phi_lo, :z * p_z - z_lo]
        return imgs
