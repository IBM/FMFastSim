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

import sys
sys.path.append('/u/kyeo/FMTS/tsfm')

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers.layer import layer
from core.layers.tsbeastbase import TSBeastBaseDecoderWithReconstructionHead, TSBeastBaseModel, TSBeastBaseConfig

import einops

class TSMixBeast(layer):
    def __init__(self,
        dim_r = 18,
        dim_a = 50,
        dim_v = 45,
        dim_c = 4,
    #TSMixBeast parameters
        channel_dim        = 'r',
        patch_dim          = 'v',
        expansion_factor   = 2,
        num_layers         = 4,
        dropout            = 0.0,
        mode               = "mix_channel",  # either common_channel,  mix_channel
        gated_attn         = True,
        norm_mlp           = "LayerNorm",
        head_dropout       = 0.0,
        scaling            = "none",
        use_positional_encoding=False,
        positional_encoding_type='random',
        self_attn           = False,
        self_attn_heads     = 1,
        decoder_num_layers  = 4,
        decoder_mode        = "mix_channel",
        reconstruction_type = "patchwise",
        #encoder compression
        d_model_layerwise_scale             = [1, 0.75, 0.5, 0.5],
        num_patches_layerwise_scale         = [1, 0.75, 0.5, 0.5],
        num_channels_layerwise_scale        = [1, 0.75, 0.5, 0.5],
        #decoder decompression
        decoder_d_model_layerwise_scale     = [0.5, 0.5, 0.75, 1],
        decoder_num_patches_layerwise_scale = [0.5, 0.5, 0.75, 1],
        decoder_num_channels_layerwise_scale= [0.5, 0.5, 0.75, 1],
        #residual connection
        encoder_resconn = True,
        decoder_resconn = True,
        #variational auto-encoder
        variational=False,
    #Output
        lower_bound=None,
        output_scale = 1.0,
        output_bias  = 0.0,
        ):

        super().__init__()

        self.dim_v = dim_v
        self.dim_a = dim_a
        self.dim_r = dim_r
        self.dim_c = dim_c

        length_dim = {'a':dim_a,'v':dim_v,'r':dim_r}

        context_length = 1
        for i in [k for k in ['a','v','r'] if k != channel_dim]:
            context_length *= length_dim[i]

        patch_length       = length_dim[  patch_dim]
        patch_stride       = length_dim[  patch_dim]
        num_input_channels = length_dim[channel_dim]

        first_dim  = [k for k in ['a','v','r'] if (k != patch_dim) and (k != channel_dim)][0]
        second_dim = patch_dim
        third_dim  = channel_dim

        self.enc_dim = [length_dim[first_dim],length_dim[second_dim],length_dim[third_dim]]
        self.enc_arrange = f'b r a v -> b ({first_dim} {second_dim}) {third_dim}'
        self.dec_arrange = f'b {first_dim} {second_dim} {third_dim} -> b r a v'

        self.lower_bound = lower_bound
        self.variational = variational

        self.output_scale = output_scale
        self.output_bias  = output_bias

        config = TSBeastBaseConfig(
            context_length=context_length,
            patch_length=patch_length,
            num_input_channels=num_input_channels,
            patch_stride=patch_stride,
            expansion_factor=expansion_factor,
            num_layers=num_layers,
            dropout=dropout,
            mode=mode,
            gated_attn=gated_attn,
            norm_mlp=norm_mlp,
            head_dropout=head_dropout,
            scaling="none", #scaling is not appplied outside of the model
            use_positional_encoding=use_positional_encoding,
            positional_encoding_type=positional_encoding_type,
            self_attn=self_attn,
            self_attn_heads=self_attn_heads,
            decoder_num_layers=decoder_num_layers,
            decoder_mode=decoder_mode,
            d_model_layerwise_scale=d_model_layerwise_scale,
            decoder_d_model_layerwise_scale=decoder_d_model_layerwise_scale,
            num_patches_layerwise_scale=num_patches_layerwise_scale,
            decoder_num_patches_layerwise_scale=decoder_num_patches_layerwise_scale,
            num_channels_layerwise_scale=num_channels_layerwise_scale,
            decoder_num_channels_layerwise_scale=decoder_num_channels_layerwise_scale,
            d_model=patch_length,          #d_model is set to patch_length
            decoder_d_model=patch_length,  #d_model is set to patch_length
            encoder_resconn=encoder_resconn,
            decoder_resconn=decoder_resconn,
            variational=variational,
        )

        config.check_and_init_preprocessing()

        self.encoder = TSBeastBaseModel(config)
        self.decoder = TSBeastBaseDecoderWithReconstructionHead(config)

        test_data = torch.rand(1,context_length,num_input_channels)
        enc_out   = self.encoder(test_data)

        enc_emb_size = dim_r*dim_v*dim_a
        dec_emb_size = enc_out.last_hidden_flatten_state.size(-1)

        self.decoder_input = torch.zeros(1,dec_emb_size)

        self.enc_pos_emb = nn.Sequential(nn.Linear(dim_c,128),nn.SiLU(),
                                         #nn.LayerNorm(128),
                                         nn.Linear(128,128),nn.SiLU(),
                                         nn.Linear(128,enc_emb_size))

        self.enc_scale_emb = nn.Sequential(nn.Linear(dim_c,128),nn.SiLU(),
                                           #nn.LayerNorm(128),
                                           nn.Linear(128,128),nn.SiLU(),
                                           nn.Linear(128,enc_emb_size))

        self.dec_pos_emb = nn.Sequential(nn.Linear(dim_c,128),nn.SiLU(),
                                         #nn.LayerNorm(128),
                                         nn.Linear(128,128),nn.SiLU(),
                                         nn.Linear(128,dec_emb_size))

        self.dec_scale_emb = nn.Sequential(nn.Linear(dim_c,128),nn.SiLU(),
                                           #nn.LayerNorm(128),
                                           nn.Linear(128,128),nn.SiLU(),
                                           nn.Linear(128,dec_emb_size))

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def encoding(self,x_in,c_in=None,sampling=False):

        x0 = einops.rearrange(x_in,self.enc_arrange)

        nb = x0.size(0)
        nc = x0.size(-1)

        if c_in != None:
            c0   = self.  enc_pos_emb(c_in).reshape(nb,-1,nc)
            c1   = self.enc_scale_emb(c_in).reshape(nb,-1,nc)
    
            x0 = c0 + x0*(c1+1)

        enc_out = self.encoder(x0)

        if self.variational:
            z_exp = enc_out.mu_hidden_flatten_state
            z_var = enc_out.log_var_hidden_flatten_state.exp()
            if sampling:
                return z_exp + torch.randn_like(z_var)*z_var.sqrt()
            else:
                return z_exp, z_var
        else:
            return enc_out.last_hidden_flatten_state

    #Z_in : input data in the dimension of Batch x Embedding
    def decoding(self,z_in,c_in=None):

        if c_in != None:
            c0   = self.  dec_pos_emb(c_in)
            c1   = self.dec_scale_emb(c_in)

            z_in = c0 + z_in*(c1+1)

        dec_out = self.decoder(decoder_input=z_in)

        x0 = dec_out.reconstruction_outputs*self.output_scale + self.output_bias
        #x0 is in dimension Batch x (Azimuthal x Vertical) x Radial

        nb = x0.size(0)
        nc = x0.size(-1)
        x0 = x0.reshape(nb,self.enc_dim[0],self.enc_dim[1],self.enc_dim[2])
  
        x0 = einops.rearrange(x0,self.dec_arrange)

        if self.lower_bound != None:
            x0 = F.relu(x0-self.lower_bound)+self.lower_bound

        return x0

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        z_in= self.encoding(x_in,c_in,sampling=True)
        out = self.decoding(z_in,c_in)
        return out
