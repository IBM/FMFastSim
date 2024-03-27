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
import torch.nn.functional as F

from core.layers.layer import layer
from core.layers.lib_mixer import Mixer2D, Mixer3D, MixerTF_Block,init_weights

#
# Mixer Transformer
# Multihead Attention + MLPMixer
# Multihead Attention is applied on the vertical direction and
# MLPMixer is used for the other two directionss (Radial and Azimuthal directions)
# 
# input arguments
#
# dim_r: (int) dimension in the radial direction
# dim_a: (int) dimension in the azimuthal direction
# dim_v: (int) dimension in the vertical direction
# dim_c: (int) dimension of the conditioning variables, e.g. energy(1) + angle(1) + geometry(2) = 4
#
# the input arguments below defines the encoder structure. The list defines the number of dimensions of a block.
# For example, [50,30,20] indicates there are two MixerTF blocks; dim(50)->Block->dim(30) => dim(30)->Block->dim(20)
# dim_r_enc: (list of int) list of radial encoding dimension
# dim_a_enc: (list of int) list of azimuthal encoding dimension
# dim_v_enc: (list of int) list of vertical encoding dimension. It can be variable, only when mixer_ddim = 3
#
# the input arguments below defines the decoder structure. The list defines the number of dimensions of a block.
# dim_r_dec: (list of int) list of radial encoding dimension
# dim_a_dec: (list of int) list of azimuthal encoding dimension
# dim_v_dec: (list of int) list of vertical encoding dimension. It can be variable, only when mixer_ddim = 3
#
# number of the attention heads. The output of the vertical MixerTF should be divided evenly by the heads.
# For exmaple, if dim_v_enc = [45,30,20], num_enc_heads should divide evenly [30,20]
# num_enc_heads: (list of int) number of attention heads in encoder
# num_dec_heads: (list of int) number of attention heads in decoder
#
# mixer_dim: (int) 2 or 3. Use 2-D Mixer or 3-D Mixer. Only 3-D Mixer supports variable dimension in the vertical direction
# mlp_ratio: (int) expansion ratio of the mixer. 
# mlp_layers: (int) number of mlp mixer layers per each block
# res_conn: (bool) add a resudial connection to the model
# lower_bound: (float) if set a lower bound of the output. y = Relu(x-lb)+lb. If None, not enforcing a lower bound
# activation: (str) activation function, e.g., 'SiLU', 'ReLU', 'Tanh', 'Sigmoid', ...
# output_scale: (float) output transformation: y = output_scale*x + output_bias
# output_bias: (float) output transformatin: y = output_scale*x + output_bias
# final_layer: (bool) add a final MixerLayer to the MixerTF blocks
# variational: (bool) use normal distribution for the latent variable
# gated_attn: (bool) use a gated attention for the mixer block
#

class MixerTF(layer):
    def __init__(self,dim_r = 18,         
                      dim_a = 50,          
                      dim_v = 45,          
                      dim_c = 4,          
                      dim_r_enc = [18, 8],
                      dim_a_enc = [50,16],
                      dim_v_enc = [45,45],
                      dim_r_dec = [ 8,18], #list of radial decoding dimension
                      dim_a_dec = [16,50], #list of azimuthal decoding dimension
                      dim_v_dec = [45,45], #list of azimuthal decoding dimension
                      num_enc_heads  = [10], #list of number of heads of encoding transformer
                      num_dec_heads  = [ 8], #list of number of heads of decoding transformer
                      mixer_dim  = 3,
                      mlp_ratio  = 4,
                      mlp_layers = 2,
                      res_conn   = False, #Use ResNet Connection
                      lower_bound= None,
                      activation = 'SiLU',
                      output_scale = 1.0,
                      output_bias  = 0.0,
                      final_layer = True,
                      variational = False,
                      gated_attn = False,
                      ):

        super().__init__()

        self._name = 'Mixer Transformer'

        self.dim_r = dim_r
        self.dim_a = dim_a
        self.dim_v = dim_v

        self.dim_c = dim_c

        self.dim_r_enc = dim_r_enc
        self.dim_r_dec = dim_r_dec

        self.dim_a_enc = dim_a_enc
        self.dim_a_dec = dim_a_dec

        self.dim_v_enc = dim_v_enc
        self.dim_v_dec = dim_v_dec

        self.variational = variational

        self.output_scale = output_scale
        self.output_bias  = output_bias

        activation = getattr(nn,activation)

        self.encoder = MixerTF_Core(dim_v=dim_v_enc,dim_r=dim_r_enc,dim_a=dim_a_enc,dim_c=dim_c,num_heads=num_enc_heads,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=res_conn,mixer_dim=mixer_dim,activation=activation,gated_attn=gated_attn)
        self.decoder = MixerTF_Core(dim_v=dim_v_dec,dim_r=dim_r_dec,dim_a=dim_a_dec,dim_c=dim_c,num_heads=num_dec_heads,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=res_conn,mixer_dim=mixer_dim,activation=activation,decoder=final_layer,gated_attn=gated_attn)

        #Variational Inference
        d0 = [dim_r_enc[-1],dim_r_enc[-1]]
        d1 = [dim_a_enc[-1],dim_a_enc[-1]]

        if self.variational:
            self.exp_net = Mixer2D(dim_x0=d0,dim_x1=d1,mlp_ratio=4,mlp_layers=2,activation=activation,res_conn=False,gated_attn=gated_attn)
            self.var_net = Mixer2D(dim_x0=d0,dim_x1=d1,mlp_ratio=4,mlp_layers=2,activation=activation,res_conn=False,gated_attn=gated_attn)
            #self.exp_net.apply(lambda m: init_weights(m,gain=1.0))
            #self.var_net.apply(lambda m: init_weights(m,gain=1.0))

        self.decoder_input = torch.zeros(1,dim_v_dec[0],dim_r_dec[0],dim_a_dec[0])

        self.lower_bound = lower_bound

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def encoding(self,x_in,c_in=None,sampling=False):

        x0 = x_in.permute(0,3,1,2) #Batch x Vertical x Radial x Azimuthal
        z0 = self.encoder(x0,c_in)

        if self.variational:
            z_exp = self.exp_net(z0)
            z_var = self.var_net(z0).exp()

            if sampling:
                return z_exp + torch.randn_like(z_var)*z_var.sqrt()
            else:
                return z_exp, z_var
        else:
            return z0

    #Z_in : input data in the dimension of Batch x Vertical x Radial x Aximuthal
    def decoding(self,z_in,c_in=None):

        x0 = self.decoder(z_in,c_in)*self.output_scale + self.output_bias
        x0 = x0.permute(0,2,3,1) #Batch x Radial x Azimuthal x Vertical

        if self.lower_bound != None:
            x0 = F.relu(x0-self.lower_bound) + self.lower_bound

        return x0

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        z_in= self.encoding(x_in,c_in,sampling=True)
        out = self.decoding(z_in,c_in)
        return out

class MixerTF_Core(nn.Module):
    def __init__(self,dim_v,dim_r,dim_a,dim_c,num_heads,mixer_dim,mlp_ratio,mlp_layers,res_conn,activation,decoder=False,gated_attn=True):
        super().__init__()

        module = []
        cond_net = []
        for i in range(len(dim_r)-1):
            dim_in  = [dim_v[i  ],dim_r[i  ],dim_a[i  ]]
            dim_out = [dim_v[i+1],dim_r[i+1],dim_a[i+1]]
            module += [MixerTF_Block(dim_in  = dim_in,         
                                     dim_out = dim_out,        
                                     mixer_dim = mixer_dim,    
                                     mlp_ratio  = mlp_ratio,   
                                     mlp_layers = mlp_layers,  
                                     num_heads = num_heads[i], 
                                     activation = activation,  
                                     gated_attn = gated_attn,  
                                     res_conn = res_conn)]

            cond_v = [8,16,dim_v[i]]
            cond_r = [8,16,dim_r[i]]
            cond_a = [8,16,dim_a[i]]
            cond_net += [Cond_Net(cond_v,cond_r,cond_a,dim_c)]

        self.module   = nn.ModuleList(module)
        self.cond_net = nn.ModuleList(cond_net)

        if decoder:
            self.decoder = Mixer2D(dim_x0 = [dim_r[-1],dim_r[-1]],
                                   dim_x1 = [dim_a[-1],dim_a[-1]],
                                   mlp_ratio  = mlp_ratio,
                                   mlp_layers = mlp_layers,
                                   activation = activation,
                                   gated_attn = False,
                                   res_conn = False)
            #self.decoder.apply(lambda m: init_weights(m,gain=1.0))
        else:
            self.decoder = None

        #self.module  .apply(lambda m: init_weights(m,gain=1.0))
        self.cond_net.apply(lambda m: init_weights(m,gain=1.4))

    #X_in : input of dimension Batch x Vertical x Radial x Azimuthal
    def forward(self,x_in,c_in=None):

        z0 = x_in
        for i in range(len(self.module)):
            if c_in != None:
                pos_emb,scale_emb = self.cond_net[i](c_in)
                shift = pos_emb+scale_emb*z0
            else:
                shift = 0.0

            z0 = self.module[i](z0,shift)

        if self.decoder != None:
            z0 = self.decoder(z0)

        return z0

class Cond_Net(nn.Module):
    def __init__(self,dim_x0,dim_x1,dim_x2,dim_c,res_conn=True,gated_attn=False):
        super().__init__()

        self.dim_x0 = dim_x0
        self.dim_x1 = dim_x1
        self.dim_x2 = dim_x2

        self.dim_c = dim_c

        d_model = dim_x0[0]*dim_x1[0]*dim_x2[0]

        self.linear_pos   = nn.Sequential(nn.Linear(dim_c,32),nn.SiLU(),nn.Linear(32,d_model))
        self.linear_scale = nn.Sequential(nn.Linear(dim_c,32),nn.SiLU(),nn.Linear(32,d_model))

        module = []
        for i in range(len(dim_x0)-1):
            module += [Mixer3D(dim_x0 = dim_x0[i:i+2],
                               dim_x1 = dim_x1[i:i+2],
                               dim_x2 = dim_x2[i:i+2],
                               mlp_ratio  = 4,
                               mlp_layers = 2,
                               activation=nn.SiLU,
                               gated_attn=gated_attn,
                               res_conn=res_conn)]

        self.pos_mixer = nn.ModuleList(module)

        module = []
        for i in range(len(dim_x0)-1):
            module += [Mixer3D(dim_x0 = dim_x0[i:i+2],
                               dim_x1 = dim_x1[i:i+2],
                               dim_x2 = dim_x2[i:i+2],
                               mlp_ratio  = 4,
                               mlp_layers = 2,
                               activation=nn.SiLU,
                               gated_attn=gated_attn,
                               res_conn=res_conn)]

        self.scale_mixer = nn.ModuleList(module)

    def forward(self,c_in,scale=True):
        nb    = c_in.size(0)

        d0 = self.dim_x0[0]
        d1 = self.dim_x1[0]
        d2 = self.dim_x2[0]

        pos_emb = self.linear_pos(c_in).view(nb,d0,d1,d2)
        for i in range(len(self.pos_mixer)):
            pos_emb = self.pos_mixer[i](pos_emb)

        if scale:
            scale_emb = self.linear_scale(c_in).view(nb,d0,d1,d2)
            for i in range(len(self.pos_mixer)):
                scale_emb = self.scale_mixer[i](scale_emb)

            return pos_emb, scale_emb
        else:
            return pos_emb
