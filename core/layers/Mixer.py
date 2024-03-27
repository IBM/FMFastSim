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
from core.layers.MixerTF import Cond_Net
from core.layers.lib_mixer import Mixer3D, init_weights

#
# MLPMixer
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
# dim_v_enc: (list of int) list of vertical encoding dimension. 
#
# the input arguments below defines the decoder structure. The list defines the number of dimensions of a block.
# dim_r_dec: (list of int) list of radial encoding dimension
# dim_a_dec: (list of int) list of azimuthal encoding dimension
# dim_v_dec: (list of int) list of vertical encoding dimension. 
#
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

class Mixer(layer):
    def __init__(self,dim_r = 18,
                      dim_a = 50,
                      dim_v = 45,
                      dim_c = 4,
                      dim_r_enc = [18, 8], #list of radial encoding dimension
                      dim_a_enc = [50,16], #list of azimuthal encoding dimensiona
                      dim_v_enc = [45,45], #list of azimuthal encoding dimensiona
                      dim_r_dec = [ 8,18], #list of radial decoding dimension
                      dim_a_dec = [16,50], #list of azimuthal decoding dimension
                      dim_v_dec = [45,45], #list of azimuthal decoding dimension
                      mlp_ratio  = 4,
                      mlp_layers = 2,
                      res_conn   = False, #Use ResNet Connection
                      lower_bound= None,
                      activation = 'SiLU',
                      output_scale = 1.0,
                      output_bias  = 0.0,
                      final_layer = False,
                      variational = False,
                      gated_attn = False,
                      ):

        super().__init__()

        self._name = 'Mixer'

        self.output_scale = output_scale
        self.output_bias  = output_bias

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

        activation = getattr(nn,activation)

        self.encoder = Mixer_Core(dim_v=dim_v_enc,dim_r=dim_r_enc,dim_a=dim_a_enc,dim_c=dim_c,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=res_conn,activation=activation,gated_attn=gated_attn)
        self.decoder = Mixer_Core(dim_v=dim_v_dec,dim_r=dim_r_dec,dim_a=dim_a_dec,dim_c=dim_c,mlp_ratio=mlp_ratio,mlp_layers=mlp_layers,res_conn=res_conn,activation=activation,decoder=final_layer,gated_attn=gated_attn)

        #Variational Inference
        d0 = [dim_r_enc[-1],dim_r_enc[-1]]
        d1 = [dim_a_enc[-1],dim_a_enc[-1]]
        d2 = [dim_v_enc[-1],dim_v_enc[-1]]

        if self.variational:
            self.exp_net = Mixer3D(dim_x0=d0,dim_x1=d1,dim_x2=d2,mlp_ratio=4,mlp_layers=2,activation=activation,res_conn=False,gated_attn=gated_attn)
            self.var_net = Mixer3D(dim_x0=d0,dim_x1=d1,dim_x2=d2,mlp_ratio=4,mlp_layers=2,activation=activation,res_conn=False,gated_attn=gated_attn)
            self.exp_net.apply(init_weights)
            self.var_net.apply(init_weights)

        self.decoder_input = torch.zeros(1,dim_r_dec[0],dim_a_dec[0],dim_v_dec[0])

        self.lower_bound = lower_bound

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def encoding(self,x_in,c_in=None,sampling=True):

        z0 = self.encoder(x_in,c_in)

        if self.variational:
            z_exp = self.exp_net(z0)
            z_var = self.var_net(z0).exp()

            if sampling:
                return z_exp + torch.randn_like(z_var)*z_var.sqrt()
            else:
                return z_exp, z_var
        else:
            return z0

    #Z_in : input data in the dimension of Batch x Radial x Aximuthal x Vertical
    def decoding(self,z_in,c_in=None):

        x0 = self.decoder(z_in,c_in)*self.output_scale + self.output_bias

        if self.lower_bound != None:
            x0 = F.relu(x0-self.lower_bound)+self.lower_bound

        return x0

    #X_in : input data in the dimension of Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):
        z_in= self.encoding(x_in,c_in,sampling=True)
        out = self.decoding(z_in,c_in)
        return out

class Mixer_Core(nn.Module):
    def __init__(self,dim_v,dim_r,dim_a,dim_c,mlp_ratio,mlp_layers,res_conn,activation,decoder=False,gated_attn=False):
        super().__init__()

        module = []
        cond_net = []
        for i in range(len(dim_r)-1):
            module += [Mixer3D(dim_x0     = dim_r[i:i+2], 
                               dim_x1     = dim_a[i:i+2], 
                               dim_x2     = dim_v[i:i+2], 
                               mlp_ratio  = mlp_ratio,    
                               activation = activation,   
                               gated_attn = gated_attn,   
                               mlp_layers = mlp_layers,
                               res_conn   = res_conn)]

            cond_v = [8,16,dim_v[i]]
            cond_r = [8,16,dim_r[i]]
            cond_a = [8,16,dim_a[i]]
            cond_net += [Cond_Net(cond_r,cond_a,cond_v,dim_c)]

        self.module   = nn.ModuleList(module)
        self.cond_net = nn.ModuleList(cond_net)

        self.module  .apply(lambda m: init_weights(m,gain=1.0))
        self.cond_net.apply(lambda m: init_weights(m,gain=1.4))

        if decoder:
            self.decoder = Mixer3D(dim_x0 = [dim_r[-1],dim_r[-1]],
                                   dim_x1 = [dim_a[-1],dim_a[-1]],
                                   dim_x2 = [dim_v[-1],dim_v[-1]],
                                   mlp_ratio  = mlp_ratio,
                                   mlp_layers = mlp_layers,
                                   activation = activation,
                                   gated_attn = False,
                                   res_conn   = False)
            self.decoder.apply(init_weights)
        else:
            self.decoder = None

    #X_in : input of dimension Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in=None):

        z0 = x_in
        for i in range(len(self.module)):
            if c_in != None:
                pos_emb,scale_emb = self.cond_net[i](c_in)
                shift = pos_emb + z0*scale_emb
            else:
                shift = 0.0

            z0 = self.module[i](z0,shift)

        if self.decoder != None:
            z0 = self.decoder(z0)

        return z0

