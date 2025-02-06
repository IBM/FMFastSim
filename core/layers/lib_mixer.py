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

#Mixer for 2-D tensor
#dim_x0: (list) [input_dim,output_dim] of the first dimension of the mixer
#dim_x1: (list) [input_dim,output_dim] of the second diemsion of the mixer
#mlp_ratio: expansion ratio of the mlp in x0 and x1
#mlp_layers: number of mixer layers
#actication: activation function
#dropout: dropout rate (for future use. not implemented)
class Mixer2D(nn.Module):
    def __init__(self,dim_x0=[1,1],dim_x1=[1,1],mlp_ratio=4,mlp_layers=2,activation=nn.SiLU,dropout=0.0,res_conn=True,gated_attn=False,norm='layer'):
        super().__init__()

        self.res_conn = res_conn

        check_dim = (dim_x0[0] == dim_x0[1]) and (dim_x1[0] == dim_x1[1])
        
        if check_dim:
            self.adapter = None
        else:
            self.adapter = nn.ModuleList([Adapter(dim_x1[0],dim_x1[1]),
                                          Adapter(dim_x0[0],dim_x0[1])])

        d0 = dim_x0[1]
        d1 = dim_x1[1]

        self.x0_net = build_mixer_block([d1,d0],activation,mlp_ratio,mlp_layers,gated_attn,norm=norm)
        self.x1_net = build_mixer_block([d0,d1],activation,mlp_ratio,mlp_layers,gated_attn,norm=norm)

    #input x_in: tensor of which last two dimenisons are ..... x dim_x0[0] x dim_x1[0]
    #output    : tensor of the same dimension with x_in
    def forward(self,x_in,shift=0.0):

        z0 = x_in + shift

        if self.adapter != None:
            z1 = self.adapter[0](z0).transpose(-1,-2)
            z0 = self.adapter[1](z1).transpose(-1,-2)
        
        for i in range(len(self.x0_net)):
            if self.res_conn:
                z1 = (z0+self.x1_net[i](z0)).transpose(-1,-2)
                z0 = (z1+self.x0_net[i](z1)).transpose(-1,-2)
            else:
                z1 =     self.x1_net[i](z0) .transpose(-1,-2)
                z0 =     self.x0_net[i](z1) .transpose(-1,-2)

        return z0
  
#Mixer for 3-D tensor
#dim_x0: (list) [input_dim,output_dim] of the first dimension of the mixer
#dim_x1: (list) [input_dim,output_dim] of the second diemsion of the mixer
#dim_x2: (list) [input_dim,output_dim] of the third diemsion of the mixer
#mlp_ratio: expansion ratio of the mlp
#mlp_layers: number of mixer layers
#actication: activation function
#dropout: dropout rate (for future use. not implemented)
class Mixer3D(nn.Module):
    def __init__(self,dim_x0=[1,1],dim_x1=[1,1],dim_x2=[1,1],mlp_ratio=4,mlp_layers=2,activation=nn.SiLU,dropout=0.0,res_conn=True,gated_attn=False,norm='layer'):
        super().__init__()

        self.res_conn = res_conn

        check_dim = (dim_x0[0] == dim_x0[1]) and \
                    (dim_x1[0] == dim_x1[1]) and \
                    (dim_x2[0] == dim_x2[1])

        if check_dim:
            self.adapter = None
        else:
            self.adapter = nn.ModuleList([Adapter(dim_x2[0],dim_x2[1]),
                                          Adapter(dim_x1[0],dim_x1[1]),
                                          Adapter(dim_x0[0],dim_x0[1])])

        d0 = dim_x0[1]
        d1 = dim_x1[1]
        d2 = dim_x2[1]

        self.x0_net = build_mixer_block([d1,d2,d0],activation,mlp_ratio,mlp_layers,gated_attn,norm=norm)
        self.x1_net = build_mixer_block([d2,d0,d1],activation,mlp_ratio,mlp_layers,gated_attn,norm=norm)
        self.x2_net = build_mixer_block([d0,d1,d2],activation,mlp_ratio,mlp_layers,gated_attn,norm=norm)

    #input x_in: tensor of which last two dimenisons are ..... x dim_x0[0] x dim_x1[0] x dim_x2[0]
    #output    : tensor of dimension                     ..... x dim_x0[1] x dim_x1[1] x dim_x2[1]
    def forward(self,x_in,shift=0.0):

        z0 = x_in + shift

        if self.adapter != None:
            z1 = self.adapter[0](z0).permute(0,3,1,2)
            z2 = self.adapter[1](z1).permute(0,3,1,2)
            z0 = self.adapter[2](z2).permute(0,3,1,2)

        for i in range(len(self.x0_net)):
            if self.res_conn:
                z1 = (z0+self.x2_net[i](z0)).permute(0,3,1,2)
                z2 = (z1+self.x1_net[i](z1)).permute(0,3,1,2)
                z0 = (z2+self.x0_net[i](z2)).permute(0,3,1,2)
            else:
                z1 =     self.x2_net[i](z0) .permute(0,3,1,2)
                z2 =     self.x1_net[i](z1) .permute(0,3,1,2)
                z0 =     self.x0_net[i](z2) .permute(0,3,1,2)
        
        return z0
  
#MLP for the flatten last two dimensions
#dim_x0: (list) [input_dim,output_dim] of the first dimension of the mixer
#dim_x1: (list) [input_dim,output_dim] of the second diemsion of the mixer
#mlp_ratio: expansion ratio of the mlp in x0 and x1
#mlp_layers: number of mixer layers
#actication: activation function
#dropout: dropout rate (for future use. not implemented)
class MLP_2D(nn.Module):
    def __init__(self,dim_x0=[1,1],dim_x1=[1,1],mlp_ratio=4,mlp_layers=2,activation=nn.SiLU,dropout=0.0,res_conn=True,gated_attn=False,norm='layer'):
        super().__init__()

        self.dim_x0   = dim_x0
        self.dim_x1   = dim_x1
        self.res_conn = res_conn

        check_dim = (dim_x0[0] == dim_x0[1]) and (dim_x1[0] == dim_x1[1])

        dim_in  = dim_x0[0]*dim_x1[0]
        dim_out = dim_x0[1]*dim_x1[1]

        if check_dim:
            self.adapter = None
        else:
            self.adapter = nn.Linear(dim_in,dim_out)

        self.x_net = build_mixer_block([dim_out],activation,mlp_ratio,mlp_layers,gated_attn,norm=norm)

    #input x_in: tensor of which last two dimenisons are ..... x dim_x0[0] x dim_x1[0]
    #output    : tensor of the same dimension with x_in
    def forward(self,x_in,shift=0.0):

        z0 = (x_in+shift).flatten(-2)

        if self.adapter != None:
            z0 = self.adapter(z0)

        for i in range(len(self.x_net)):
            if self.res_conn:
                z0 = z0+self.x_net[i](z0)
            else:
                z0 =    self.x_net[i](z0)
        
        nb = x_in.size(0)
        d0 = x_in.size(1)
        d1 = self.dim_x0[1]
        d2 = self.dim_x1[1]
        return z0.view(nb,d0,d1,d2)
  

# Multihead Attention + MLPMixer
# input tensor is a 4-D tensor Batch x dim_in
# dim_in : (list) dimension of the input tensor
# dim_out: (list) dimension of the output tensor
# Note that it is assumed that dim_in[0] == dim_in[1] always
# mlp_ratio: (float) expansion ratio of the mixer mlp
# mlp_layers: (int) number of mlp layers in mixer
# num_heads: (int) number of heads in the multihead attention - dim_in[1]*dim_in[2] should be divisible by num_heads
# dropout: (float) dropout ratio
# res_conn: (bool) use resnet connection or not
class MixerTF_Block(nn.Module):
    def __init__(self,dim_in=[1,1,1],dim_out=[1,1,1],mixer_dim=2,mlp_ratio=4,mlp_layers=2,num_heads=0,dropout=0.1,activation=nn.SiLU,res_conn=True,gated_attn=False,norm='layer'):
        super().__init__()

        self.dim_in  = dim_in
        self.dim_out = dim_out

        dim_x0 = [dim_in[0],dim_out[0]]
        dim_x1 = [dim_in[1],dim_out[1]]
        dim_x2 = [dim_in[2],dim_out[2]]

        check_dim = [True if dim_in[i]==dim_out[i] else False for i in range(3)]

        #mixer
        self.activation = activation
        if   mixer_dim == 1:
            self.mixer = MLP_2D        (dim_x1,dim_x2,mlp_ratio,mlp_layers,activation,dropout,res_conn,gated_attn,norm=norm)
        elif mixer_dim == 2:
            self.mixer = Mixer2D       (dim_x1,dim_x2,mlp_ratio,mlp_layers,activation,dropout,res_conn,gated_attn,norm=norm)
        elif mixer_dim == 3:
            self.mixer = Mixer3D(dim_x0,dim_x1,dim_x2,mlp_ratio,mlp_layers,activation,dropout,res_conn,gated_attn,norm=norm)

        #attention
        d_model = dim_in[1]*dim_in[2]
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        #later normalization
        self.norm1 = nn.LayerNorm(d_model,eps=1.e-5) 
        #self.norm2 = nn.LayerNorm(dim_in ,eps=1.e-5) 

        #self.norm1 = nn.BatchNorm1d(dim_in[0],eps=1.e-5) 
        #self.norm2 = nn.BatchNorm2d(dim_in[0],eps=1.e-5) 

    #input: tensor of dimension Batch x dim_in
    #output:tensor of dimension Batch x dim_out
    def forward(self,x_in,shift=0.0):

        d0 = self.dim_in[0]
        d1 = self.dim_in[1]
        d2 = self.dim_in[2]

        #apply attention network
        z0   = self.norm1((x_in+shift).flatten(-2))
        z1,_ = self.attn(z0,z0,z0,need_weights=False)
        z2   = (x_in.flatten(-2) + z1).view(-1,d0,d1,d2)

        #apply mixer
        z3   = self.mixer(z2)

        return z3


###########################################################
#   Mixer to deal with conditioning variable
###########################################################
class Cond_Net(nn.Module):
    def __init__(self,dim_x0,dim_x1,dim_x2,dim_c,res_conn=True,gated_attn=False,pos=True,mode='both',norm='layer'):
        super().__init__()

        self.dim_x0 = dim_x0
        self.dim_x1 = dim_x1
        self.dim_x2 = dim_x2

        self.dim_c = dim_c

        self.mode = mode

        d_model = dim_x0[0]*dim_x1[0]*dim_x2[0]

        self.embedding = nn.Sequential(nn.Linear(dim_c,128),nn.SiLU(),
                                       nn.LayerNorm(128,elementwise_affine=False, bias=False),
                                       nn.Linear(128,128),nn.SiLU(),
                                       nn.Linear(128,d_model))

        self.  pos_mixer = None
        self.scale_mixer = None

        if (self.mode=='both') or (self.mode=='pos'):
            module = []
            for i in range(len(dim_x0)-1):
                module += [Mixer3D(dim_x0 = dim_x0[i:i+2],
                                   dim_x1 = dim_x1[i:i+2],
                                   dim_x2 = dim_x2[i:i+2],
                                   mlp_ratio  = 3,
                                   mlp_layers = 3,
                                   activation=nn.SiLU,
                                   gated_attn=gated_attn,
                                   res_conn=res_conn,
                                   norm=norm)]

            self.pos_mixer = nn.Sequential(*module)

        if (self.mode=='both') or (self.mode=='scale'):
            module = []
            for i in range(len(dim_x0)-1):
                module += [Mixer3D(dim_x0 = dim_x0[i:i+2],
                                   dim_x1 = dim_x1[i:i+2],
                                   dim_x2 = dim_x2[i:i+2],
                                   mlp_ratio  = 3,
                                   mlp_layers = 3,
                                   activation=nn.SiLU,
                                   gated_attn=gated_attn,
                                   res_conn=res_conn,
                                   norm=norm)]

            self.scale_mixer = nn.Sequential(*module)

    def forward(self,c_in,scale=True):
        nb    = c_in.size(0)

        d0 = self.dim_x0[0]
        d1 = self.dim_x1[0]
        d2 = self.dim_x2[0]

        emb = self.embedding(c_in).view(nb,d0,d1,d2)

        if self.pos_mixer:
            pos_emb = self.pos_mixer(emb)

        if self.scale_mixer:
            scale_emb = self.scale_mixer(emb)

        if self.mode == 'both':
            return pos_emb, scale_emb
        else:
            out = pos_emb if self.mode == 'pos' else scale_emb
            return out

#####################
#   Utility layers
#####################
class Adapter(nn.Module):
    def __init__(self,x_in,x_out):
        super().__init__()

        if x_in != x_out:
            self.map = nn.Linear(x_in,x_out)
        else:
            self.map = nn.Identity()
    def forward(self,x_in):
        return self.map(x_in)

class GatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.
    This Module is adopted from HuggingFace TSBeastBase library

    Args:
        in_size (`int`): The input size.
    """

    def __init__(self, in_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, in_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


def build_mixer_block(dim,activation,mlp_ratio,mlp_layers,gated_attn,norm='layer'):

    d1 = dim[-1]
    d_mlp = int(d1*mlp_ratio)

    net = []
    for i in range(mlp_layers):
        mlp  = []
        if gated_attn:
            mlp += [GatedAttention(d1)]
        if norm == 'layer':
            mlp += [nn.LayerNorm(dim,elementwise_affine=False, bias=False)]
        mlp += [nn.Linear(d1,d_mlp),activation()]
        mlp += [nn.Linear(d_mlp,d1)]
        net += [nn.Sequential(*mlp)]

    return nn.ModuleList(net)


def set_bias(m,bias=0.0):
    if type(m) == nn.Linear:
        m.bias.data.fill_(bias)

def init_weights(m,gain=1,bias=[-0.05,0.05]):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight,gain=gain)
        if m.bias != None:
            nn.init.uniform_(m.bias, a=bias[0], b=bias[1])
