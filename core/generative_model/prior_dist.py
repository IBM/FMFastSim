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
import torch.distributions as distributions

class prior_dist(nn.Module):
    def __init__(self,dist_type='normal',dim_tensor=None,dim_c=None):
        super().__init__()

        if dist_type == 'std':
            self.prior_dist = Std_Normal(dim_tensor)
        elif dist_type == 'lognormal':
            self.prior_dist = LogNormal(dim_tensor)
        elif dist_type == 'laplace':
            self.prior_dist = Laplace(dim_tensor)
        elif dist_type == 'gamma':
            self.prior_dist = Gamma(dim_tensor)
        elif dist_type == 'std_bias':
            self.prior_dist = Std_Bias(dim_tensor,dim_c)
        else:
            raise ValueError

    def forward(self,nbatch=1,c_in=None):
        x_out = self.prior_dist(nbatch,c_in)
        return x_out

class Std_Normal(nn.Module):
    def __init__(self,dim_tensor=None):
        super().__init__()

        self.name = 'Std_Normal'
        self.dummy = nn.Parameter(torch.zeros_like(dim_tensor))

    def forward(self,nbatch=1,c_in=None):
        with torch.no_grad():
            dummy = self.dummy.repeat_interleave(nbatch,dim=0)
            x_out = torch.randn_like(dummy)
        return x_out

class Std_Bias(nn.Module):
    def __init__(self,dim_tensor=None,dim_c=None):
        super().__init__()

        self.name = 'Std_Bias'
        self.total = nn.Parameter(torch.zeros_like(dim_tensor))

        dim = dim_tensor.size()

        enc_net  = []
        bias_net = []
        for i in range(len(dim)-1):
            enc_net  += [nn.Sequential(nn.Linear(dim_c,64),nn.SiLU(),
                                       nn.Linear(   64,64),nn.SiLU(),
                                       nn.LayerNorm(64,elementwise_affine=False, bias=False))]
            bias_net += [nn.Sequential(nn.Linear(64,64),nn.SiLU(),
                                       nn.Linear(64,64),nn.SiLU(),
                                       nn.Linear(64,dim[i+1]))]

        self. enc_net = nn.ModuleList( enc_net)
        self.bias_net = nn.ModuleList(bias_net)

    def forward(self,nbatch=1,c_in=None):
        total = self.total.repeat_interleave(nbatch,dim=0)
        x_out = torch.randn_like(total)

        #add biases

        bias = []
        for i in range(len(self.bias_net)):
            z0 = self.enc_net[i](c_in)
            z1 = z0 + torch.randn_like(z0)
            bias += [ self.bias_net[i](z1) ]

        c_out = bias[0]
        for i in range(1,len(bias)):
            for j in range(i,len(bias)):
                bias[j] = bias[j].unsqueeze(1)
            c_out = c_out.unsqueeze(-1) + bias[i]

        x_out = x_out + c_out

        return x_out

class Normal(nn.Module):
    def __init__(self,dim_tensor=None,learnable=False):
        super().__init__()

        self.name = 'Normal'
        self.param_a = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=False))
        self.param_b = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=False))

        self.dist = distributions.Normal

    def forward(self,nbatch=1,c_in=None):
        param_a = self.param_a      .repeat_interleave(nbatch,dim=0)
        param_b = self.param_b.exp().repeat_interleave(nbatch,dim=0)

        dist  = self.dist(param_a,param_b)
        x_out = dist.rsample()
        return x_out

class LogNormal(nn.Module):
    def __init__(self,dim_tensor=None,learnable=False):
        super().__init__()

        self.name = 'LogNormal'
        self.param_a = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=False))
        self.param_b = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=False))

        self.dist = distributions.LogNormal

    def forward(self,nbatch=1,c_in=None):
        param_a = self.param_a      .repeat_interleave(nbatch,dim=0)
        param_b = self.param_b.exp().repeat_interleave(nbatch,dim=0)

        dist  = self.dist(param_a,param_b)
        x_out = dist.rsample()
        return x_out

class Laplace(nn.Module):
    def __init__(self,dim_tensor=None,learnable=False):
        super().__init__()

        self.name = 'Laplace'
        self.param_a = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=False))
        self.param_b = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=False))

        self.dist = distributions.Laplace

    def forward(self,nbatch=1,c_in=None):
        param_a = self.param_a      .repeat_interleave(nbatch,dim=0)
        param_b = self.param_b.exp().repeat_interleave(nbatch,dim=0)

        dist  = self.dist(param_a,param_b)
        x_out = dist.rsample()
        return x_out

class Gamma(nn.Module):
    def __init__(self,dim_tensor=None,learnable=False):
        super().__init__()

        self.name = 'Gamma'
        self.param_a = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=False))
        self.param_b = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=False))

        self.dist = distributions.Gamma

    def forward(self,nbatch=1,c_in=None):
        param_a = self.param_a.exp().repeat_interleave(nbatch,dim=0)
        param_b = self.param_b.exp().repeat_interleave(nbatch,dim=0)

        dist  = self.dist(param_a,param_b)
        x_out = dist.rsample()
        return x_out

