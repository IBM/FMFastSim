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
    def __init__(self,dist_type='normal',dim_tensor=None):
        super().__init__()

        if dist_type == 'normal':
            self.prior_dist = Normal(dim_tensor)
        elif dist_type == 'lognormal':
            self.prior_dist = LogNormal(dim_tensor)
        elif dist_type == 'laplace':
            self.prior_dist = Laplace(dim_tensor)
        elif dist_type == 'gamma':
            self.prior_dist = Gamma(dim_tensor)
        elif dist_type == 'std':
            self.prior_dist = Std_Normal(dim_tensor)
        else:
            raise ValueError

    def forward(self,nbatch=1):
        x_out = self.prior_dist(nbatch)
        return x_out

class Std_Normal(nn.Module):
    def __init__(self,dim_tensor=None):
        super().__init__()

        self.name = 'Std_Normal'
        self.dummy = nn.Parameter(torch.zeros_like(dim_tensor))

    def forward(self,nbatch=1):
        with torch.no_grad():
            dummy = self.dummy.repeat_interleave(nbatch,dim=0)
            x_out = torch.randn_like(dummy)
        return x_out

class Normal(nn.Module):
    def __init__(self,dim_tensor=None):
        super().__init__()

        self.name = 'Normal'
        self.param_a = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=True))
        self.param_b = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=True))

        self.dist = distributions.Normal

    def forward(self,nbatch=1):
        param_a = self.param_a      .repeat_interleave(nbatch,dim=0)
        param_b = self.param_b.exp().repeat_interleave(nbatch,dim=0)

        dist  = self.dist(param_a,param_b)
        x_out = dist.rsample()
        return x_out

class LogNormal(nn.Module):
    def __init__(self,dim_tensor=None):
        super().__init__()

        self.name = 'LogNormal'
        self.param_a = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=True))
        self.param_b = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=True))

        self.dist = distributions.LogNormal

    def forward(self,nbatch=1):
        param_a = self.param_a      .repeat_interleave(nbatch,dim=0)
        param_b = self.param_b.exp().repeat_interleave(nbatch,dim=0)

        dist  = self.dist(param_a,param_b)
        x_out = dist.rsample()
        return x_out

class Laplace(nn.Module):
    def __init__(self,dim_tensor=None):
        super().__init__()

        self.name = 'Laplace'
        self.param_a = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=True))
        self.param_b = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=True))

        self.dist = distributions.Laplace

    def forward(self,nbatch=1):
        param_a = self.param_a      .repeat_interleave(nbatch,dim=0)
        param_b = self.param_b.exp().repeat_interleave(nbatch,dim=0)

        dist  = self.dist(param_a,param_b)
        x_out = dist.rsample()
        return x_out

class Gamma(nn.Module):
    def __init__(self,dim_tensor=None):
        super().__init__()

        self.name = 'Gamma'
        self.param_a = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=True))
        self.param_b = nn.Parameter(torch.zeros_like(dim_tensor,requires_grad=True))

        self.dist = distributions.Gamma

    def forward(self,nbatch=1):
        param_a = self.param_a.exp().repeat_interleave(nbatch,dim=0)
        param_b = self.param_b.exp().repeat_interleave(nbatch,dim=0)

        dist  = self.dist(param_a,param_b)
        x_out = dist.rsample()
        return x_out

