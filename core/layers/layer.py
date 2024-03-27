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

class layer(nn.Module):
    """
    Class template for models.

    1. Together with the neural network parameters, layer class should have "variational" (bool).
       When "variational==True", the latent space of the model should be a normal distribution.
       When "variational==False", the model has a deterministic latent variable

    2. __init__ should define an array "self.decoder_input". 
       "decoder_input" is a tensor, whose size is torch.zeros(1,size of input variable of decoder)
    """
    def __init__(self):
        super().__init__()

        self.encoder = nn.Linear(1,3)
        self.decoder = nn.Linear(3,1)

        self.variational = False

        if self.variational:
            self.exp_net = nn.Linear(3,3)
            self.var_net = nn.Linear(3,3)

        self.decoder_input = torch.zeros(1,3)

    """
    encoding function takes an input variable (x_in,c_in) and returns the latent variable.
    If (self.variational == True) and (sampling == False), encoding returns mean and variance 
    If (self.variational == True) and (sampling == True ), encoding returns a sample from mean and variance 
    If self.varional == False, encoding returns a deterministic latent variable
    """
    def encoding(self,x_in,c_in=None,sampling=True):

        z0 = self.encoder(x_in,c_in)

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
        out = self.decoder(z_in,c_in)
        return out

    """
    forward function combines encoding and decoding.
    If self.variational == True, forward function is based on the sampling
    """
    def forward(self,x_in,c_in=None):
        z_in = self.encoding(x_in,c_in,sampling=True)
        out  = self.decoding(z_in,c_in)
        return out
