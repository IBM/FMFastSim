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

# Standard
import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

#scaling function
class voxel_scaling:
    def __init__(self,scale_method=None,scale_param={'dummy':None},scale_by_energy=True):

        if scale_method == 'log_trans':
            self.scale = log_trans(**scale_param)
        elif scale_method == 'lin_trans':
            self.scale = lin_trans(**scale_param)
        elif scale_method == 'logit_trans':
            self.scale = logit_trans(**scale_param)
        elif scale_method == 'ds2_logit_trans_and_nomalization':
            self.scale = ds2_logit_trans_and_nomalization(**scale_param)
        else:
            print('data is not scaled')
            self.scale = identity()

        self.scale_by_energy = scale_by_energy
        self.scale_method = scale_method

    def transform(self,x_in,e_in=None):
        if self.scale_by_energy and (e_in is not None):
            x_in = x_in/e_in
        out = self.scale.transform(x_in)
        return out

    def inverse_transform(self,y_in,e_in=None):
        out = self.scale.inverse_transform(y_in)
        if self.scale_by_energy and (e_in is not None):
            out = out*e_in
        return out

    def transform_energy(self, energy):
        energy_min = 1 #after division by 1000
        energy_max = 1000 #after division by 1000
        energy = np.log10(energy/energy_min)/np.log10(energy_max/energy_min)
        return energy

    def inverse_transform_energy(self, energy, energy_max=1000):
        energy_min = 1 #after division by 1000
        energy_max = energy_max #after division by 1000
        energy = energy_min*(energy_max/energy_min)**energy
        return energy

    #theta from 0.0 to 3.14 -> 0 and 1 (could also use either cos or sin?)
    def transform_theta(self, theta):
        theta_min = 1e-8
        theta_max = np.pi
        theta = np.log10(theta/theta_min)/np.log10(theta_max/theta_min)
        return theta

    def inverse_transform_theta(self, theta):
        theta_min = 1e-8 
        theta_max = np.pi 
        theta = theta_min*(theta_max/theta_min)**theta
        return theta

    #phi from -pi to pi -> 0 and 1 (periocity)
    def transform_phi(self, phi):
        phi_sin = np.sin(phi)
        phi_cos = np.cos(phi)
        phi = np.concatenate((phi_sin, phi_cos), axis=-1)
        return phi

    def inverse_transform_phi(self, phi):
        phi_from_sin = np.arcsin(phi)
        phi_from_cos = np.arccos(phi)
        phi = phi_from_sin 
        return phi

#log_trans => y = mag*log[ (x+bias)/scale ]
class log_trans:
    def __init__(self,scale=1,mag=1,bias=1,positive=False):
        self.name='log transform'

        self.mag   = mag
        self.bias  = bias
        self.scale = scale
        self.positive = positive
        
        if positive:
            self.shift = min(np.log(self.bias*1.00001/self.scale),0)
        else:
            self.shift = 0

    def transform(self,x_in):
        out = self.mag*(np.log((x_in+self.bias)/self.scale)-self.shift)
        if self.positive:
            out = np.maximum(out,0)
        return out

    def inverse_transform(self,y_in,numpy=True):
        z = y_in/self.mag+self.shift
        if numpy:
            x = self.scale*np.exp(z)-self.bias
            return np.maximum(x,0)
        else:
            x = self.scale*z.exp()-self.bias
            return F.relu(x)

#lin_trans => y = (x+bias)/scale
class lin_trans:
    def __init__(self,scale=1,bias=1,**kwargs):
        self.name='linear transform'

        self.bias  = bias
        self.scale = scale

    def transform(self,x_in):
        return (x_in+self.bias)/self.scale

    def inverse_transform(self,y_in,numpy=True):
        x = self.scale*y_in-self.bias
        if numpy:
            return np.maximum(x,0)
        else:
            return F.relu(x)

#logit_trans => y = tanh((x+eps)/scale) => z = log(y) - log(1-y)
class logit_trans:
    def __init__(self,scale=1,bias=1.e-6,mag=1,positive=False):
        self.name='logit transformation'
        self.scale = scale
        self.bias  = bias
        self.mag   = mag
        self.positive = positive

        if positive:
            y = np.tanh(self.bias/self.scale)*0.999999
            self.shift = np.log(y) - np.log(1-y)
        else:
            self.shift = 0.0

    def transform(self,x_in):
        y = np.tanh((x_in+self.bias)/self.scale)*0.999999
        z = np.log(y) - np.log(1-y)
        out = self.mag*(z-self.shift)
        if self.positive:
            out = np.maximum(out,0)
        return out

    def inverse_transform(self,z_in,numpy=True):
        z = z_in/self.mag+self.shift
        if numpy:
            y = 1/(1+np.exp(-z))
            x = np.arctanh(y)*self.scale-self.bias
            return np.maximum(x,0)
        else:
            y = 1/(1+z.mul(-1).exp())
            x = y.arctanh()*self.scale-self.bias
            return F.relu(x)

#identity
class identity:
    def __init__(self):
        self.name='identity'

    def transform(self,x_in):
        return x_in

    def inverse_transform(self,y_in):
        return y_in

#original CaloDit shower preprocessing (logit + normalization)
class ds2_logit_trans_and_nomalization:
    def __init__(self,mean,std,epsilon_logit=1.e-6,scale_shower=1.5):
        self.name='ds2_logit_trans_and_nomalization'
        self.epsilon_logit = epsilon_logit
        self.mean = mean
        self.std = std
        self.scale_shower = scale_shower

    def transform(self,x_in):
        shower = x_in/(self.scale_shower)
        shower = self.epsilon_logit + (1 - 2 * self.epsilon_logit) * shower #remove 0 and 1
        #shower = np.ma.log(shower/(1-shower)).filled(0) #applies logit on the shower, ma is a masked array, if it falls out the validity domain fills the value with 0
        shower = np.log(shower/(1-shower)) #applies logit on the shower
        shower = (shower - self.mean) / self.std
        return shower

    def inverse_transform(self,z_in):
        orignial_shower = (z_in * self.std) + self.mean
        orignial_shower = np.clip(orignial_shower, -88.72, 88.72) #clip values need to cahnge based on precision
        exp = np.exp(orignial_shower)    
        x_exp = exp/(1+exp)
        orignial_shower = (x_exp-self.epsilon_logit)/(1 - 2*self.epsilon_logit)
        orignial_shower = (orignial_shower * self.scale_shower) #* 1000
        return orignial_shower
