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
import torch.nn.functional as F

class regularizer:
    def __init__(self,reg_model='none',reg_coef=1.0):
        self.reg_coef = reg_coef
        if reg_model == 'moment_diff':
            self.reg_model = moment_diff
        elif reg_model == 'mean_diff':
            self.reg_model = mean_diff
        elif reg_model == 'l2_diff':
            self.reg_model = l2_diff
        elif reg_model == 'min_max':
            self.reg_model = min_max
        elif reg_model == 'none':
            self.reg_model = null_reg
        else:
            raise ValueError

    def compute(self,y_hat,y_true):
        out = self.reg_model(y_hat,y_true)*self.reg_coef
        return out

def null_reg(y_hat,y_true):
    return 0.0

def mean_diff(y_hat,y_true):

    #first moment
    avg_dim = [(1,2),(1,3),(2,3),(1,2,3)]

    reg = 0
    for dim in avg_dim:
        reg = reg + (y_true.mean(dim) - y_hat.mean(dim)).pow(2).mean()

    return reg

def moment_diff(y_hat,y_true,moments=[1,2,4]):

    avg_dim = [(1,2),(1,3),(2,3)]

    reg = 0
    for m in moments:
        for dim in avg_dim:
            m_true = get_moments(y_true,avg_dim=dim,order=m) + 1.e-3
            m_hat  = get_moments(y_hat ,avg_dim=dim,order=m) + 1.e-3

            reg += (m_hat.log()-m_true.log()+m_true/m_hat).mean()

            #reg = reg +(m_true-m_hat).pow(2).mean()
            #print(f'moment {m} for x{i} has max {m_true.max().item()}')
    return reg

def l2_diff(y_hat,y_true):

    avg_dim = [(1,2),(1,3),(2,3)]

    reg = 0
    for dim in avg_dim:
        m_true = y_true.pow(2).mean(dim)+ 1.e-3
        m_hat  = y_hat .pow(2).mean(dim)+ 1.e-3

        reg += (m_hat.log()-m_true.log()+m_true/m_hat).mean()
    return reg


def get_moments(x_in,avg_dim,order=1):
    if order == 1:
        x_out = x_in.mean(avg_dim)
    else:
        x_out = x_in - x_in.mean(avg_dim,keepdim=True)
        x_out = x_out.pow(order).mean(avg_dim)
    return x_out

def min_max(y_hat,y_true):
    #regularize by maximum and mininum values
    x_max = F.relu(y_hat - y_true.amax(dim=(1,2,3),keepdim=True))
    x_min = F.relu(y_true.amin(dim=(1,2,3),keepdim=True) - y_hat)
    reg = x_max.pow(2).sum()/(x_max.count_nonzero()+1.e-2) + x_min.pow(2).sum()/(x_min.count_nonzero()+1.e-2)
    return reg
