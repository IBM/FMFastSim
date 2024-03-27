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

class regularizer:
    def __init__(self,reg_model='none',reg_coef=0.0):
        self.reg_coef = reg_coef
        if reg_model == 'moment_diff':
            self.reg_model = moment_diff
        elif reg_model == 'mean_diff':
            self.reg_model = mean_diff
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
    m0 = (y_true.mean((1,2)  )-y_hat.mean((1,2))  ).pow(2).div(20).tanh().mul(20).mean()
    m1 = (y_true.mean((1,3)  )-y_hat.mean((1,3))  ).pow(2).div(20).tanh().mul(20).mean()
    m2 = (y_true.mean((2,3)  )-y_hat.mean((2,3))  ).pow(2).div(20).tanh().mul(20).mean()
    m3 = (y_true.mean((1,2,3))-y_hat.mean((1,2,3))).pow(2).div(20).tanh().mul(20).mean()

    out = m0+m1+m2+m3
    return out

def moment_diff(y_hat,y_true):

    reg = 0
    for m in range(4):
        for i in range(3):
            m_true = get_moments(y_true,i+1,m+1)
            m_hat  = get_moments(y_hat ,i+1,m+1)
            reg = reg +(m_true-m_hat).pow(2).div(20).tanh().mul(20).mean()

            #print(f'moment {m} for x{i} has max {m_true.max().item()}')
    return reg


def get_moments(x_in,avg_dim,order=1):
    x_out  = x_in.pow(order).mean(avg_dim)
    return x_out
