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

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.handler import ModelHandler

from core.generative_model.regularizer import regularizer

import math

class Diffusion(nn.Module):
    def __init__(self, network, 
                 diff_model='rescaled',         #diffusion model: 'orig' or 'rescaled'
                 dim_t_emb = 32,                #time embedding dimension
                 beta_info = None,              #noise schedule infor
                 dim_c     = 4,                 #dimension of conditional variable
                 res_conn  = True,              #resnet connection
                 add_mse   = False,             #addtional mse
                 reg_coef=0.0,reg_model='none', #regularization parameters
                 ):

        super().__init__()

        self.model = network

        self.diff_model = diff_model

        self.res_conn = res_conn
        self.add_mse  = add_mse

        self.regularizer = regularizer(reg_model,reg_coef)

        self.dim_r = self.model.dim_r
        self.dim_a = self.model.dim_a
        self.dim_v = self.model.dim_v

        alpha,beta,gamma = noise_scheduler(**beta_info) 

        self.register_buffer('alpha',alpha.float())
        self.register_buffer('beta' ,beta .float())
        self.register_buffer('gamma',gamma.float())
        
        sigma = torch.zeros_like(beta)
        sigma[1:] = beta[1:]*(1-gamma[:-1])/(1-gamma[1:])
        self.register_buffer('sigma',sigma.sqrt().float())

        self.register_buffer('sqrt_one_minus_gamma',(1-gamma).sqrt().float())
        self.register_buffer('sqrt_gamma'          ,   gamma .sqrt().float())
        self.register_buffer('one_over_sqrt_alpha' ,(1/alpha).sqrt().float())

        if self.diff_model == 'orig':
            self.register_buffer('backward_coef_xt',torch.ones_like(beta.float()))
            self.register_buffer('backward_coef_ft',-(beta/(1-gamma).sqrt()).float())
        elif self.diff_model == 'rescaled':
            tmp = beta/(1-gamma)
            self.register_buffer('backward_coef_xt',(1-tmp).float())
            self.register_buffer('backward_coef_ft',   tmp .float())
        else:
            raise ValueError


        self.register_buffer('T',torch.tensor([beta.size(0)],dtype=torch.int32))

        dim_t = dim_t_emb
        self.time_emb = Position_Embeddings(dim_t)
        self.cond_emb = nn.Sequential(nn.Linear(dim_c,128),nn.SiLU(),
                                      nn.Linear(128  , 64),nn.SiLU(),nn.LayerNorm(64),
                                      nn.Linear(64   ,dim_t))

        self.loss_counter = -1

    def prior_sampling(self,x0,t):

        drift = self.sqrt_gamma[t]*x0
        eps   = torch.randn_like(x0)

        xt = drift + self.sqrt_one_minus_gamma[t]*eps

        if self.diff_model == 'orig':
            return xt,eps
        elif self.diff_model == 'rescaled':
            return xt,drift
        else:
            raise ValueError


    def forward(self, x_in, c_in, t_in):
        c0 = self.cond_emb(c_in)
        t0 = self.time_emb(t_in)

        z_in = torch.cat([c0,t0],dim=1)

        x_out = self.model(x_in,z_in)

        if self.res_conn and (self.diff_model == 'orig'):
            x_out = x_out + x_in

        return x_out

    def prepare_input(self,X,return_cond=False):
        x_input = X[0]

        C = []
        for i in range(1,len(X)):
            if X[i].dim() == 1:
                C += [X[i].unsqueeze(1)]
            else:
                C += [X[i]]
        cond_var = torch.cat(C,dim=1)

        if return_cond:
            return cond_var
        else:
            return x_input, cond_var

    def generate(self,inputs):
        x_out = self.backward_sampling(inputs)
        return x_out

    def step_counter(self,counter=4):
        self.loss_counter = (self.loss_counter+1)%counter
        return self.loss_counter

    def loss(self,y_hat=None,y_true=None):
        mse_loss = F.mse_loss(y_hat,y_true)
        mse_loss = mse_loss + self.regularizer.compute(y_hat,y_true)
        return mse_loss

    def masked_loss(self,y_hat=None,y_true=None,mask=None):
        if mask != None:
            mse_loss = torch.masked_select( (y_hat-y_true).pow(2), mask).mean()
        else:
            mse_loss = (y_hat-y_true).pow(2).mean()
        mse_loss = mse_loss + self.regularizer.compute(y_hat,y_true)
        return mse_loss

    @torch.no_grad()
    def backward_sampling(self,X):

        x_input,cond_var = self.prepare_input(X)

        nb = cond_var.size(0)

        if x_input == None:
            xT = torch.zeros(nb,self.dim_r,self.dim_a,self.dim_v,device=cond_var.device)
            xT.normal_()
        else:
            xT = torch.randn_like(x_input)

        x_new = xT
        for i in range(len(self.beta)):
            x_old = x_new

            tt = self.T.item()-i-1
            t_in = torch.ones_like(self.T).mul(tt).repeat(nb)

            drift = self.forward(x_old,cond_var,t_in)

            x_new = (self.backward_coef_xt[tt]*x_old + 
                     self.backward_coef_ft[tt]*drift)*self.one_over_sqrt_alpha[tt]
            x_new = x_new + self.sigma[tt]*torch.randn_like(x_old)

        return x_new

#Positional Embedding
class Position_Embeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        self.register_buffer('embeddings',torch.exp(torch.arange(half_dim) * -embeddings))

    def forward(self, time):
        embeddings = time[:, None] * self.embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

#Define noise schedulers
def noise_scheduler(schedule='cos',num_steps=200,tau=1,**kwargs):
    if schedule == 'cos':
        print('use cosine gamma scheduler')
        gamma = cosine_schedule(num_steps,tau=tau)
    elif schedule == 'sigmoid':
        print('use sigmoid gamma scheduler')
        gamma = sigmoid_schedule(num_steps,tau=tau)
    elif scheduler == 'linear':
        print('use linear gamma scheduler')
        gamma = linear_schedule(num_steps)
    else:
        print(f'{gamma_schedule} scheduler is not defined')
        raise ValueError

    alpha = gamma[1:]/gamma[:-1]
    beta  = 1-alpha
    gamma = gamma[1:] #remove t = 0

    print(f'beta : max {beta [-1].item()} and min {beta [ 0].item()}')
    print(f'gamma: max {gamma[ 0].item()} and min {gamma[-1].item()}')

    return alpha,beta,gamma


#define gamma scheduler
#number of steps: number of diffusion steps
def linear_schedule(num_steps=200):
    t = torch.arange(0,1+1.e-6,step=1/(num_steps+1),dtype=torch.double)

    gamma = 1-t
    gamma = gamma[:-1] #remove the last knot
    return gamma

def sigmoid_schedule(num_steps,start=-3,end=3,tau=1):
    t = torch.arange(0,1+1.e-6,step=1/(num_steps+1),dtype=torch.double)

    x = ((t*(end-start)+start)/tau).sigmoid()

    gamma = (x-x[-1])/(x[0]-x[-1])
    gamma = gamma[:-1] #remove the last knot
    return gamma

def cosine_schedule(num_steps,start=0,end=1,tau=1):
    t = torch.arange(0,1+1.e-6,step=1/(num_steps+1),dtype=torch.double)

    r = start/end
    x = ((t*(1-r)+r)*math.pi/2-1.e-6).cos().pow(2*tau)

    gamma = (x-x[-1])/(x[0]-x[-1])
    gamma = gamma[:-1] #remove the last knot
    return gamma

#########################################################################################################
#   Handler Definition
#########################################################################################################
class DiffusionHandler(ModelHandler):
    def __init__(self, gen_param, network, **kwargs):

        self._gen_param = gen_param

        self._model = Diffusion(network=network,**gen_param)
        self._loss  = self._model.masked_loss

        super().__init__(**kwargs)

    #def _get_wandb_extra_config(self):
    #    return {
    #        "activation": ACTIVATION,
    #        "out_activation": OUT_ACTIVATION,
    #        "intermediate_dims": INTERMEDIATE_DIMS,
    #        "latent_dim": LATENT_DIM,
    #        "num_layers": len(INTERMEDIATE_DIMS)
    #    }

    def _set_model_inference(self):
        self._decoder = self._model

    def generate(self,data_in):
        return self._model.backward_sampling(data_in)

    def _train_one_epoch(self, trainloader, validloader, optimizer):

        model         = self._model

        _loss         = model.masked_loss
        T             = model.T.item()
        step_counter  = model.step_counter
        prepare_input = model.prepare_input
        prior_sampling= model.prior_sampling
        add_mse       = model.add_mse
        diff_model    = model.diff_model
        sqrt_gamma    = model.sqrt_gamma

        if self._ddp:
            model = self._ddp_model

        time = torch.randint(T,(trainloader.batch_size,),device=self._device).view(-1,1,1,1)

        inv_trans = trainloader.dataset.scale_method.inverse_transform

        #train
        train_loss = 0.0
        for X,y in trainloader:
            X,y = self._to_dev(X), self._to_dev(y)
            cond_var = prepare_input(X,return_cond=True)

            #sample time
            tt = time.random_(0,T)[:y.size(0)]

            #prior sampling
            xt,yy = prior_sampling(y,tt)

            optimizer.zero_grad()

            y_hat = model(xt,cond_var,tt.squeeze())

            switch =  step_counter()
            loss  = _loss(y_hat=y_hat,y_true=yy)

            if add_mse and (diff_model == "rescaled"):
            #add log loss
                sg = sqrt_gamma[tt]
                yy = sg*(y    +1.e-4   ).log()
                yh =    (y_hat+1.e-4*sg).log() - sg.log()
                yh = sg*yh

                loss  = loss + _loss(y_hat=yh,y_true=yy)*1.e-2

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        #validation
        self._model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X,y in validloader:
                X,y = self._to_dev(X), self._to_dev(y)
                cond_var = prepare_input(X,return_cond=True)

                #sample time
                tt = time.random_(0,T)[:y.size(0)]

                #prior sampling
                xt,yy = prior_sampling(y,tt)

                y_hat = model(xt,cond_var,tt.squeeze())
                loss  = _loss(y_hat=y_hat,y_true=yy)

                val_loss += loss.item()

        return train_loss / len(trainloader), val_loss / len(validloader)


    def save_model(self,epoch=None,save_file=None):

        super().save_model(epoch=epoch,save_file=save_file)

        if self._rank == 0:
            torch.save({'model'     :self._model.state_dict(),
                        'all_params':self._params},
                       self.save_file)

    def load_model(self,epoch=None,load_file=None):

        super().load_model(epoch=epoch,load_file=load_file)

        theta = torch.load(self.load_file,map_location=self._device)['model']

        self._model.load_state_dict(theta)
