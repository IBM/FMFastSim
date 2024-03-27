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

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed import all_reduce

import numpy as np

from core.handler import ModelHandler

from core.generative_model.prior_dist import prior_dist
from core.generative_model.regularizer import regularizer

from core.layers.MixerTF import Cond_Net
from core.layers.lib_mixer import Mixer3D,init_weights

class GAN(nn.Module):
    def __init__(self, network=None, 
                 reg_coef=0.0,reg_model='none',  #regularization parameters
                 g_net_substep=1,                #generator iterations per discriminator iteration
                 prior_distribution='std',       #prior distributions: std, normal, gamma, laplace
                 d_net_param=None,               #define discriminator
                 gan_model='wgan',               #'gan' or 'wgan'
                 grad_norm_coef = 0,             #penalty for the gradient norm
                 cr_gan = 0,                     #consistency regularization
                 ):

        super().__init__()

        self.model = network

        self.gan = gan_model
        if self.gan == 'gan':
            print('Standard GAN')
        elif self.gan == 'wgan':
            print('Wasserstein GAN')
        else:
            raise ValueError

        if cr_gan:
            print('CR-GAN')

        self.g_net_substep = g_net_substep

        self.regularizer = regularizer(reg_model,reg_coef)

        self.grad_norm_coef = grad_norm_coef

        dim_r = self.model.dim_r
        dim_a = self.model.dim_a
        dim_v = self.model.dim_v

        if d_net_param == None:
            d_net_param = {'dim_r':[dim_r,16, 8,4],
                           'dim_a':[dim_a,32,16,8],
                           'dim_v':[dim_v,32,16,4],
                           'mlp_ratio' :4,
                           'mlp_layers':4}

        print('Prior Distribution is '+prior_distribution)

        #Define Prior
        input_dim = self.model.decoder_input
        self.prior = prior_dist(prior_distribution,input_dim)

        #Define Discriminator
        self.d_net = D_Net(**d_net_param,gan_model=self.gan,cr_gan=cr_gan)

        self.loss_counter = -1

    def forward(self, X, gan_step = None):

        x_input, cond_var = self.prepare_input(X)

        eps = self.prior(nbatch=cond_var.size(0))

        x_out = self.model.decoding(eps,cond_var)

        if gan_step == None:
            return x_out
        elif gan_step == 'd_step':
            #Discriminator Step
            x_out  = x_out.detach() 
            d_true = self.d_net(x_input,cond_var)
            d_fake = self.d_net(x_out  ,cond_var)

            if self.gan == 'gan':
                d_true = d_true*0.98+0.01
                d_fake = d_fake*0.98+0.01
                d_score = -(d_true.log().mean()+d_fake.mul(-1).add(1).log().mean())
            elif self.gan == 'wgan':
                d_score = -(d_true - d_fake).mean()
            else:
                raise ValueError

            if not self.training:
                d_score = -d_score
                if self.gan == 'gan':
                    d_score = d_score - 2*np.log(0.5)

            if (self.training==True) and (self.grad_norm_coef > 1.e-6):
                gp = self.d_net.gradient_penalty(x_out,x_input,cond_var)
                d_score = d_score+gp*self.grad_norm_coef

            return d_score

        elif gan_step == 'g_step':
            #Generator Step
            self.d_net.requires_grad_(False)

            d_fake = self.d_net(x_out,cond_var)

            if self.gan == 'gan':
                d_fake = d_fake*0.98+0.01
                d_score = - d_fake.log().mean()
            elif self.gan == 'wgan':
                d_score = - d_fake.mean()
            else:
                raise ValueError
         
            reg = self.regularizer.compute(y_hat=x_out,y_true=x_input)

            d_score = d_score + reg

            self.d_net.requires_grad_(True)

            return d_score


    def prepare_input(self,X,return_cond=False):
        (x_input, e_input, angle_input, geo_input) = X

        if e_input.dim() == 1:
            e_input     = e_input    .unsqueeze(1)
            angle_input = angle_input.unsqueeze(1)

        cond_var = torch.cat([e_input,angle_input,geo_input],dim=1)

        if return_cond:
            return cond_var
        else:
            return x_input, cond_var

    def generate(self,inputs):
        x_out = self.forward(inputs)
        return x_out

    def update_loss_counter(self):
        self.loss_counter = (self.loss_counter+1)%(self.g_net_substep+1)
        return self.loss_counter

    def reset_loss_counter(self):
        self.loss_counter = -1

    def loss(self,y_hat=None,y_true=None,cond_var=None):
   
        d_score = y_hat

        return d_score


##############################################################################
#   Discriminator Network
##############################################################################
class D_Net(nn.Module):
    def __init__(self,dim_r,dim_a,dim_v,dim_c,mlp_ratio=4,mlp_layers=2,activation=nn.SiLU,gan_model='gan',add_filter=False,cr_gan=0):
        super().__init__()

        self.module = DNet_Core(dim_x0 = dim_r,   \
                                dim_x1 = dim_a,   \
                                dim_x2 = dim_v,   \
                                dim_c  = dim_c,   \
                                mlp_ratio  = mlp_ratio,  \
                                mlp_layers = mlp_layers, \
                                activation = activation)

        d_model = dim_v[-1]*dim_r[-1]*dim_a[-1]

        if add_filter:
            self.highpass = None
            #self.highpass = DNet_Core(dim_x0 = dim_r,   \
            #                          dim_x1 = dim_a,   \
            #                          dim_x2 = dim_v,   \
            #                          dim_c  = dim_c,   \
            #                          mlp_ratio  = mlp_ratio,  \
            #                          mlp_layers = mlp_layers, \
            #                          activation = activation)
            #d_model = d_model + dim_v[-1]*dim_r[-1]*dim_a[-1]

            #self.logtrans = None
            self.logtrans = DNet_Core(dim_x0 = dim_r,   \
                                      dim_x1 = dim_a,   \
                                      dim_x2 = dim_v,   \
                                      dim_c  = dim_c,   \
                                      mlp_ratio  = mlp_ratio,  \
                                      mlp_layers = mlp_layers, \
                                      activation = activation)
            d_model = d_model + dim_v[-1]*dim_r[-1]*dim_a[-1]
        else:
            self.highpass = None
            self.logtrans = None

        self.d_score = nn.Sequential(nn.Linear(d_model,256),nn.SiLU(),
                                     nn.Linear(    256,256),nn.SiLU(),
                                     nn.Linear(    256,  1))
        if gan_model == 'gan':
            self.d_score.add_module('scale',nn.Sigmoid())
        #self.d_score .apply(lambda m: init_weights(m,gain=1.4))

        if cr_gan > 0:
            cr_d_net = []
            cr_d_net+= [CR_D_Net(dim_r[0],dim_c,avg_dim=(2,3),activation=nn.SiLU,gan_model=gan_model,moment=cr_gan)]
            cr_d_net+= [CR_D_Net(dim_a[0],dim_c,avg_dim=(1,3),activation=nn.SiLU,gan_model=gan_model,moment=cr_gan)]
            cr_d_net+= [CR_D_Net(dim_v[0],dim_c,avg_dim=(1,2),activation=nn.SiLU,gan_model=gan_model,moment=cr_gan)]
            cr_d_net+= [Total_E_D_Net(    dim_c,              activation=nn.SiLU,gan_model=gan_model              )]

            self.cr_d_net = nn.ModuleList(cr_d_net)

            self.init_cr_scale = True
        else:
            self.cr_d_net = None
            self.init_cr_scale = False


    #x_in : dimension Batch x Radial x Azimuthal x Vertical
    def forward(self,x_in,c_in):

        score = self.get_score(x_in,c_in)

        if self.cr_d_net != None:
            score = [score]
            for i in range(len(self.cr_d_net)):
                score += [self.cr_d_net[i](x_in,c_in)]
            score = torch.cat(score,dim=0)

        return score

    def get_score(self,x_in,c_in):
        nb = x_in.size(0)

        #Compute D_Net score
        z0 = self.module(x_in,c_in).reshape(nb,-1)

        if self.highpass != None:
            #x0 = (x_in/2).pow(4)
            x0 = x_in.pow(2)
            z1 = self.highpass(x0,c_in).reshape(nb,-1)
            z0 = torch.cat([z0,z1],dim=1)

        if self.logtrans!= None:
            x0 = (x_in.add(1.e-3).log()-np.log(1.e-3))
            z1 = self.logtrans(x0,c_in).reshape(nb,-1)
            z0 = torch.cat([z0,z1],dim=1)

        score = self.d_score(z0)
        return score

    def gradient_penalty(self,y_hat,y_true,c_in):
        nb = y_hat.size(0)

        eps = torch.rand_like(y_hat[:,:1,:1,:1])

        y0 = eps*y_hat.detach() + (1-eps)*y_true.detach()
        
        yy = y0  *torch.ones_like(y0  ,requires_grad=True)
        c0 = c_in*torch.ones_like(c_in,requires_grad=True)

        score = self.get_score(yy,c0)

        gp = self.gradient_norm((yy,c0),score)

        if self.cr_d_net != None:
            for i in range(len(self.cr_d_net)):
                yy = self.cr_d_net[i].scale_data(y0)
                yy = yy  *torch.ones_like(  yy,requires_grad=True)
                c0 = c_in*torch.ones_like(c_in,requires_grad=True)
                score = self.cr_d_net[i].get_score(yy,c0)
                gp = gp + self.gradient_norm((yy,c0),score)

        return gp

    def gradient_norm(self,x_in,x_out):
        nb = x_in[0].size(0)

        gradient = torch.autograd.grad(inputs =x_in,
                                       outputs=x_out,
                                       grad_outputs=torch.ones_like(x_out), 
                                       create_graph=True,
                                       retain_graph=True,
                                       )[0]

        grad_norm = gradient.view(nb, -1).norm(2, dim=1)

        return grad_norm.add(-1).pow(2).mean()

    def init_cr_scale_coef(self):
        init_cr_scale = self.init_cr_scale
        self.init_cr_scale = False
        return init_cr_scale

    def update_cr_scale_coef(self,x_in):
        if self.cr_d_net != None:
            for p in self.cr_d_net:
                p.update_scale_coef(x_in)
      
    def normalize_cr_scale_coef(self,norm_fac):
        if self.cr_d_net != None:
            for p in self.cr_d_net:
                p.normalize_scale_coef(norm_fac)

    def reduce_cr_scale_coef(self):
        if self.cr_d_net != None:
            for p in self.cr_d_net:
                all_reduce(p.data_scale)

class DNet_Core(nn.Module):
    def __init__(self,dim_x0,dim_x1,dim_x2,dim_c,mlp_ratio,mlp_layers,activation=nn.SiLU):
        super().__init__()

        module = []
        cond_net = []
        for i in range(len(dim_x0)-1):
            module += [Mixer3D(dim_x0     = dim_x0[i:i+2], 
                               dim_x1     = dim_x1[i:i+2], 
                               dim_x2     = dim_x2[i:i+2], 
                               mlp_ratio  = mlp_ratio,    
                               activation = activation,   
                               mlp_layers = mlp_layers,
                               res_conn   = False)]

            cond_x0 = [8,16,dim_x0[i]]
            cond_x1 = [8,16,dim_x1[i]]
            cond_x2 = [8,16,dim_x2[i]]
            cond_net += [Cond_Net(cond_x0,cond_x1,cond_x2,dim_c)]

        self.module   = nn.ModuleList(module)
        self.cond_net = nn.ModuleList(cond_net)

        #self.module  .apply(lambda m: init_weights(m,gain=1.0))
        self.cond_net.apply(lambda m: init_weights(m,gain=1.4))

        #self.norm = nn.LayerNorm([dim_x0[0],dim_x1[0],dim_x2[0]])

    #X_in : input of dimension Batch x Vertical x Radial x Azimuthal
    def forward(self,x_in,c_in):

        z0 = x_in
        #z0 = self.norm(x_in)
        for i in range(len(self.module)):
            pos_emb,scale_emb = self.cond_net[i](c_in)
            shift = pos_emb + z0*scale_emb
            z0    = self.module[i](z0,shift)

        return z0

class CR_D_Net(nn.Module):
    def __init__(self,dim_x,dim_c,avg_dim,activation,gan_model='gan',moment=1):
        super().__init__()
        self.dim_x   = dim_x
        self.dim_c   = dim_c
        self.avg_dim = avg_dim
        self.moment  = moment

        self.data_scale = 0
        
        #self.core = nn.Sequential(nn.LayerNorm(dim_x),
        #                          nn.Linear(dim_x,128),activation(),
        #                          nn.Linear(  128, 32))

        #self.norm = nn.BatchNorm1d(dim_x,affine=False)
        #self.norm = nn.LayerNorm(dim_x)

        self.core = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=4,kernel_size=5),activation(),
                                  nn.Conv1d(in_channels=4,out_channels=4,kernel_size=5),activation(),
                                  nn.Conv1d(in_channels=4,out_channels=4,kernel_size=5),activation())
        self.enc  = nn.Linear((dim_x-4*3)*4,64)

        self.pos_emb   = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                       nn.Linear(   64,64),activation(),
                                       nn.Linear(   64,64))

        self.scale_emb = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                       nn.Linear(   64,64),activation(),
                                       nn.Linear(   64,64))

        self.score= nn.Sequential(nn.LayerNorm(64),
                                  nn.Linear(64,64),activation(),
                                  nn.Linear(64,64),activation(),
                                  nn.Linear(64, 1))

        if gan_model == 'gan':
            self.score.add_module('scale',nn.Sigmoid())

        #self.core .apply(lambda m: init_weights(m,gain=1.0))
        self.pos_emb  .apply(lambda m: init_weights(m,gain=1.4))
        self.scale_emb.apply(lambda m: init_weights(m,gain=1.4))

    def forward(self,x_in,c_in):
        x0  = self.scale_data(x_in)
        out = self.get_score(x0,c_in)
        return out

    def scale_data(self,x_in):
        out   = x_in.pow(self.moment).sum(self.avg_dim)
        return out

    def get_score(self,x_in,c_in):
        nb = x_in.size(0)

        pos_emb   = self.pos_emb(c_in)
        scale_emb = self.scale_emb(c_in)

        #z0 = self.norm(x_in)
        z0 = x_in/self.data_scale
        z1 = self.core(z0.unsqueeze(1))
        z2 = self.enc (z1.view(nb,-1))

        z3 = pos_emb+z2*(scale_emb+1)

        out = self.score(z3)
        return out

    @torch.no_grad()
    def update_scale_coef(self,x_in):
        self.data_scale += x_in.pow(self.moment).sum(self.avg_dim).mean()

    @torch.no_grad()
    def normalize_scale_coef(self,norm_fac):
        self.data_scale = self.data_scale*norm_fac

class Total_E_D_Net(nn.Module):
    def __init__(self,dim_c,activation,gan_model='gan',moment=1):
        super().__init__()
        self.dim_c   = dim_c
        self.moment  = moment

        self.data_scale =  0
        
        d0 = dim_c+1
        self.score= nn.Sequential(nn.Linear( d0,128),activation(),
                                  nn.Linear(128,128),activation(),
                                  nn.Linear(128,  1))

        #self.norm = nn.BatchNorm1d(d0,affine=False)

        if gan_model == 'gan':
            self.score.add_module('scale',nn.Sigmoid())

        #self.score.apply(lambda m: init_weights(m,gain=1.0))

    def forward(self,x_in,c_in):
        x0  = self.scale_data(x_in)
        out = self.get_score(x0,c_in)
        return out

    def scale_data(self,x_in):
        out   = x_in.pow(self.moment).sum((1,2,3)).unsqueeze(1)
        return out

    def get_score(self,x_in,c_in):
        x0  = x_in/self.data_scale
        x0  = torch.cat([x0,c_in],dim=1)
        #x0  = self.norm (x0)
        out = self.score(x0)
        return out

    @torch.no_grad()
    def update_scale_coef(self,x_in):
        self.data_scale += x_in.pow(self.moment).sum((1,2,3)).mean()

    @torch.no_grad()
    def normalize_scale_coef(self,norm_fac):
        self.data_scale = self.data_scale*norm_fac

def spectral_norm(m):
    if type(m) == nn.Linear:
        nn.utils.parametrizations.spectral_norm(m)

#########################################################################################
#   GAN Handler
#########################################################################################
class GANHandler(ModelHandler):
    def __init__(self, gen_param, network, **kwargs):

        self._gen_param = gen_param

        self._model = GAN(network=network,**gen_param)

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
        return self._model(data_in)


    def _train_one_epoch(self, trainloader, validloader, optimizer):

        model         = self._model

        _loss         = model.loss
        gan_type      = model.gan
        prepare_input = model.prepare_input
        grad_norm_coef= model.grad_norm_coef

        if self._ddp:
            model = self._ddp_model
            d_net_params   = model.module.d_net.parameters
            reset_counter  = model.module. reset_loss_counter
            update_counter = model.module.update_loss_counter
            init_cr_scale  = model.module.d_net. init_cr_scale_coef
            get_cr_scale   = model.module.d_net.update_cr_scale_coef
            norm_cr_scale  = model.module.d_net.normalize_cr_scale_coef
            reduce_cr_scale= model.module.d_net.reduce_cr_scale_coef
        else:
            d_net_params   = model.d_net.parameters
            reset_counter  = model. reset_loss_counter
            update_counter = model.update_loss_counter
            init_cr_scale  = model.d_net. init_cr_scale_coef
            get_cr_scale   = model.d_net.update_cr_scale_coef
            norm_cr_scale  = model.d_net.normalize_cr_scale_coef

        #prepare scaling
        if init_cr_scale():
            for _,y in trainloader:
                y = self._to_dev(y)
                get_cr_scale(y)

            if self._ddp:
                norm_fac =1/(self._num_gpu*len(trainloader))
                reduce_cr_scale()
            else:
                norm_fac = 1/len(trainloader)

            norm_cr_scale(norm_fac)

        reset_counter()

        train_loss = 0.0
        for X,y in trainloader:
            step_switch = update_counter()
            if step_switch == 0:
                gan_step = 'd_step'
            else:
                gan_step = 'g_step'

            X,y = self._to_dev(X), self._to_dev(y)

            cond_var = prepare_input(X,return_cond=True)

            optimizer.zero_grad()
            y_hat = model(X,gan_step=gan_step)
            loss  = _loss(y_hat=y_hat)

            loss.backward()

            optimizer.step()

            #weight clipping
            if (gan_type == 'wgan') and (grad_norm_coef < 1.e-5):
                for p in d_net_params():
                    p.data.clamp_(-0.01,0.01)

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X,y in validloader:
                X,y = self._to_dev(X), self._to_dev(y)

                cond_var = prepare_input(X,return_cond=True)

                y_hat = model(X,gan_step='d_step')
                loss  = _loss(y_hat=y_hat)

                val_loss += loss.item()
        model.train()

        return train_loss / len(trainloader), val_loss / len(validloader)


    def save_model(self,epoch=None,save_file=None):

        super().save_model(epoch=epoch,save_file=save_file)

        if self._rank == 0:
            torch.save({'network'   :self._model.model.state_dict(),
                        'prior'     :self._model.prior.state_dict(),
                        'D_Net'     :self._model.d_net.state_dict(),
                        'all_params':self._params},
                       self.save_file)

    def load_model(self,epoch=None,load_file=None):

        super().load_model(epoch=epoch,load_file=load_file)

        model_load = torch.load(self.load_file,map_location=self._device)

        t0 = model_load['network']
        t1 = model_load['prior']
        t2 = model_load['D_Net']

        self._model.model.load_state_dict(t0)
        self._model.prior.load_state_dict(t1)
        self._model.d_net.load_state_dict(t2)
