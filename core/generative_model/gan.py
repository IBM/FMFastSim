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
                 pretrain_epoch = 0,             #number of pretraining steps
                 final_layer=False,              #add a final mixer layer
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

        if cr_gan > 0:
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

        if final_layer:
            self.final = Mixer3D(dim_x0 = [dim_r,dim_r],
                                 dim_x1 = [dim_a,dim_a], 
                                 dim_x2 = [dim_v,dim_v],
                                 mlp_ratio=3, mlp_layers=1,res_conn=False,norm='identity')
        else:
            self.final = None
                                 

        #Define Prior
        #input_dim = self.model.decoder_input
        input_dim = torch.zeros(1,dim_r,dim_a,dim_v)
        self.prior = prior_dist(prior_distribution,input_dim,self.model.dim_c)

        #Define Discriminator
        self.d_net = D_Net(**d_net_param,gan_model=self.gan,cr_gan=cr_gan)

        self. loss_counter = -1
        self.epoch_counter = -1

        self.pretrain_epoch = pretrain_epoch

    def forward(self, X):

        x_input, cond_var = self.prepare_input(X)

        eps = self.prior(cond_var.size(0),cond_var)

        x_fake = self.model(eps,cond_var)
        if self.final:
            x_fake = self.final(x_fake)

        return x_fake


    def prepare_input(self,X,return_cond=False):
        x_input, conditions = X[0], list(X[1:])

        for i in range(len(conditions)):
            if conditions[i].dim() == 1:
                conditions[i]     = conditions[i].unsqueeze(1)

        cond_var = torch.cat(conditions,dim=1)

        if return_cond:
            return cond_var
        else:
            return x_input, cond_var

    def generate(self,inputs):
        x_out = self.forward(inputs)
        return x_out

    def update_loss_counter(self):
        if self.epoch_counter < self.pretrain_epoch:
            self.loss_counter = 0
        else:
            self.loss_counter = (self.loss_counter+1)%(self.g_net_substep+1)
        return self.loss_counter

    def reset_loss_counter(self):
        self.loss_counter = -1

    def loss(self,x_fake,x_true,gan_step):

        x_input, cond_var = self.prepare_input(x_true)
   
        if self.d_net.cr_d_net == None:
            n_cr = 0
        else:
            n_cr = len(self.d_net.cr_d_net)
        d_score_weight = [1.0] + [1.0 for i in range(n_cr)]

        #Discriminator Step
        if gan_step == 'd_step':
            x_fake = x_fake.detach()

            x_in = torch.cat([x_input ,x_fake  ])
            c_in = torch.cat([cond_var,cond_var])
            
            out = self.d_net(x_in,c_in).chunk(2*(n_cr+1),dim=0)

            d_score = 0.0
            d_score_gstep = 0.0
            for i in range(n_cr+1):
                d_true = out[2*i  ]
                d_fake = out[2*i+1]

                if self.gan == 'gan':
                    d_true = d_true*0.98+0.01
                    d_fake = d_fake*0.98+0.01

                    loss = -(d_true.log().mean()+d_fake.mul(-1).add(1).log().mean())
                elif self.gan == 'wgan':
                    loss = -(d_true - d_fake).mean()
                else:
                    raise ValueError

                d_score = d_score + loss*d_score_weight[i]/sum(d_score_weight)

                if self.gan == 'gan':
                    d_score_gstep += d_fake.detach().log().mean().item() - np.log(0.5)
                elif self.gan == 'wgan':
                    d_score_gstep += d_fake.detach().mean().item()
                else:
                    raise ValueError

            d_score_gstep /= n_cr+1

            if (self.training==True) and (self.grad_norm_coef > 1.e-6):
                gp = self.d_net.gradient_penalty(x_fake,x_input,cond_var)
                d_score = d_score+gp*self.grad_norm_coef

            return d_score,d_score_gstep

        #Generator Step
        elif gan_step == 'g_step':
            self.d_net.requires_grad_(False)
            out = self.d_net(x_fake,cond_var).chunk(n_cr+1,dim=0)
            self.d_net.requires_grad_(True)

            d_score = 0.0
            d_score_gstep = 0.0
            for i in range(n_cr+1):
                d_fake = out[i]

                if self.gan == 'gan':
                    d_fake = d_fake*0.98+0.01
                    loss = -d_fake.log().mean()
                elif self.gan == 'wgan':
                    loss = -d_fake.mean()
                else:
                    raise ValueError

                d_score = d_score + loss*d_score_weight[i]/sum(d_score_weight)

                if self.gan == 'gan':
                    d_score_gstep += d_fake.detach().log().mean().item() - np.log(0.5)
                elif self.gan == 'wgan':
                    d_score_gstep += d_fake.detach().mean().item()
                else:
                    raise ValueError

            d_score_gstep /= n_cr+1

            if self.training:
                d_score = d_score + self.regularizer.compute(y_hat=x_fake,y_true=x_input)

            return d_score,d_score_gstep

        else:
            print('step is not correctly defined')
            raise ValueError


##############################################################################
#   Discriminator Network
##############################################################################
class D_Net(nn.Module):
    def __init__(self,dim_r,dim_a,dim_v,dim_c,mlp_ratio=4,mlp_layers=2,activation=nn.SiLU,gan_model='gan',add_filter=None,cr_gan=0,res_conn=True,norm='layer'):
        super().__init__()

        self.module = DNet_Core(dim_x0 = dim_r,   \
                                dim_x1 = dim_a,   \
                                dim_x2 = dim_v,   \
                                dim_c  = dim_c,   \
                                mlp_ratio  = mlp_ratio,  \
                                mlp_layers = mlp_layers, \
                                activation = activation, \
                                res_conn = res_conn,     \
                                norm = norm)

        d_model = dim_v[-1]*dim_r[-1]*dim_a[-1]

        self.highpass = None
        self.logtrans = None

        if add_filter != None :
            if add_filter == 'highpass':
                self.highpass = DNet_Core(dim_x0 = dim_r,   \
                                          dim_x1 = dim_a,   \
                                          dim_x2 = dim_v,   \
                                          dim_c  = dim_c,   \
                                          mlp_ratio  = mlp_ratio,  \
                                          mlp_layers = mlp_layers, \
                                          activation = activation, \
                                          res_conn = res_conn,     \
                                          norm = norm)
            elif add_filter == 'logtrans':
                self.logtrans = DNet_Core(dim_x0 = dim_r,   \
                                          dim_x1 = dim_a,   \
                                          dim_x2 = dim_v,   \
                                          dim_c  = dim_c,   \
                                          mlp_ratio  = mlp_ratio,  \
                                          mlp_layers = mlp_layers, \
                                          activation = activation, \
                                          res_conn = res_conn,     \
                                          norm = norm)
            else:
                print(f'filter {add_filter} is not defined')
                raise ValueError
            d_model = d_model + dim_v[-1]*dim_r[-1]*dim_a[-1]

        self.d_score = nn.Sequential(nn.Linear(d_model,256),activation(),
                                     nn.Linear(256,256),activation(),
                                     nn.Linear(256,  1))
        if gan_model == 'gan':
            self.d_score.add_module('scale',nn.Sigmoid())

        #self.d_score.apply(lambda m: init_weights(m,gain=1.2))

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
            x0 = x_in.pow(3)
            x1 = self.highpass(x0,c_in).reshape(nb,-1)
            z0 = torch.cat([z0,x1],dim=1)

        if self.logtrans!= None:
            x0 = x_in.add(1.e-4).log()
            x1 = self.logtrans(x0,c_in).reshape(nb,-1)
            z0 = torch.cat([z0,x1],dim=1)

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
    def __init__(self,dim_x0,dim_x1,dim_x2,dim_c,mlp_ratio,mlp_layers,activation=nn.SiLU,res_conn=True,norm='layer'):
        super().__init__()

        module = []
        pos_emb = []
        scale_emb = []
        for i in range(len(dim_x0)-1):
            module += [Mixer3D(dim_x0     = dim_x0[i:i+2], 
                               dim_x1     = dim_x1[i:i+2], 
                               dim_x2     = dim_x2[i:i+2], 
                               mlp_ratio  = mlp_ratio,    
                               activation = activation,   
                               mlp_layers = mlp_layers,
                               res_conn   = res_conn,
                               norm       = norm)]

            pos_emb   += [Cond_Net(dim_x0[i],dim_x1[i],dim_x2[i],dim_c)]
            scale_emb += [Cond_Net(dim_x0[i],dim_x1[i],dim_x2[i],dim_c)]

        self.module    = nn.ModuleList(module)
        self.  pos_emb = nn.ModuleList(  pos_emb)
        self.scale_emb = nn.ModuleList(scale_emb)

    #X_in : input of dimension Batch x Vertical x Radial x Azimuthal
    def forward(self,x_in,c_in):

        z0 = x_in
        for i in range(len(self.module)):
            b = self.  pos_emb[i](c_in)
            s = self.scale_emb[i](c_in)

            z0 = self.module[i](b+(s+1)*z0)

        return z0

class Cond_Net(nn.Module):
    def __init__(self,dim_x0,dim_x1,dim_x2,dim_c,activation=nn.SiLU,norm=False):
        super().__init__()

        self.dim_c  = dim_c
        self.dim_x0 = dim_x0
        self.dim_x1 = dim_x1
        self.dim_x2 = dim_x2

        self.norm = norm

        self.x0_net = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                    nn.LayerNorm(64,elementwise_affine=False, bias=False),
                                    nn.Linear(64,64),activation(),
                                    nn.Linear(64,dim_x0))

        self.x1_net = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                    nn.LayerNorm(64,elementwise_affine=False, bias=False),
                                    nn.Linear(64,64),activation(),
                                    nn.Linear(64,dim_x1))

        self.x2_net = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                    nn.LayerNorm(64,elementwise_affine=False, bias=False),
                                    nn.Linear(64,64),activation(),
                                    nn.Linear(64,dim_x2))

    def forward(self,c_in):
        #create dimensional embedding
        x0 = self.x0_net(c_in)
        x1 = self.x1_net(c_in)
        x2 = self.x2_net(c_in)

        y0 = x0.unsqueeze(-1).repeat(1,1,self.dim_x1)   + x1.unsqueeze(1)
        y1 = y0.unsqueeze(-1).repeat(1,1,1,self.dim_x2) + x2.unsqueeze(1).unsqueeze(1)

        if self.norm:
            y1 = F.layer_norm(y1,[self.dim_x0,self.dim_x1,self.dim_x2])

        return y1


class CR_D_Net(nn.Module):
    def __init__(self,dim_x,dim_c,avg_dim,activation,gan_model='gan',moment=1):
        super().__init__()
        self.dim_x   = dim_x
        self.dim_c   = dim_c
        self.avg_dim = avg_dim
        self.moment  = moment

        self.data_scale = 0
        
        self.pos_emb = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                     nn.LayerNorm(64,elementwise_affine=False, bias=False),
                                     nn.Linear(64,64),activation(),
                                     nn.Linear(64,128))

        self.scale_emb = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                       nn.LayerNorm(64,elementwise_affine=False, bias=False),
                                       nn.Linear(64,64),activation(),
                                       nn.Linear(64,128))

        self.encoder = nn.Sequential(nn.Linear(dim_x,128),activation(),
                                     nn.Linear(128,128))

        self.score= nn.Sequential(nn.Linear(128,128),activation(),
                                  nn.Linear(128,128),activation(),
                                  nn.Linear(128,  1))

        if gan_model == 'gan':
            self.score.add_module('scale',nn.Sigmoid())

        #self.score.apply(lambda m: init_weights(m,gain=1.2))

    def forward(self,x_in,c_in):
        x0  = self.scale_data(x_in)
        out = self.get_score(x0,c_in)
        return out

    def scale_data(self,x_in):
        out = x_in.pow(self.moment).sum(self.avg_dim)
        out = out/self.data_scale
        return out

    def get_score(self,x_in,c_in):

        pos_emb   = self.  pos_emb(c_in)
        scale_emb = self.scale_emb(c_in)

        z = pos_emb + self.encoder(x_in)*(scale_emb+1)
        out = self.score(z)

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
        
        self.pos_emb = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                     nn.LayerNorm(64,elementwise_affine=False, bias=False),
                                     nn.Linear(64,64),activation(),
                                     nn.Linear(64,64))

        self.scale_emb = nn.Sequential(nn.Linear(dim_c,64),activation(),
                                       nn.LayerNorm(64,elementwise_affine=False, bias=False),
                                       nn.Linear(64,64),activation(),
                                       nn.Linear(64,64))

        self.encoder = nn.Sequential(nn.Linear( 1,64),activation(),
                                     nn.Linear(64,64))

        self.score= nn.Sequential(nn.Linear(64,64),activation(),
                                  nn.Linear(64,64),activation(),
                                  nn.Linear(64, 1))

        if gan_model == 'gan':
            self.score.add_module('scale',nn.Sigmoid())

        #self.score.apply(lambda m: init_weights(m,gain=1.2))

    def forward(self,x_in,c_in):
        x0  = self.scale_data(x_in)
        out = self.get_score(x0,c_in)
        return out

    def scale_data(self,x_in):
        out = x_in.pow(self.moment).sum((1,2,3)).unsqueeze(1)
        out = out/self.data_scale
        return out

    def get_score(self,x_in,c_in):

        pos_emb   = self.  pos_emb(c_in)
        scale_emb = self.scale_emb(c_in)

        z = pos_emb + self.encoder(x_in)*(scale_emb+1)
        out = self.score(z)

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

        g_net_substep = model.g_net_substep

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
        model.epoch_counter += 1

        train_loss = 0.0
        for X,y in trainloader:
            X,y = self._to_dev(X), self._to_dev(y)

            step_switch = update_counter()
            if step_switch == 0:
                gan_step = 'd_step'
            else:
                gan_step = 'g_step'

            optimizer.zero_grad()
            x_fake = model(X)

            loss,loss_gstep = _loss(x_fake=x_fake,x_true=X,gan_step=gan_step)

            loss.backward()

            optimizer.step()

            #weight clipping
            if (gan_type == 'wgan') and (grad_norm_coef < 1.e-5):
                for p in d_net_params():
                    p.data.clamp_(-0.01,0.01)

            train_loss += loss_gstep

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X,y in validloader:
                X,y = self._to_dev(X), self._to_dev(y)

                x_fake = model(X)
                loss,loss_gstep = _loss(x_fake=x_fake,x_true=X,gan_step='g_step')

                val_loss += loss_gstep
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
