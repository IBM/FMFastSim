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

from core.handler import ModelHandler

from core.generative_model.generative_decoder import Decoder_Distribution
from core.generative_model.regularizer import regularizer

class VAE(nn.Module):
    def __init__(self, network, kl_coef=1.0, reg_coef=0.0, reg_model = 'none', decoder_distribution = 'gamma'):
        super().__init__()

        self.model = network
        self.kl_coef = kl_coef

        self.regularizer = regularizer(reg_model,reg_coef)

        dim_r = self.model.dim_r
        dim_a = self.model.dim_a
        dim_v = self.model.dim_v

        pdf = decoder_distribution

        self.gen_decoder = Decoder_Distribution(dim_r=dim_r,dim_a=dim_a,dim_v=dim_v,pdf=pdf)

    def forward(self, inputs):

        x_input, cond_var = self.prepare_input(X)

        if x_input == None:
            z = torch.zeros_like(self.model.decoder_input,device=e_input.device) \
                     .repeat_interleave(cond_var.size(0),dim=0)
            z.normal_()
        else:
            mu, var = self.model.encoding(x_input)
            self.kl_loss  = (var.log() + (1+mu.pow(2))/var).mean()

            z = mu + torch.randn_like(var)*var.sqrt()

        x0    = self.model.decoding(z,cond_var)
        x_out = self.gen_decoder(x0)

        return x_out

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

    def loss(self,y_hat=None,y_true=None):
        nll_loss = self.gen_decoder.Loss(y_hat=y_hat,y_true=y_true)
        vae_loss = nll_loss + self.kl_loss*self.kl_coef
        reg      = self.regularizer.compute(y_hat,y_true)
        return vae_loss+reg


class VAEHandler(ModelHandler):
    def __init__(self, gen_param, network, **kwargs):

        self._gen_param = gen_param

        self._model = VAE(network=network,**gen_param)
        self._loss  = self._model.loss

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

    def save_model(self,epoch=None,save_file=None):

        super().save_model(epoch=epoch,save_file=save_file)

        if self._rank == 0:
            torch.save({'network'    :self._model.model      .state_dict(),
                        'gen_decoder':self._model.gen_decoder.state_dict(),
                        'all_params' :self._params},
                       self.save_file)

    def load_model(self,epoch=None,load_file=None):

        super().load_model(epoch=epoch,load_file=load_file)

        model_load = torch.load(self.load_file,map_location=self._device)

        t0 = model_load['network']
        t1 = model_load['gen_decoder']

        self._model.model      .load_state_dict(t0)
        self._model.gen_decoder.load_state_dict(t1)
