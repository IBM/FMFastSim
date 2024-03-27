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

from core.generative_model.prior_dist import prior_dist
from core.generative_model.generative_decoder import Decoder_Distribution
from core.generative_model.regularizer import regularizer

class MLE(nn.Module):
    def __init__(self, network, prior_distribution = 'std', decoder_distribution = 'gamma', fix_mixture = False,
                 reg_coef = 0.0, reg_model = 'none'):
        super().__init__()

        self.model = network

        dim_r = self.model.dim_r
        dim_a = self.model.dim_a
        dim_v = self.model.dim_v

        print('Prior Distribution is '+p_dist)

        self.regularizer = regularizer(reg_model,reg_coef)

        self.gen_decoder = Decoder_Distribution(dim_r=dim_r,dim_a=dim_a,dim_v=dim_v,pdf=pdf,fix_mix=fix_mix)

        #Define Prior
        self.prior = prior_dist(p_dist,self.model.decoder_input)

    def forward(self, inputs):
        (x_input, e_input, angle_input, geo_input) = inputs

        if e_input.dim() == 1:
            e_input     = e_input    .unsqueeze(1)
            angle_input = angle_input.unsqueeze(1)

        c_input = torch.cat([e_input,angle_input,geo_input],dim=1)

        z = self.prior(nbatch=c_input.size(0))

        x0    = self.model.decoding(z,c_input)
        x_out = self.gen_decoder(x0)

        return x_out

    def generate(self,inputs):
        x_out = self.forward(inputs)
        return x_out

    def loss(self,y_hat=None,y_true=None):
        nll_loss = self.gen_decoder.Loss(y_hat=y_hat,y_true=y_true)
        reg      = self.regularizer.compute(y_hat,y_true)
        return nll_loss+reg

class MLEHandler(ModelHandler):
    def __init__(self, gen_param, network, **kwargs):

        self._gen_param = gen_param

        self._model = MLE(gen_param,network=network)
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
                        'prior'      :self._model.prior      .state_dict(),
                        'gen_decoder':self._model.gen_decoder.state_dict(),
                        'all_params' :self._params},
                       self.save_file)

    def load_model(self,epoch=None,load_file=None):

        super().load_model(epoch=epoch,load_file=load_file)

        model_load = torch.load(self.load_file,map_location=self._device)

        t0 = model_load['network']
        t1 = model_load['prior']
        t2 = model_load['gen_decoder']

        self._model.model      .load_state_dict(t0)
        self._model.prior      .load_state_dict(t1)
        self._model.gen_decoder.load_state_dict(t2)
