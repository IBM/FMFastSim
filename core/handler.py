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
import time

from dataclasses import dataclass
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distributed
from torch.distributed import ReduceOp

import glob

import core

from dataloader.data_handler import DataInfo

import wandb
import plotly.graph_objects as go

from utils.validation_utils import compare_profiles

def ResolveModel(model_info,**kwargs):
    model_type   = model_info['model_type']
    network_type = model_info['network_type']

    if kwargs['_rank'] == 0:
        print('Generative Model:')
        print(model_type)

        print('Encoder-Decoder Network:')
        print(network_type)

    if model_type=='AE':
        gen_model =  core.AEHandler
    elif model_type=='VAE':
        gen_model =  core.VAEHandler
    elif model_type=='MLE':
        gen_model =  core.MLEHandler
    elif model_type=='GAN':
        gen_model =  core.GANHandler
    elif model_type=='Diffusion':
        gen_model =  core.DiffusionHandler
    else:
        raise ValueError

    if network_type=='Mixer':
        network = core.Mixer
    elif network_type=='MixerTF':
        network = core.MixerTF
    elif network_type=='PatchTSMixer':
        network = core.PatchTSMixer
    elif network_type=='CaloDiT':
        network = core.CaloDiT
    #elif network_type=='MixBeast':
    #    network = core.TSMixBeast
    else:
        raise ValueError

    if 'load_model' in model_info:
        load_model = model_info['load_model']
    else:
        load_model = None

    if load_model != None:
        if kwargs['_rank'] == 0:
            print(f'Load model saved in {load_model}')

        saved_model   = torch.load(load_model,map_location='cpu')
        saved_params  = saved_model['all_params']['model_info']
        gen_param     = saved_params['gen_param']
        network_param = saved_params['network_param']

    #overwrite the model parameters if given in the input file
    if 'gen_param' in model_info:
        gen_param = model_info['gen_param']

    if 'network_param' in model_info:
        network_param = model_info['network_param']

    network = network(**network_param)

    if 'save_model' in model_info:
        kwargs['_save_model'] = model_info['save_model']

    ret_model = gen_model(gen_param=gen_param,network=network,**kwargs) 
    ret_model._network_param = network_param

    if load_model != None:
        ret_model.load_model(load_file=load_model)

    ret_model._model.eval()

    return ret_model

@dataclass
class ModelHandler:
    """
    Class to handle building and training of models.
    """
    _run_name: str = None
    _project_name: str = None
    _log_to_wandb: bool = False
    _checkpoint_dir: str = None
    _validation_dir: str = None
    _save_model: bool = True
    _num_gpu: int = 0
    _rank: int = 0
    _device: torch.device = torch.device('cpu')
    _setup_dir: bool = True
    _wandb_entity: str = None

    def __post_init__(self) -> None:
        # Setup Wandb.
        if self._log_to_wandb:
            print("WANDB is running..")
            self._setup_wandb()

        if self._device != torch.device('cpu'):
            self._device = torch.device(self._device)

        self._initialize_model()

        if self._setup_dir:
            if self._checkpoint_dir == None:
                if self._rank == 0:
                    print('Warning: Checkpoint directory is not set')
            else:
                self._save_dir = f"{self._checkpoint_dir}/{self._project_name}/{self._run_name}"
                if self._rank == 0:
                    if not os.path.exists(self._save_dir):
                        os.system(f"mkdir -p {self._save_dir}")
                    else:
                        os.system(f"rm -rf {self._save_dir}/*")

            if self._validation_dir == None:
                if self._rank == 0:
                    print('Warning: Validation directory is not set')
            else:
                self._val_dir  = f"{self._validation_dir}/{self._project_name}/{self._run_name}"
                if self._rank == 0:
                    if not os.path.exists(self._val_dir):
                        os.system(f"mkdir -p {self._val_dir}")
                    else:
                        os.system(f"rm -rf {self._val_dir}/*")

        if self._ddp:
            distributed.barrier()

    def _setup_wandb(self) -> None:
       config = {}
    #    # Add model specific config
    #    config.update(self._get_wandb_extra_config())
       # Reinit flag is needed for hyperparameter tuning. Whenever new training is started, new Wandb run should be created.
       wandb.init(name=self._run_name, project=self._project_name, entity=self._wandb_entity, reinit=True, config=config)

    #def _get_wandb_extra_config(self):
    #    raise NotImplementedError

    def _initialize_model(self):
        """
        Builds a model. Defines self._model. Should be defined by the inherited class.
        """
        self._model.to(self._device)

        torch.manual_seed(3120+self._rank*10)

        if self._num_gpu > 1:
            ddp_model = DDP(self._model, device_ids=[self._device],find_unused_parameters=True)
            ddp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ddp_model)

            self._ddp_model = ddp_model
            self._ddp = True
        else:
            self._ddp = False
        #raise NotImplementedError

    def _set_model_inference(self):
        """
        Inference pipeline. Defines self._decoder. Should be defined by the inherited class.
        """
        return None
        #raise NotImplementedError

    def _train_one_epoch(self, trainloader, validloader, optimizer):
        if self._ddp:
            model = self._ddp_model
            _loss = self._ddp_model.module.loss
        else:
            model = self._model
            _loss = self._model.loss

        train_loss = 0.0
        for X,y in trainloader:
            X,y = self._to_dev(X), self._to_dev(y)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = _loss(y_hat=y_hat, y_true=y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X,y in validloader:
                X,y = self._to_dev(X), self._to_dev(y)

                y_hat = model(X)
                loss = _loss(y_hat, y)

                val_loss += loss.item()
        model.train()

        return train_loss / len(trainloader), val_loss / len(validloader)

    def train_model(self,train_info,train_data,valid_data):

        if self._ddp:
            _model = self._ddp_model
        else:
            _model = self._model

        _model.train()

        optimizer = getattr(optim,train_info['optimizer'])(_model.parameters(), train_info['learning_rate'])

        if 'lr_scheduler' in train_info:
            lr_scheduler = train_info['lr_scheduler']
        else:
            lr_scheduler = {'scheduler':'on_plateau', 
                            'factor':0.5,
                            'patience':30}

        if lr_scheduler['scheduler'] == 'on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', factor=lr_scheduler['factor'], patience=lr_scheduler['patience'], verbose=True)
        elif lr_scheduler['scheduler'] == 'scheduled':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=lr_scheduler['milestones'], gamma= lr_scheduler['gamma'], verbose=False)
        else:
            raise ValueError

        wandb.watch(self._model, log_freq=100, log='all')

        if 'max_valid_events' in train_info:
            max_valid_events = train_info['max_valid_events']
        else:
            max_valid_events = None

        cnt = 0
        num_proc = max(1,self._num_gpu)
        shower_observables_callbacks = []
        if len(train_info['plot_config'][0])==3:
            # from valid dataloader (old dataset 3, not suitable for calochallenge datasets as it is continuous)
            callback_type = ValidationPlotCallback
        elif len(train_info['plot_config'][0])==4:
            # discrete points, for 1M datasets
            callback_type = ValidationPlotCallbackDiscrete
        else:
            raise NotImplementedError

        for args in train_info['plot_config']:
            if cnt%num_proc == self._rank:
                shower_observables_callbacks.append(
                    callback_type(train_info['plot_freq'], self, *args, valid_data, max_valid_events)
                )
            cnt += 1

        # Shower observables (untrained)
        for so_config in shower_observables_callbacks:
            so_config(-1)

        if 'num_workers' in train_info:
            num_workers = train_info['num_workers']
        else:
            num_workers = 0

        n_batch = train_info['batch_size']

        if self._num_gpu > 1:
            shuffle=False
            train_sampler = DistributedSampler(train_data)
            valid_sampler = DistributedSampler(valid_data)
        else:
            shuffle=True
            train_sampler = None
            valid_sampler = None

        trainloader = DataLoader(train_data,batch_size=n_batch,shuffle=shuffle,sampler=train_sampler)
        validloader = DataLoader(valid_data,batch_size=n_batch,shuffle=shuffle,sampler=valid_sampler)

        val_loss_min = float('inf')
        for epoch in range(train_info['epochs']):
            # One epoch
            if self._num_gpu > 1:
                train_sampler.set_epoch(epoch)
                valid_sampler.set_epoch(epoch)

            start_time = time.time()
            train_loss, val_loss = self._train_one_epoch(trainloader, validloader, optimizer)

            if self._ddp:
                loss_values = torch.tensor([train_loss,val_loss],device=self._device)
                distributed.all_reduce(loss_values, op=ReduceOp.SUM)
                train_loss = loss_values[0]
                val_loss   = loss_values[1]

            end_time = time.time()
            epoch_time = end_time - start_time

            if lr_scheduler['scheduler'] == 'on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if self._rank == 0:
                print("Epoch {}:\tTrainLoss: {} \tValidLoss: {}, \tTime: {:.2f}sec.".format(epoch + 1, train_loss, val_loss, epoch_time),flush=True)

            # WandB logs
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch
            })

            # Save model if improvement
            if epoch%10 == 0:
                if val_loss_min > val_loss:
                    if self._rank == 0:
                        print("Saving model..")
                    val_loss_min = val_loss
                    if self._save_model:
                        self.save_model()
            
            # save model anyway every 'save_freq' epochs
            if self._save_model and epoch % train_info['save_freq'] == 0:
                self.save_model(epoch)

            # Shower observables
            for so_config in shower_observables_callbacks:
                so_config(epoch)

        self._model.eval()

    def generate(self):
        """
        Inference pipeline. Should be defined by the inherited class.
        """
        raise NotImplementedError

    def predict(self, dataloader):
        """
        Inference loop using self._decoder.
        """
        self._set_model_inference()
        #self._decoder.to(device)
        self._decoder.eval()

        results = []
        for X,_ in dataloader:
            X = self._to_dev(X)
            with torch.no_grad():
                y_hat = self._decoder(X)
            results.append(y_hat)

        results = torch.cat(results, dim=0)
        return results.cpu().numpy()

    def _to_dev(self,X,dest=None):
        if self._device == torch.device('cpu'):
            return X
        else:
            if dest == None:
                dest = self._device
            if (type(X) == list) or (type(X) == tuple):
                return tuple(x.to(dest) for x in X)
            else:
                return X.to(dest)

    def save_params(self,param_in):
        self._params = param_in

    def save_model(self,epoch=None,save_file=None):
        if save_file == None:
            if epoch == None:
                fname = f"{self._save_dir}/model_weights.pt"
            else:
                fname = f"{self._save_dir}/model_weights_{epoch}.pt"
        else:
            fname = save_file
        self.save_file = fname

        if not hasattr(self,'_params'):
            self._params = None

    def load_model(self,epoch=None,load_file=None):
        if load_file == None:
            if epoch == None:
                fname = f"{self._save_dir}/model_weights.pt"
            else:
                fname = f"{self._save_dir}/model_weights_{epoch}.pt"
        else:
            fname = load_file
        self.load_file = load_file

class ValidationPlotCallback:
    def __init__(self, verbose, handler, angle, energy, geometry, dataloader, max_valid_events=None):
        self.verbose = verbose
        self.handler = handler
        self.val_angle = angle
        self.val_energy = energy
        self.val_geometry = geometry
        self.max_events = max_valid_events
        self._setup(dataloader)

    def _setup(self,dataloader):

        num_total_events = len(dataloader.data_cond_e)
        if self.max_events != None:
            num_total_events = min(self.max_events,num_total_events)

        energy = dataloader.data_cond_e[:num_total_events]*DataInfo().MAX_ENERGY
        angle  = dataloader.data_cond_a[:num_total_events]*DataInfo().MAX_ANGLE
        geo    = dataloader.data_cond_g[:num_total_events]
        geo    = np.array(['Scipb' if geo[i,0] == 1 else 'SiW' for i in range(len(geo))])

        self.scale_method = dataloader.scale_method

        id_a = angle  == self.val_angle
        id_e = energy == self.val_energy
        id_g = geo    == self.val_geometry

        idx = id_a & id_e & id_g

        energy = dataloader.data_energy[:num_total_events]
        cond_e = dataloader.data_cond_e[:num_total_events]
        cond_a = dataloader.data_cond_a[:num_total_events]
        cond_g = dataloader.data_cond_g[:num_total_events]

        self.valid_data = dataloader._torch(energy[idx],cond_e[idx],cond_a[idx],cond_g[idx])

        #delete previous images
        val_dir = self.handler._val_dir
        if os.path.exists(val_dir):
            files = glob.glob(val_dir+'/*.png')
            for f in files:
                os.remove(f)

    def __call__(self, epoch):
        if ((epoch+1) % self.verbose)==0:
            print(f'{self.handler._rank}:Plotting..')

            self.handler._set_model_inference()
            with torch.no_grad():
                x_in = self.handler._to_dev(self.valid_data)
                generated_events = self.handler.generate(x_in)

            str_out = f'data max {self.valid_data[0].max().item():.3f} and '
            str_out+= f'data min {self.valid_data[0].min().item():.3f}'
            print(str_out,flush=True)

            str_out = f'simulation max {generated_events.max().item():.3f} and '
            str_out+= f'simulation min {generated_events.min().item():.3f}'
            print(str_out,flush=True)

            showers          = self.valid_data[0].to('cpu').numpy()
            generated_events =   generated_events.to('cpu').numpy()

            showers          = self.scale_method.inverse_transform(showers,         self.val_energy)
            generated_events = self.scale_method.inverse_transform(generated_events,self.val_energy)

            val_dir = self.handler._val_dir

            compare_profiles(showers,generated_events,self.val_energy,self.val_angle,self.val_geometry,val_dir)

            observable_names = ["LatProf", "LongProf", "PhiProf", "E_tot", "E_cell", "E_cell_non_log", "E_cell_non_log_xlog",
               "E_layer", "LatFirstMoment", "LatSecondMoment", "LongFirstMoment", "LongSecondMoment", "Radial_num_zeroes"]
            plot_names = [
               f"{val_dir}/{metric}_Geo_{self.val_geometry}_E_{self.val_energy}_Angle_{self.val_angle}"
               for metric in observable_names
            ]
            for file in plot_names:
               wandb.log({file: wandb.Image(f"{file}.png")})

            # 3D shower
            N_CELLS_R   = DataInfo().N_CELLS_R
            N_CELLS_PHI = DataInfo().N_CELLS_PHI
            N_CELLS_Z   = DataInfo().N_CELLS_Z
            shower = generated_events[0].reshape(N_CELLS_R, N_CELLS_PHI, N_CELLS_Z)
            r, phi, z, inn = np.stack([x.ravel() for x in np.mgrid[:N_CELLS_R, :N_CELLS_PHI, :N_CELLS_Z]] + [shower.ravel(),], axis=1).T
            phi = phi / phi.max() * 2 * np.pi
            x = r * np.cos(phi)
            y = r * np.sin(phi)

            normalize_intensity_by = 30  # knob for transparency
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker_symbol='square',
                marker_color=[f"rgba(0,0,255,{i*100//normalize_intensity_by/100})" for i in inn],
            )
            go.Figure(trace).write_html(f"{val_dir}/3d_shower.html")
            wandb.log({'shower_{}_{}'.format(self.val_angle, self.val_energy): wandb.Html(f"{val_dir}/3d_shower.html")})

#PLease use this one for dataset 2
class ValidationPlotCallbackDiscrete:
    def __init__(self, verbose, handler, theta, phi, energy, geometry, dataloader, max_valid_events=None):
        self.verbose = verbose
        self.handler = handler
        self.val_phi = phi
        self.val_theta = theta
        self.val_energy = energy
        self.val_geometry = geometry
        self.max_events = max_valid_events
        self._setup(dataloader)

    def _setup(self,dataloader):

        num_total_events = len(dataloader.data_cond_e)
        if self.max_events != None:
            num_total_events = min(self.max_events,num_total_events)

        self.scale_method = dataloader.scale_method

        val_energy = self.scale_method.transform_energy(np.array([self.val_energy,]))
        val_theta = self.scale_method.transform_theta(np.array([self.val_theta,]))
        val_phi = self.scale_method.transform_phi(np.array([self.val_phi,]))
        val_geo = np.array([0, 1] if self.val_geometry=="SiW" else [1, 0])

        id_e = np.isclose(dataloader.data_cond_e, val_energy)
        id_t = np.isclose(dataloader.data_cond_theta, val_theta)
        id_p = np.isclose(dataloader.data_cond_phi, val_phi).sum(axis=-1)==len(val_phi)
        id_g = np.isclose(dataloader.data_cond_g, val_geo).sum(axis=-1)==len(val_geo)

        idx = id_e & id_t & id_p & id_g
        idx = idx[:num_total_events]

        energy = dataloader.data_energy[idx]
        cond_e = dataloader.data_cond_e[idx]
        cond_phi = dataloader.data_cond_phi[idx]
        cond_theta = dataloader.data_cond_theta[idx]
        cond_g = dataloader.data_cond_g[idx]

        self.valid_data = dataloader._torch(energy,cond_e,cond_theta,cond_phi,cond_g)

        #delete previous images
        val_dir = self.handler._val_dir
        if os.path.exists(val_dir):
            files = glob.glob(val_dir+'/*.png')
            for f in files:
                os.remove(f)

    def __call__(self, epoch):
        if ((epoch+1) % self.verbose)==0:
            print(f'{self.handler._rank}:Plotting..')

            self.handler._set_model_inference()
            with torch.no_grad():
                x_in = self.handler._to_dev(self.valid_data)
                generated_events = self.handler.generate(x_in)

            str_out = f'data max {self.valid_data[0].max().item():.3f} and '
            str_out+= f'data min {self.valid_data[0].min().item():.3f}'
            print(str_out,flush=True)

            str_out = f'simulation max {generated_events.max().item():.3f} and '
            str_out+= f'simulation min {generated_events.min().item():.3f}'
            print(str_out,flush=True)

            showers          = self.valid_data[0].to('cpu').numpy()
            generated_events =   generated_events.to('cpu').numpy()

            showers          = self.scale_method.inverse_transform(showers,         self.val_energy)
            generated_events = self.scale_method.inverse_transform(generated_events,self.val_energy)

            # moved here as energy descaling is outside of ds2-scaling
            ecut = 0.0151
            if(ecut > 0):
                showers[showers < ecut ] = 0
                generated_events[generated_events < ecut ] = 0

            val_dir = self.handler._val_dir

            compare_profiles(showers,generated_events,self.val_energy,self.val_theta,
                             self.val_geometry,val_dir,particle_phi=self.val_phi)

            observable_names = ["LatProf", "LongProf", "PhiProf", "E_tot", "E_cell", "E_cell_non_log", "E_cell_non_log_xlog",
               "E_layer", "LatFirstMoment", "LatSecondMoment", "LongFirstMoment", "LongSecondMoment", "Radial_num_zeroes"]
            plot_names = [
               f"{val_dir}/{metric}_Geo_{self.val_geometry}_E_{self.val_energy}_Theta_{self.val_theta}_Phi_{self.val_phi}"
               for metric in observable_names
            ]
            for file in plot_names:
               wandb.log({file: wandb.Image(f"{file}.png")})

            try:
                # 3D shower
                N_CELLS_R   = DataInfo().N_CELLS_R
                N_CELLS_PHI = DataInfo().N_CELLS_PHI
                N_CELLS_Z   = DataInfo().N_CELLS_Z
                shower = generated_events[0].reshape(N_CELLS_R, N_CELLS_PHI, N_CELLS_Z)
                r, phi, z, inn = np.stack([x.ravel() for x in np.mgrid[:N_CELLS_R, :N_CELLS_PHI, :N_CELLS_Z]] + [shower.ravel(),], axis=1).T
                phi = phi / phi.max() * 2 * np.pi
                x = r * np.cos(phi)
                y = r * np.sin(phi)

                normalize_intensity_by = 30  # knob for transparency
                trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker_symbol='square',
                    marker_color=[f"rgba(0,0,255,{i*100//normalize_intensity_by/100})" for i in inn],
                )
                go.Figure(trace).write_html(f"{val_dir}/3d_shower.html")
                wandb.log({'shower_{}_{}'.format(self.val_theta, self.val_energy): wandb.Html(f"{val_dir}/3d_shower.html")})
            except:
                print("skip 3d shower, wrong pixel range")