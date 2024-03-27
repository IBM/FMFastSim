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
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np

import torch
torch.manual_seed(1231)

from utils.data_scaling import voxel_scaling

from core import ResolveModel
from dataloader.data_handler import DataInfo

def main():
    # 0. Read YAML file
    all_params = yaml.safe_load(open(sys.argv[1],'r'))

    gen_info  = all_params['gen_info']
    data_info = all_params['data_info']
    load_info = all_params['model_info']

    #read model parameters
    model_file  = load_info['trained_model']
    load_params = torch.load(model_file,map_location='cpu')['all_params']

    model_info = load_params['model_info']
    scale_info = load_params['scale_info']

    #override model loading
    model_info['load_model'] = model_file

    #load gpu info
    use_gpu = False
    use_ddp = False
    if 'gpu_info' in all_params:
        gpu_info = all_params['gpu_info']
        if 'use_gpu' in gpu_info:
            use_gpu = gpu_info['use_gpu']

    rank = 0
    if use_gpu: 
        num_gpu = 1 #DDP is not set up for validation
        device = rank
    else:
        num_gpu = 0
        device = 'cpu'

    # define scaling function
    scale_method = voxel_scaling(**scale_info)

    # Data not loaded. DataInfo is loaded to read in problem specific variables (CONSTANTS)
    Data = DataInfo(data_info['dataloader'])

    # Manufacture model handler.
    print('Start Loading network')
    model_handler = ResolveModel(model_info,
        _device=device,_rank=rank,_num_gpu=num_gpu,
        _setup_dir=False
        )

    # Start MC sampling
    print('Start Generating Simulations')

    num_samples = gen_info['num_samples']
    cond_vars   = gen_info['cond_vars']

    gen_dir = gen_info['gen_loc']
    if not os.path.exists(gen_dir):
        os.system(f"mkdir -p {gen_dir}")
    else:
        os.system(f"rm -rf {gen_dir}/*")

    for _angle, _energy, _geo in cond_vars:
        cond_e = torch.tensor([_energy / Data.MAX_ENERGY]*num_samples,dtype=torch.float).unsqueeze(1)
        cond_a = torch.tensor([_angle  / Data.MAX_ANGLE ]*num_samples,dtype=torch.float).unsqueeze(1)

        if _geo == "SiW":
            cond_g = torch.stack([torch.tensor([0,1],dtype=torch.float)]*num_samples,dim=0)
        elif _geo == "SciPb":
            cond_g = torch.stack([torch.tensor([1,0],dtype=torch.float)]*num_samples,dim=0)
        else:
            print('Geometry is not implemented')
            raise ValueError

        c_input = model_handler._to_dev((cond_e,cond_a,cond_g))
        x_input = (None,)+c_input

        with torch.no_grad():
            generated_events = model_handler.generate(x_input)
    
        generated_events = generated_events.to('cpu').numpy()
        generated_events = scale_method.inverse_transform(generated_events,_energy)

        file_name = f"{gen_dir}/generated_event_for_{_energy}GeV_{_angle}Deg_on_{_geo}.pt"
        torch.save(generated_events,file_name)

if __name__ == "__main__":
    sys.exit(main())
