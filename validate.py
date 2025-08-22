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
from core.handler import ValidationPlotCallback
from dataloader.data_handler import DataInfo
from utils.validation_utils import compare_profiles

def main():
    # 0. Read YAML file
    all_params = yaml.safe_load(open(sys.argv[1],'r'))

    data_info = all_params['data_info']
    load_info = all_params['model_info']
    valid_info= all_params['valid_info']

    #read model parameters
    model_file  = load_info['trained_model']
    load_params = torch.load(model_file,map_location='cpu')['all_params']

    model_info = load_params['model_info']
    scale_info = load_params['scale_info']

    #override model loading
    model_info['load_file'] = model_file

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

    # Data loading/preprocessing
    # define scaling function
    scale_method = voxel_scaling(**scale_info)

    print('Start Loading data')
    Data = DataInfo(data_info['dataloader'])
    valid_data = Data.get_dataset(**data_info,split='test',scale_method=scale_method)
    print('End Loading data')

    # Manufacture model handler.
    print('Start Loading network')
    model_handler = ResolveModel(model_info,
        _device=device,_rank=rank,_num_gpu=num_gpu,
        _setup_dir=False
        )

    # Generating Validation plots
    print('Start Validation')

    val_dir   = valid_info['plot_loc']
    if not os.path.exists(val_dir):
        os.system(f"mkdir -p {val_dir}")
    else:
        os.system(f"rm -rf {val_dir}/*")

    model_handler._val_dir = val_dir

    for args in valid_info['valid_plots']:
        valid_func = ValidationPlotCallback(1,model_handler,*args, valid_data, save_showers=True, keep_previous=True)
        valid_func(-1)

if __name__ == "__main__":
    sys.exit(main())
