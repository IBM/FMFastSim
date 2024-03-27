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

import torch
torch.manual_seed(1231)

import torch.distributed as distributed

from utils.data_scaling import voxel_scaling

from core import ResolveModel
from dataloader.data_handler import DataInfo

def cleanup():
    print('Exiting..')
    if os.path.exists(f"{VALID_DIR}"):
        os.system(f"rm -r {VALID_DIR}")


def main():
    # 0. Read YAML file
    all_params = yaml.safe_load(open(sys.argv[1],'r'))

    exp_info   = all_params['experiment']
    data_info  = all_params['data_info']
    model_info = all_params['model_info']
    train_info = all_params['train_info']
    scale_info = all_params['scale_info']

    #load gpu info
    use_gpu = False
    use_ddp = False
    if 'gpu_info' in all_params:
        gpu_info = all_params['gpu_info']
        if 'use_gpu' in gpu_info:
            use_gpu = gpu_info['use_gpu']
        if 'use_ddp' in gpu_info:
            use_ddp = gpu_info['use_ddp']
        if 'max_num_ddp_gpu' in gpu_info:
            max_ddp_gpu = gpu_info['max_num_ddp_gpu']
        else:
            max_ddp_gpu = 100000

    # 1. Set GPU memory limits.
    if use_gpu:
        if use_ddp:
            num_gpu = min(torch.cuda.device_count(),max_ddp_gpu)
        else:
            num_gpu = 1
    else:
        num_gpu = 0

    # 2. Data loading/preprocessing
    #   define scaling function
    scale_method = voxel_scaling(**scale_info)

    print('Start Loading data')
    Data = DataInfo(data_info['dataloader'])
    train_data = Data.get_dataset(**data_info,split='train',scale_method=scale_method)
    valid_data = Data.get_dataset(**data_info,split='val'  ,scale_method=scale_method)
    print('End Loading data')

    if num_gpu == 0:
        device = 'cpu'
        rank = 0
    else:
        if use_ddp:
            distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

            rank = int(os.environ["LOCAL_RANK"])

            if rank == 0:
                print(f'Use Distributed DataParallel on {num_gpu} gpus')
        else:
            rank = 0

        device = rank
            
    #experiment info
    study_name = exp_info['study_name']
    run_name   = exp_info['run_name']

    checkpoint_dir = exp_info['checkpoint_dir']
    validation_dir = exp_info['validation_dir']

    # 3. Manufacture model handler.
    print('Start Loading network')
    model_handler = ResolveModel(model_info,
        _run_name=run_name, _project_name=study_name, _log_to_wandb=False,
        _checkpoint_dir=checkpoint_dir,_validation_dir=validation_dir,
        _device=device,_rank=rank,_num_gpu=num_gpu
        )
    model_handler.save_params(all_params)

    # 4. Train model.
    print('Start Training')
    model_handler.train_model(train_info,train_data,valid_data)

    # Note : One history object can be used to plot the loss evaluation as function of the epochs. Remember that the
    # function returns a list of those objects. Each of them represents a different fold of cross validation.

    #Save trained model
    if 'save_file' in model_info:
        save_file = model_info['save_file']
    else:
        save_file = None

    model_handler.save_model(save_file=save_file)

    if use_ddp:
        distributed.destroy_process_group()

if __name__ == "__main__":
    sys.exit(main())
