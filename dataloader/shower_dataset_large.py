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

# Standard
import os
import sys
import warnings
import time

# Third Party
import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np

import h5py

warnings.filterwarnings("ignore")

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


###############################################################
#   Define Problem Specific Parameters
#   CONSTATNS is used as a Global Variable
###############################################################
CONSTANTS = AttributeDict()

"""
Geometry constants.
"""
# Number of calorimeter layers (z-axis segmentation).
CONSTANTS.ORG_N_CELLS_Z = 45
# Segmentation in the r,phi direction.
CONSTANTS.ORG_N_CELLS_R = 18
CONSTANTS.N_CELLS_PHI = 50
# Cell size in the r and z directions
CONSTANTS.SIZE_R = 2.325
CONSTANTS.SIZE_Z = 3.4

# In case of restricting data (Count from 0, Including)
CONSTANTS.R_HIGH = None  # [0, 17]
CONSTANTS.Z_LOW = None  # [0, 44]
CONSTANTS.Z_HIGH = None  # [0, 44]
if CONSTANTS.R_HIGH is None: CONSTANTS.R_HIGH = 17
if CONSTANTS.Z_LOW is None: CONSTANTS.Z_LOW = 0
if CONSTANTS.Z_HIGH is None: CONSTANTS.Z_HIGH = 44
assert CONSTANTS.R_HIGH < CONSTANTS.ORG_N_CELLS_R
assert (CONSTANTS.Z_HIGH - CONSTANTS.Z_LOW) < CONSTANTS.ORG_N_CELLS_Z
CONSTANTS.N_CELLS_R = CONSTANTS.R_HIGH + 1
CONSTANTS.N_CELLS_Z = CONSTANTS.Z_HIGH - CONSTANTS.Z_LOW + 1

#Maximum Energy and Angles
CONSTANTS.MAX_ENERGY = 256
CONSTANTS.MAX_ANGLE  = 70

#############################################
# Copy the global parameters
#############################################
ORG_N_CELLS_Z = CONSTANTS.ORG_N_CELLS_Z
ORG_N_CELLS_R = CONSTANTS.ORG_N_CELLS_R
N_CELLS_PHI   = CONSTANTS.N_CELLS_PHI
# Cell size in the r and z directions
SIZE_R = CONSTANTS.SIZE_R
SIZE_Z = CONSTANTS.SIZE_Z

# In case of restricting data (Count from 0, Including)
R_HIGH    = CONSTANTS.R_HIGH
Z_LOW     = CONSTANTS.Z_LOW
Z_HIGH    = CONSTANTS.Z_HIGH
N_CELLS_R = CONSTANTS.N_CELLS_R
N_CELLS_Z = CONSTANTS.N_CELLS_Z

MAX_ENERGY = CONSTANTS.MAX_ENERGY
MAX_ANGLE  = CONSTANTS.MAX_ANGLE 

#data loader
class Dataset(Dataset):
    def __init__(
        self,
        data_split   = [0.8,0.1,0.1],
        root_path    = '.',
        geo          = ['SiW'],
        angles       = [70],
        energies     = [64,128,256],
        use_cond_info= True,
        split="train",
        scale_method = None,
        max_num_events = None,
        max_local_data = 20000,
        **kwargs
    ):
        assert split in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[split]

        self.data_split   = data_split

        self.root_path = root_path

        self.geo      = geo
        self.angles   = angles
        self.energies = energies

        self.use_cond_info = use_cond_info

        self.scale_method = scale_method

        self.max_num_events = max_num_events
        self.max_local_data = max_local_data
    
        self.__read_data__()

    def __read_data__(self):
        energies_data = []
        cond_e_data = []
        cond_angle_data = []
        cond_geo_data = []

        if max(self.energies) > MAX_ENERGY:
            print('Warning: maximum energy in the data is larger than MAX_ENERGY')
            print(f'data max: {max(self.energies)} v.s. MAX_ENERGY {MAX_ENERGY}')

        if max(self.angles) > MAX_ANGLE:
            print('Warning: maximum angle in the data is larger than MAX_ANGLE')
            print(f'data max: {max(self.angle)} v.s. MAX_ENERGY {MAX_ANGLE}')
            
        max_local_data = self.max_local_data

        #First check the total file size
        total_num_events = 0
        local_events = []
        for geo in self.geo:
            dir_geo = self.root_path + "/"
            for angle_particle in self.angles:
                for energy_particle in self.energies:
                    f_name = f"detector_SiW_eM_{energy_particle}GeV_angle{angle_particle}_250kevents_part1.h5"
                    F_name = dir_geo+f_name

                    with h5py.File(F_name, 'r') as hdf5_file:
                        h5 = hdf5_file[str(energy_particle)]

                        h5_events = h5.shape[0]
                        if self.max_num_events == None:
                            num_events = h5_events
                        else:
                            num_events = min(self.max_num_events,h5_events)
                        #print(f'{num_events} number of events in {f_name}')

                        #data split
                        ntrain = int(self.data_split[0]*num_events)
                        nvalid = int(self.data_split[1]*num_events)

                        data_bd = [0,ntrain,ntrain+nvalid,num_events]

                        local_events.append([data_bd[self.set_type],data_bd[self.set_type+1]])
                        total_num_events += local_events[-1][1] - local_events[-1][0]

        #Pre-allocate the array
        energies_data = np.zeros((total_num_events,N_CELLS_R,N_CELLS_PHI,N_CELLS_Z),dtype=np.float32)

        file_idx  = 0
        start_idx = 0
        for geo in self.geo:
            dir_geo = self.root_path + "/"
            for angle_particle in self.angles:
                for energy_particle in self.energies:
                    f_name = f"detector_SiW_eM_{energy_particle}GeV_angle{angle_particle}_250kevents_part1.h5"
                    F_name = dir_geo+f_name

                    local_id0 = local_events[file_idx][0]
                    local_id1 = local_events[file_idx][1]
                    local_num_data = local_id1-local_id0

                    #print(f_name)
                    #print(start_idx)
                    #print(local_id0,local_id1)
                    #print(local_num_data)
                    
                    t0 = time.time()
                    with h5py.File(F_name, 'r') as hdf5_file:
                        h5 = hdf5_file[str(energy_particle)]

                        # events in Num_data x Radial x Azimuthal x Vertical
                        if local_num_data > max_local_data:
                            id0 = local_id0
                            for k in range(int(local_num_data/max_local_data)):
                                id1 = id0 + max_local_data

                                local_energy = h5[id0:id1].astype(np.float32)
                                local_energy = local_energy[:, :(R_HIGH + 1), :, Z_LOW:(Z_HIGH + 1)]
                                id0 = id1

                                local_energy = self.scale_method.transform(local_energy,energy_particle)

                                end_idx = start_idx + max_local_data
                                energies_data[start_idx:end_idx] = local_energy
                                start_idx = end_idx
                                #print(end_idx)
                            if id0 < local_id1:
                                local_energy = h5[id0:local_id1].astype(np.float32)
                                local_energy = local_energy[:, :(R_HIGH + 1), :, Z_LOW:(Z_HIGH + 1)]

                                local_energy = self.scale_method.transform(local_energy,energy_particle)

                                end_idx = start_idx + local_id1-id0
                                energies_data[start_idx:end_idx] = local_energy
                                start_idx = end_idx
                                #print(end_idx)
                        else:
                            local_energy = h5[local_id0:local_id1].astype(np.float32)
                            local_energy = local_energy[:, :(R_HIGH + 1), :, Z_LOW:(Z_HIGH + 1)]

                            local_energy = self.scale_method.transform(local_energy,energy_particle)
                    
                            end_idx = start_idx + local_num_data
                            energies_data[start_idx:end_idx] = local_energy
                            start_idx = end_idx
                    #print('data retrieval time:',time.time()-t0)


                    # Bring the conditions b/w [0,1]
                    cond_e_data    .append([energy_particle / MAX_ENERGY] * local_num_data)
                    cond_angle_data.append([angle_particle  / MAX_ANGLE ] * local_num_data)

                    # build the geometry condition vector (1 hot encoding vector)
                    if geo == "SiW":
                        cond_geo_data.append([[0, 1]] * local_num_data)
                    if geo == "SciPb":
                        cond_geo_data.append([[1, 0]] * local_num_data)

        # return numpy arrays
        cond_e_data     = np.concatenate(cond_e_data)
        cond_angle_data = np.concatenate(cond_angle_data)
        cond_geo_data   = np.concatenate(cond_geo_data)

        self.data_energy = torch.from_numpy(energies_data)
        self.data_cond_e = torch.from_numpy(cond_e_data)
        self.data_cond_a = torch.from_numpy(cond_angle_data)
        self.data_cond_g = torch.from_numpy(cond_geo_data)

    def __getitem__(self, index):

        if self.use_cond_info:
            return self._torch(self.data_energy[index],
                               self.data_cond_e[index:index+1],
                               self.data_cond_a[index:index+1],
                               self.data_cond_g[index]),        \
                   self._torch(self.data_energy[index])[0]
        else:
            return self._torch(self.data_energy[index]),        \
                   self._torch(self.data_energy[index])[0]

    def __len__(self):
        return len(self.data_energy)

    def _torch(self,*dfs):
        if torch.is_tensor(dfs[0]):
            return tuple(x.float() for x in dfs)
        else:
            return tuple(torch.from_numpy(x).float() for x in dfs)
