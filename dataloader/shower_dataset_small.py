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
        max_local_data = 50000,
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

        for geo in self.geo:
            dir_geo = self.root_path + "/" + geo + "/"
            for angle_particle in self.angles:
                f_name = f"{geo}_angle_{angle_particle}.h5"
                f_name = dir_geo + f_name
                h5 = h5py.File(f_name, "r")
                for energy_particle in self.energies:
                    events = np.array(h5[f"{energy_particle}"])
                    num_events = events.shape[0]
                    if num_events > max_local_data:
                        num_events = max_local_data
                        events = events[:num_events]
                   
                    events = self.scale_method.transform(events,energy_particle)
                    # For restricted geometry
                    events = events[:, :(R_HIGH + 1), :, Z_LOW:(Z_HIGH + 1)]

                    ntrain = int(self.data_split[0]*num_events)
                    nvalid = int(self.data_split[1]*num_events)

                    data_bd = [0,ntrain,ntrain+nvalid,num_events]

                    # events in Num_data x Radial x Azimuthal x Vertical
                    events = events[data_bd[self.set_type]:data_bd[self.set_type+1],:,:,:]

                    energies_data.append(events)

                    # Bring the conditions b/w [0,1]
                    cond_e_data    .append([energy_particle / MAX_ENERGY] * events.shape[0])
                    cond_angle_data.append([angle_particle  / MAX_ANGLE ] * events.shape[0])

                    # build the geometry condition vector (1 hot encoding vector)
                    if geo == "SiW":
                        cond_geo_data.append([[0, 1]] * events.shape[0])
                    if geo == "SciPb":
                        cond_geo_data.append([[1, 0]] * events.shape[0])

        # return numpy arrays
        energies_data   = np.concatenate(energies_data)
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
