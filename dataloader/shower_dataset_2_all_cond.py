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
CONSTANTS.ORG_N_CELLS_R = 9 #18
CONSTANTS.N_CELLS_PHI = 16 #50
# Cell size in the r and z directions
CONSTANTS.SIZE_R = 2.325
CONSTANTS.SIZE_Z = 3.4

# In case of restricting data (Count from 0, Including)
CONSTANTS.R_HIGH = 8 #None  # [0, 17]
CONSTANTS.Z_LOW = None  # [0, 44]
CONSTANTS.Z_HIGH = 44 # None  # [0, 44]
if CONSTANTS.R_HIGH is None: CONSTANTS.R_HIGH = ORG_N_CELLS_R - 1
if CONSTANTS.Z_LOW is None: CONSTANTS.Z_LOW = 0
if CONSTANTS.Z_HIGH is None: CONSTANTS.Z_HIGH = ORG_N_CELLS_Z - 1
assert CONSTANTS.R_HIGH < CONSTANTS.ORG_N_CELLS_R
assert (CONSTANTS.Z_HIGH - CONSTANTS.Z_LOW) < CONSTANTS.ORG_N_CELLS_Z
CONSTANTS.N_CELLS_R = CONSTANTS.R_HIGH + 1
CONSTANTS.N_CELLS_Z = CONSTANTS.Z_HIGH - CONSTANTS.Z_LOW + 1

#Maximum Energy and Angles
CONSTANTS.MAX_ENERGY = 1000 # changed to 1 TeV
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
        valid_path   = None,
        geo          = ['SiW'],
        use_cond_info= True,
        split="train",
        scale_method = None,
        max_num_events = None,
        max_local_data = 100000,
        **kwargs
    ):
        assert split in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[split]

        self.data_split   = data_split
        self.root_path = root_path

        if valid_path != None and self.set_type == 0:
            self.data_split = [self.data_split[0]+self.data_split[1], 0, self.data_split[2]]
        elif valid_path != None and self.set_type == 1:
            self.data_split = [0, 1.0, 0]
            self.root_path = valid_path


        self.geo      = geo

        self.use_cond_info = use_cond_info
        self.scale_method = scale_method
        self.max_num_events = max_num_events
        self.max_local_data = max_local_data
    
        self.__read_data__()

    def __read_data__(self):
        energies_data = []
        cond_e_data = []
        cond_phi_train = []
        cond_theta_train = []
        cond_geo_data = []
            
        max_local_data = self.max_local_data

        #get all the file names
        file_list = []
        for file in os.listdir(self.root_path):
            if file.endswith(".h5"):
                file_list.append(file)

        #First check the total file size
        total_num_events = 0
        local_events = []

        dir_geo = self.root_path + "/"
        for f_name in file_list:
            F_name = dir_geo+f_name
            with h5py.File(F_name, 'r') as hdf5_file:
                h5 = hdf5_file['showers']

                h5_events = h5.shape[0]
                if self.max_num_events == None:
                    num_events = h5_events
                else:
                    num_events = min(self.max_num_events,h5_events)
                print(f'{num_events} number of events in {f_name}')

                #data split
                ntrain = int(self.data_split[0]*num_events)
                nvalid = int(self.data_split[1]*num_events)

                data_bd = [0,ntrain,ntrain+nvalid,num_events]

                local_events.append([data_bd[self.set_type],data_bd[self.set_type+1]])
                total_num_events += local_events[-1][1] - local_events[-1][0]

        #Pre-allocate the array
        energies_data = np.zeros((total_num_events,N_CELLS_R,N_CELLS_PHI,N_CELLS_Z),dtype=np.float32)
        cond_e_data = np.zeros((total_num_events,),dtype=np.float32)
        theta_data = np.zeros((total_num_events,),dtype=np.float32)
        phi_data = np.zeros((total_num_events, 2),dtype=np.float32) # each phi is processed into 2 values

        file_idx  = 0
        start_idx = 0

        dir_geo = self.root_path + "/"
        for f_name in file_list:
            F_name = dir_geo+f_name

            local_id0 = local_events[file_idx][0]
            local_id1 = local_events[file_idx][1]
            local_num_data = local_id1-local_id0

            print(f_name)
            #print(start_idx)
            #print(local_id0,local_id1)
            # print(local_num_data)
            angle_particle = 70
            geo = "SiW"
            
            t0 = time.time()
            with h5py.File(F_name, 'r') as hdf5_file:
                h5_showers = hdf5_file['showers']
                h5_energy_particle = hdf5_file['incident_energy']
                h5_theta = hdf5_file['incident_theta']
                h5_phi = hdf5_file['incident_phi']

                # events in Num_data x Radial x Azimuthal x Vertical
                if local_num_data > max_local_data:
                    id0 = local_id0
                    for k in range(int(local_num_data/max_local_data)):
                        
                        id1 = id0 + max_local_data

                        local_energy, local_energy_particle, local_theta, local_phi = self.read_local(id0, id1, h5_showers, h5_energy_particle, h5_theta, h5_phi)

                        id0 = id1

                        end_idx = start_idx + max_local_data
                        energies_data[start_idx:end_idx] = local_energy
                        cond_e_data[start_idx:end_idx] = local_energy_particle
                        theta_data[start_idx:end_idx] = local_theta
                        phi_data[start_idx:end_idx] = local_phi
                        start_idx = end_idx
                        #print(end_idx)
                        
                    if id0 < local_id1:

                        local_energy, local_energy_particle, local_theta, local_phi = self.read_local(id0, local_id1, h5_showers, h5_energy_particle, h5_theta, h5_phi)

                        end_idx = start_idx + local_id1-id0
                        energies_data[start_idx:end_idx] = local_energy
                        cond_e_data[start_idx:end_idx] = local_energy_particle
                        theta_data[start_idx:end_idx] = local_theta
                        phi_data[start_idx:end_idx] = local_phi
                        start_idx = end_idx
                        #print(end_idx)
                else:

                    local_energy, local_energy_particle, local_theta, local_phi = self.read_local(local_id0, local_id1, h5_showers, h5_energy_particle, h5_theta, h5_phi)

                    end_idx = start_idx + local_num_data
                    energies_data[start_idx:end_idx] = local_energy
                    cond_e_data[start_idx:end_idx] = local_energy_particle
                    theta_data[start_idx:end_idx] = local_theta
                    phi_data[start_idx:end_idx] = local_phi
                    start_idx = end_idx
                    
            #print('data retrieval time:',time.time()-t0)

            # build the geometry condition vector (1 hot encoding vector)
            if geo == "SiW":
                cond_geo_data.append([[0, 1]] * local_num_data)
            if geo == "SciPb":
                cond_geo_data.append([[1, 0]] * local_num_data)

        cond_geo_data   = np.concatenate(cond_geo_data)

        self.data_energy = torch.from_numpy(energies_data)
        self.data_cond_e = torch.from_numpy(cond_e_data)
        self.data_cond_theta = torch.from_numpy(theta_data)
        self.data_cond_phi = torch.from_numpy(phi_data)
        self.data_cond_g = torch.from_numpy(cond_geo_data)


    def __getitem__(self, index):

        if self.use_cond_info:
            return self._torch(self.data_energy[index],
                               self.data_cond_e[index:index+1],
                               self.data_cond_theta[index:index+1],
                               self.data_cond_phi[index],
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

    def read_local(self, id0, id1, h5_showers, h5_energy_particle, h5_theta, h5_phi):

        #read local
        local_energy = h5_showers[id0:id1].astype(np.float32)/1000.0
        local_energy = local_energy[:, :(R_HIGH + 1), :, Z_LOW:(Z_HIGH + 1)]
        local_energy_particle = h5_energy_particle[id0:id1].astype(np.float32)/1000.0
        local_theta = h5_theta[id0:id1].astype(np.float32)
        local_phi = h5_phi[id0:id1].astype(np.float32)

        # reshape for preprocessing
        local_energy = local_energy.reshape(-1,N_CELLS_R*N_CELLS_PHI*N_CELLS_Z)
        local_energy_particle = local_energy_particle.reshape(-1,1)
        local_theta = local_theta.reshape(-1,1)
        local_phi = local_phi.reshape(-1,1)

        # preprocess
        local_energy = self.scale_method.transform(local_energy,local_energy_particle)
        local_energy_particle = self.scale_method.transform_energy(local_energy_particle)
        local_theta = self.scale_method.transform_theta(local_theta)
        local_phi = self.scale_method.transform_phi(local_phi)

        # reshape to original
        local_energy = local_energy.reshape(-1,N_CELLS_R,N_CELLS_PHI,N_CELLS_Z)
        local_energy_particle = local_energy_particle.reshape(-1)
        local_theta = local_theta.reshape(-1)

        return local_energy, local_energy_particle, local_theta, local_phi
