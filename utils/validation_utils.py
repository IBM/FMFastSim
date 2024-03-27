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
import argparse
import numpy as np

import glob

from utils.observables import LongitudinalProfile, LateralProfile, Energy, PhiProfile
from utils.plotters import ProfilePlotter, EnergyPlotter

def compare_profiles(e_layer_g4, e_layer_model, particle_energy, particle_angle, geometry, valid_dir, keep_previous=True):

    #Check directory
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    #delete previous images
    if not keep_previous:
        files = glob.glob(valid_dir+'/*.png')
        for f in files:
            os.remove(f)

    #1.Longitudinal Profile
    full_sim_long = LongitudinalProfile(_input=e_layer_g4)
    ml_sim_long   = LongitudinalProfile(_input=e_layer_model)
    longitudinal_profile_plotter = ProfilePlotter(valid_dir, particle_energy, particle_angle, 
                                                  geometry, full_sim_long, ml_sim_long,_plot_gaussian=False)
    longitudinal_profile_plotter.plot_and_save()

    #2.Lateral Profile
    full_sim_lat = LateralProfile(_input=e_layer_g4)
    ml_sim_lat = LateralProfile(_input=e_layer_model)
    lateral_profile_plotter = ProfilePlotter(valid_dir, particle_energy, particle_angle,
                                             geometry, full_sim_lat, ml_sim_lat, _plot_gaussian=False)
    lateral_profile_plotter.plot_and_save()

    #3.Azimuthal Profile
    full_sim_phi = PhiProfile(_input=e_layer_g4)
    ml_sim_phi = PhiProfile(_input=e_layer_model)
    phi_profile_plotter = ProfilePlotter(valid_dir, particle_energy, particle_angle,
                                         geometry, full_sim_phi, ml_sim_phi, _plot_gaussian=False)
    phi_profile_plotter.plot_and_save()

    #4.Total energy 
    full_sim_energy = Energy(_input=e_layer_g4)
    ml_sim_energy = Energy(_input=e_layer_model)

    energy_plotter = EnergyPlotter(valid_dir, particle_energy, particle_angle, 
                                   geometry, full_sim_energy, ml_sim_energy)
    energy_plotter.plot_and_save()

    print(flush=True)
