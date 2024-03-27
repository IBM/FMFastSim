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
from importlib import import_module

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class DataInfo(metaclass=Singleton):
    def __init__(self,dataloader_name):

        self._dataloader_name = 'dataloader.'+ dataloader_name
        self._data_module = import_module(self._dataloader_name)

        CONSTANTS = self._data_module.CONSTANTS
        for p in CONSTANTS:
            setattr(self,p,CONSTANTS[p])

    def get_dataset(self,**kwargs):
        return self._data_module.Dataset(**kwargs)
