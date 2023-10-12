'''
Author: Radillus
Date: 2023-05-31 16:44:10
LastEditors: Radillus
LastEditTime: 2023-06-01 21:15:41
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import logging
import copy

import ray
import torch
import torch.nn as nn

import designer.utils as utils


@ray.remote
class RemoteNetwork:
    def __init__(
        self,
        model: nn.Module,
    ):
        self.model = model.cpu()
        self.version = 0
    def get_version(self):
        return self.version
    def update_model(self, model: nn.Module):
        self.model = model.cpu()
        self.version += 1
    def get_model(self):
        return self.model

class LocalNetwork(nn.Module):
    def __init__(
        self,
        remote_model_ref: RemoteNetwork,
    ):
        super().__init__()
        self.model: nn.Module
        self.remote_model_ref: RemoteNetwork
        self.remote_model_ref = remote_model_ref
        self.model = ray.get(remote_model_ref.get_model.remote())
        self.version = ray.get(remote_model_ref.get_version.remote())
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    def upload_to_remote(self):
        remote_version = ray.get(self.remote_model_ref.get_version.remote())
        if remote_version > self.version:
            logging.warning(utils.logging_time_stamp() + f" Upload model conflict! Local version: {self.version}, remote version: {remote_version}")
            self.download_from_remote()
        else:
            self.remote_model_ref.update_model.remote(copy.deepcopy(self.model).cpu())
            self.version += 1
    def download_from_remote(self):
        remote_version = ray.get(self.remote_model_ref.get_version.remote())
        if remote_version == self.version:
            pass
        else:
            self.device = torch.get_device(self.model)
            self.version = ray.get(self.remote_model_ref.get_version.remote())
            self.model = ray.get(self.remote_model_ref.get_model.remote()).to(self.device)