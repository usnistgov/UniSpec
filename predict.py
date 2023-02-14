# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 08:18:08 2022

@author: jsl6
"""
import torch
from models import FlipyFlopy
from utils import DicObj, EvalObj
import yaml
import matplotlib.pyplot as plt
plt.close('all')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("./input_data/configuration/predict.config", 'r') as stream:
    config = yaml.safe_load(stream)

# Instantiate DicObj
with open("./input_data/configuration/dic.yaml", 'r') as stream:
    dconfig = yaml.safe_load(stream)
D = DicObj(**dconfig)

# Instantiate model
with open("saved_models/model_config.yaml") as stream:
    model_config = yaml.safe_load(stream)
model1 = FlipyFlopy(**model_config, device=device)
model1.load_state_dict(
    torch.load(config['model_ckpt'])
)

# Instantiate EvalObj
mlist = [model1]
E = EvalObj(config, mlist, D, enable_gpu=config['enable_gpu'])
