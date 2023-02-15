import torch
from models import FlipyFlopy
from utils import DicObj, EvalObj
import yaml
import os
import matplotlib.pyplot as plt
plt.close('all')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("./input_data/configuration/predict.yaml", 'r') as stream:
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

if config['mode']=='write_msp':
    
    cecorr = config['write_msp']['cecorr']
    outfn = config['write_msp']['outfn']
    
    if config['write_msp']['new']==True:
        
        assert os.path.exists(config['write_msp']['label_path']), 'label path does not exist'
        labels = open(config['write_msp']['label_path']).read().split("\n")
        
        if config['write_msp']['comments_path']!='None':
            assert os.path.exists(config['write_msp']['comments_path']), 'comments path does not exist'
            comments = open(config['write_msp']['comments_path']).read().split('\n')
        else:
            comments = None
        
        print_every = config['write_msp']['print_every']
        print_every = eval(print_every) if type(print_every)==str else print_every
        
        E.write_msp(
            labels, cecorr, comments=comments, print_every=print_every, outfn=outfn
        )
    
    else:
        
        inp = config['write_msp']['dset']
        E.write_msp(inp, cecorr=cecorr, outfn=outfn)
        