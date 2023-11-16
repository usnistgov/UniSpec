import torch
from models import FlipyFlopy
from utils import DicObj, EvalObj
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open("./input_data/configuration/predict.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Instantiate DicObj
with open(config['dic_config'], 'r') as stream:
    dconfig = yaml.safe_load(stream)

# Make some switches (if nec.) between config and dconfig
dconfig['criteria_path'] = (config['criteria'] 
                         if config['criteria'] is not None else 
                         dconfig['criteria_path']
)
dconfig['mod_path'] = (config['mod_path'] 
                         if config['mod_path'] is not None else 
                         dconfig['mod_path']
)
dconfig['stats_path'] = (config['stats_txt'] 
                         if config['stats_txt'] is not None else 
                         dconfig['stats_path']
)
D = DicObj(**dconfig)

# Instantiate model
if config['model_config'] is not None:
    with open(config['model_config']) as stream:
        model_config = yaml.safe_load(stream)
    model1 = FlipyFlopy(**model_config, device=device)
    model1.load_state_dict(torch.load(config['model_ckpt']))
    mlist = [model1]
else:
    mlist = []

# Instantiate EvalObj
E = EvalObj(config, mlist, D, enable_gpu=config['enable_gpu'])

if config['mode']=='write_msp':
    
    cecorr = config['write_msp']['cecorr']
    outfn = config['write_msp']['outfn']
    
    # Leave 'label_type' blank if you want to write the msp in 'dset'
    if config['write_msp']['label_type'] is not None:
        
        # Make sure label path exists
        assert os.path.exists(config['write_msp']['label_path']), ( 
            'label path does not exist'
        )
        from utils import Labeler
        labeler = Labeler(D)
        # Two ways we can do this
        #  1. label_type: noev. Input file has space separated format: 
        #     seq charge mod_string nce  
        if config['write_msp']['label_type'].lower() == 'noev':
            labels = labeler.IncompleteLabels(config['write_msp']['label_path'])
        # 2. label_type: complete. Input file has newline separated list of 
        #    complete labels: {seq}/{charge}_{mods}_{ev}eV_NCE{nce}
        elif config['write_msp']['label_type'].lower() == 'complete':
            labels = labeler.CompleteLabels(config['write_msp']['label_path'])
        else:
            RuntimeError("Choose for label_type 'noev' or 'complete'")
        
        # Make sure comments path exists
        if config['write_msp']['comments_path'] != None:
            assert os.path.exists(config['write_msp']['comments_path']), (
                'comments path does not exist')
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
elif config['mode']=='calc_cs':
    # Load in Label_pred - Label_raw map, if appplicable
    if config['calc_cs']['map_path'] is not None:
        Map = np.loadtxt(config['calc_cs']['map_path'], dtype='str')
        assert Map.shape[1]==2, "Map doesn't have 2 columns"
    else:
        Map = None
    
    cskwargs = (
       {} 
       if config['calc_cs']['CSkwargs']==None else 
       config['calc_cs']['CSkwargs']
    )
        
    _ = E.CalcCSdset(
        config['calc_cs']['predset'], 
        config['calc_cs']['rawset'], 
        Map = Map,
        closest_match=config['calc_cs']['closest_match'],
        out_fn = config['calc_cs']['outfn'],
        **cskwargs
    )
