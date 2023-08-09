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

label = 'MQNTLYLSLTR/2_0_31.6eV_NCE25'

import numpy as np
# by=True
# dset='ftms35'
# dsetps='%sps'%dset
psions = []
i = 1
for ion in ['b', 'y']:
    for ext in np.arange(1,30,1):
        for ch in ['', '^2', '^3']:
            psions.append('%s%s%s'%(ion, str(ext), ch))
            i+=1

rawspec = E._inp_spec(E.dsets['ftms25']['lab'][label]['ind'], 'ftms25', 'raw')[-1]
boo = [ion.split('/')[0] in psions for ion in rawspec[-1]]
rawspec = [spec[boo] for spec in rawspec]
raw_peaks = E.lst2spec(rawspec[0], rawspec[1])
pred = E._inp_spec(E.dsetspred['ftms25']['lab'][label]['ind'], 'ftms25', 'pred')[-1]
boo = [ion.split('/')[0] in psions for ion in pred[-1]]
pred = [spec[boo] for spec in pred]
pred_peaks = E.lst2spec(pred[0], pred[1])
cs, diffs = E.CosineScore(
    pred_peaks, raw_peaks, pions=pred[-1], rions=rawspec[-1]
)

