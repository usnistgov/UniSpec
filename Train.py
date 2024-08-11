import numpy as np
import sys
import os
import yaml
from time import time
import torch
from models import FlipyFlopy
import matplotlib.pyplot as plt
plt.close('all')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
When using multiple workers, windows uses spawn rather than fork, which means the
main script will be run over for each worker process, leading to memory issues.

https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection

Must factor code in the following way:

def main():
    run training
if __name__ == "__main__":
    main()
"""
def main():

    with open("./input_data/configuration/Train.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    ###############################################################################
    ############################## Dictionaries ###################################
    ###############################################################################

    with open("./input_data/configuration/dic.yaml", 'r') as stream:
        dconfig = yaml.safe_load(stream)

    from utils import DicObj
    D = DicObj(**dconfig)

    # Configuration dictionary
    if config['config'] is not None:
        # Load model config
        with open(config['config'], 'r') as stream:
            model_config = yaml.safe_load(stream)
    else:
        channels = D.seq_channels if config['model_config']['CEembed'] else D.channels
        model_config = {
            'in_ch': channels,
            'seq_len': D.seq_len,
            'out_dim': len(D.dictionary),
            **config['model_config']
        }
    
    # Dataset dictionary
    with open('./input_data/configuration/dataset.yaml', 'r') as stream:
        dset_config = yaml.safe_load(stream)

    ###############################################################################
    ################################ Dataset ######################################
    ###############################################################################

    from utils import LoadObj
    L = LoadObj(
        dobj=D, 
        embed=False,#model_config['CEembed'],
        **dset_config,
    )

    # Labels
    label_dict = {
        'train': np.array([
            line.strip() for line in open(config['train']['labels'],'r')
        ]),
        'val': np.array([
            line.strip() for line in open(config['val']['labels'],'r')
        ]),
        'test': np.array([
            line.strip() for line in open(config['test']['labels'],'r')
        ]),
    }

    # find long sequence for mirrorplot
    Lens = []
    #for pos in fposte:
    #    test_point.seek(pos) 
    #    Lens.append(len(test_point.readline().split()[1].split('|')[0]))
    MPIND = 1000

    ###############################################################################
    ################################## Model ######################################
    ###############################################################################

    # Instantiate model
    model = FlipyFlopy(**model_config, device=device)
    arrdims = len(model(L.input_from_str(label_dict['train'][0:1])[0], test=True)[1][0])
    model.to(device)

    # Load weights
    if config['weights'] is not None:
        model.load_state_dict(torch.load(config['weights']))

    # TRANSFER LEARNING
    if config['transfer'] is not None:
        model.final = torch.nn.Sequential(torch.nn.Linear(512,D.dicsz), torch.nn.Sigmoid())
        for parm in model.parameters(): parm.requires_grad=False
        for parm in model.final.parameters(): parm.requires_grad=True

    sys.stdout.write("Total model parameters: ")
    model.total_params()

    # Optimizer
    opt = torch.optim.Adam(model.parameters(), eval(config['lr']))
    if config['restart'] is not None:
        # loading optimizer state requires it to be initialized with model GPU parms
        opt.load_state_dict(torch.load(config['restart'], map_location=device))

    ###############################################################################
    ############################# Reproducability #################################
    ###############################################################################
    
    if not os.path.exists('./saved_models'): os.makedirs('./saved_models/')
    with open("saved_models/model_config.yaml","w") as file:
        yaml.dump(model_config, file)
    with open("saved_models/dic.yaml","w") as file:
        yaml.dump(dconfig, file)
    with open("saved_models/criteria.txt", 'w') as file:
        file.write(open(dconfig['criteria_path']).read())
    with open("saved_models/modifications.txt", 'w') as file:
        file.write(open(dconfig['mod_path']).read())
    with open("saved_models/ion_stats_train.txt", 'w') as file:
        file.write(open(dconfig['stats_path']).read())
    
    ###########################################################################
    ############################## Training ###################################
    ###########################################################################
    
    from utils import Trainer
    
    trainer = Trainer(
        L,
        model,
        opt,
        config,
        device,
        label_dict,
    )
    
    trainer.train(
        config['epochs'], 
        batch_size=config['batch_size'], 
        lr_decay_start=config['lr_decay_start'], 
        lr_decay_rate=config['lr_decay_rate'], 
        svwts=config['svwts'],
    )

if __name__ == "__main__":
    __spec__ = None # workaround for ipython, ipdb (for windows only?)
    main()