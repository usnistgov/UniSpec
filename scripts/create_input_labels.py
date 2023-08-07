# -*- coding: utf-8 -*-
"""
Create the input labels from user provided input containing:
    1. sequence
    2. charge
    3. entire modification string
    4. nce

- Allowed modifications (230807):
    - Acetyl, Carbamidomethyl, Gln->pyro-Glu, Glu->pyro-Glu, Oxidation, 
      Phospho, Pyro-carbamidomethyl
- Allowed instrument types (230807):
    - q_exactive, q_exactive_hfx, elite, velos, lumos
"""

import sys
import yaml

# Read in configuration yaml file
with open("input_options/create_input_labels.yaml") as stream:
    config = yaml.safe_load(stream)
sys.path.append(config['usdir'])

# Need some utilities for ce conversion and mass calculation
from utils import NCE2eV, DicObj
D = DicObj()

# Read in space separated label information
with open(config['txtpath'], 'r') as f:
    labs = [a.split() for a in f.read().split("\n")]

# Calculate m/z, eV, and write out the labels
labels = []
for data in labs:
    seq = data[0]
    charge = int(data[1])
    mods = data[2]
    nce = int(data[3])
    
    mz = D.calcmass(seq, charge, mods, 'p') / charge
    ev = NCE2eV(nce, mz, charge)
    
    label = '%s/%d_%s_%.1feV_NCE%d'%(seq,charge,mods,ev,nce)
    labels.append(label)

with open(config['usdir']+config['outpath'], 'w') as f:
    f.write("\n".join(labels))

