# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 08:18:08 2022

@author: jsl6
"""
import torch
from models import FlipyFlopy
from utils import DicObj, EvalObj
import matplotlib.pyplot as plt
plt.close('all')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = "C:/Users/jsl6/Documents/Python Scripts/Pytorch/SpecPred/prosit/AIData7/"

# Instantiate DicObj
criteria = [line.strip() for line in 
            open(PATH + "input_data/ion_stats/criteria.txt")]
D = DicObj(criteria=criteria)

# Instantiate model
config = {
    line.split("\t")[0]:eval(line.split("\t")[1]) 
    for line in open(PATH+"saved_models/archive/220106/config.tsv","r")
}
model1 = FlipyFlopy(**config, device=device)
model1.load_state_dict(
    torch.load(PATH+'saved_models/archive/220106/ckpt_epoch16_0.9263')
)

# Instantiate EvalObj
mlist = [model1]
E = EvalObj(mlist, D, enable_gpu=True)

labels = open("C:/Users/jsl6/Documents/Paper3/library_labels/large_library/labels_33.txt").read().split("\n")
comments = open("C:/Users/jsl6/Documents/Paper3/library_labels/large_library/msp_comments_33.txt").read().split("\n")

# import os
# files = os.listdir("./temp/")
# for i in range(10):
#     l  = len(files)//10 + 1
#     with open("C:/Users/jsl6/Documents/Python Scripts/Pytorch/SpecPred/prosit/AIData7/predictions/prositplus/insilico_library/islib_24_%d.msp"%i, "w") as f:
#         for file in files[i*l:(i+1)*l]:
#             f.write(open("./temp/"+file,'r').read())