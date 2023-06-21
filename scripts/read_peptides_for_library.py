# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:13:37 2023

@author: jsl6

Create labels and comments from tsv files that Qian sends me 
"""

filepath = "/path/to/location/of/files/"

fnm_labels = 'HMPhos20230221_labelsNCE16-40.tsv'
fnm_prot = 'uniprotHM20191015_protlabels.tsv'

# with open(filepath + fnm_prot, 'r') as f:
#     # pin: protein index number
#     # name: protein name
#     A = {99999:'Protein=sp|P99999|Multiprotein'}
#     for line in f:
#         [pin, name] = line.split('\t')
#         A[int(pin)] = name.strip()

with open(filepath + fnm_labels, 'r') as f:
    # In: global index number
    # pin: protein index number
    # pm: precursor mass
    # seq: sequence
    # charge: precursor charge
    # mod: modification string
    # nce: NCE
    # ev: eV
    B = {'In':[], 'pin':[], 'pm':[], 'seq':[], 'charge':[], 'mod':[],
         'nce':[], 'ev':[]}
    dts = [int, int, float, str, int, str, float, float]
    for line in f:
        split = line.strip().split('\t')
        assert len(split)==len(B.keys())
        for key,entry,typ in zip(B.keys(), split, dts):
            B[key].append(typ(entry))

# comments = [
#     'Protein=%s'%A[pin] for pin in B['pin']
# ]
labels = [
    '%s/%d_%s_%.1feV_NCE%.0f'%(
        B['seq'][m], B['charge'][m], B['mod'][m], B['ev'][m], B['nce'][m]
    ) for m in range(len(B['seq']))
]

with open("path/to/labels_textfile/HMPhos_labels16.txt",'w') as f:
    f.write("\n".join(labels))
# with open("path/to/comments_textfile/uniprot_comments_40.txt",'w') as f:
#     f.write("\n".join(comments))
    
