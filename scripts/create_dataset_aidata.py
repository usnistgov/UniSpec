"""
1. Survey dataset, collect data
files = train
write = False
write_stats = False
mode = True
combo = True
collect_{} = True
2. Write stats
write_stats = True
mode = 'ann in dictionary.keys()'
collect_{} = False
3. Write dataset
write = True
write_stats = False
combo = False
collect_labels = True
"""
import numpy as np 
import sys
import os
import re
import yaml
from time import time

with open('./input_options/create_dataset.yaml','r') as stream:
    config = yaml.safe_load(stream)

with open("./input_options/peptide_criteria.yaml",'r') as stream:
    pep = yaml.safe_load(stream)

curdir = config['curdir']
sys.path.append(config['curdir'])
###############################################################################
############################### Ion dictionary ################################
###############################################################################

with open("./input_options/combo.yaml", 'r') as stream:
    combcon = yaml.safe_load(stream)

# ion types
it = {b:a for a,b in enumerate(combcon['ion_types'])}
# neutrals plus null
neut = {b:a for a,b in enumerate(['']+combcon['neutral_losses'])}
# Fragment lengths
mer = np.arange(1, combcon['max_fragment_length']+1, 1)
# Fragment charges
chars = ['']+['^'+str(i) for i in np.arange(1, combcon['max_fragment_charge']+1, 1)]
# Isotopic peaks
isotopes = ['']+['i' if i==1 else str(i)+'i' 
                 for i in np.arange(1, combcon['max_isotope']+1, 1)
                ]
# Create a dictionary for target indices
if config['combo']:
    f = open(curdir+'input_data/ion_stats/'+"combo_dictionary.txt", "w")
    dictionary={}
    count=0
    for ityp in it:
        for p in mer:
            for ni in neut:
                # Add a hyphen separator for neutral ions
                ni = '+'+ni if ni=='H' else ('-'+ni if ni!='' else ni)
                for char in chars:
                    for isp in isotopes:
                        # add a plus sign is isotopic peak
                        isp = '+%s'%isp if isp!='' else isp
                        
                        """
                        Cut down on permuations by adding limiting criteria
                        here.
                        """
                        # a-ions only have Qian's phospho losses
                        if (ityp=='a') & (ni!=''):
                            if ('PO' not in ni):
                                continue
                        # p-ions have no extent
                        if (ityp=='p') and (int(p)!=1):
                            continue
                        # Immonium ions have no extent, charge, or losses
                        if (
                            ((ityp[0]=='I')&(ityp!='ICCAM')) & 
                            ((int(p)!=1)|(char!='')|(ni!=''))
                        ):
                            continue
                        # ICCAM can only have neutral loss H2O, NH3, or none
                        if (ityp=='ICCAM') & ((ni!='')&(ni!='-H2O')&(ni!='-NH3')):
                            continue
                        # ICCAM cannot have charge or extent
                        if (ityp=='ICCAM') & ((char!='') | (int(p)!=1)):
                            continue
                        # ONLY p-ions can have ^1
                        if (char=='^1') & (ityp!='p'):
                            continue
                        # ONLY p-ions can have RP... or TMT neutral losses
                        if (('RP' in ni)|('TMT' in ni)) & (ityp!='p'):
                            continue
                        # ALL TMT ions have no loss, extent, or charge
                        if ('TMT' in ityp) & ((ni!='')|(p!=1)|(char!='')):
                            continue
                        # TMT1... ions do not have neutral losses, isotopes
                        if ('TMT1' in ityp) & (isp!=''):
                            continue
                            
                        P = (
                            '' if (
                                (ityp=='p') |
                                (ityp[0]=='I') |
                                (ityp=='ICCAM') |
                                ('TMT' in ityp)
                            ) else p
                        )
                        spec = '%s%s%s%s%s'%(ityp,P,ni,char,isp)
                        dictionary[spec] = count
                        f.write("%s %d\n"%(spec, count))
                        count+=1
    f.close()
    # add internals
    if not config['collect_internals']:
        for i,line in enumerate(open(curdir+"input_data/ion_stats/internal_counts.txt","r")):
            if int(line.split()[1])>0: dictionary[line.split()[0]] = len(dictionary)
else:
    """Alternatively I could create a dictionary beforehand and read it in from
    file here. This could be useful if I get rid of all the ion types that are
    absent from the combination of train/val/test sets."""
    from utils import DicObj
    criteria = open(curdir+"input_data/ion_stats/criteria.txt","r").read().split("\n")
    D = DicObj(criteria=criteria, massdir="../input_data/", statsdir='../input_data/ion_stats/')
    dictionary = D.dictionary
revdictionary = {n:m for m,n in dictionary.items()}

###############################################################################
############################# Main part of script #############################
###############################################################################

Files = (
         config['train_files'] if config['files']=='train' else (
         config['val_files'] if config['files']=='val' else (
         config['test_files'] if config['files']=='test' else sys.exit('Choose either train/val/test')
         )))

if config['write']:
    if not os.path.exists(curdir+'input_data/datasets/'): os.makedirs(curdir+'input_data/datasets/')
    g = open(curdir+"input_data/datasets/%s.txt"%config['files'], 'w')
    if not os.path.exists(curdir+'input_data/txt_pos/'): os.makedirs(curdir+'input_data/txt_pos/')
    h = open(curdir+"input_data/txt_pos/fpos%s.txt"%config['files'], 'w')
neutlst=[];modlst=[];intlst=[];immlst=[];labels=[];tmtlst=[]
LENGTHS = [];CHARGES = [];ENERGIES = [];others={}
dic_counter = np.zeros((len(dictionary),3))
ERR_counter = {'int':0}
Startclock = time()
for file in Files:
    print(file.split('/')[-1])
    with open(file,'r') as f:
        startclock = time()
        for line in f:
            if line[:5]=='Name:':
                label = line.split()[1]
                seq,other = label.split("/") # [seq, charge_mods_ev_nce]
                LENGTHS.append(len(seq))
                other_split = other.split('_')
                if len(other_split)==3: 
                    other_split += [0]
                    label += '_NCE0'
                [charge,Mods,ev,nce] = other_split # [charge, mods, ev, nce]
                charge = int(charge);CHARGES.append(charge)
                ce = float(ev[:-2]) # get ev from headline string
                ENERGIES.append(ce)
                
                Mstart = Mods.find('(') if Mods!='0' else 1
                modamt = int(Mods[0:Mstart])
                mods = ([re.sub("[()]",'',m).split(',') for m in Mods[Mstart:].split(')(')]
                        if modamt>0 else []
                )
                typ='';mod_types = []
                seqInt =  list(seq) # seqInt will be written in the Internal style, i.e. lowercase PTMs
                for mod in mods:
                    [pos,aa,typ] = [int(mod[0]),mod[1],mod[2]]
                    seqInt[int(mod[0])] = seqInt[int(mod[0])].lower()
                    mod_types.append(typ)
                    if config['collect_modifications']: modlst.append(typ)
                seqInt = "".join(seqInt)
                # print("\r\033[K%s"%seqInt.strip(), end='')
                
                while True: 
                    line = f.readline() # Skip these lines, which are not in other dataset
                    if line[:3]=='Num': break
                nmpks = int(line.split(":")[1])
                Ints=[];DIC = {}
                for pk in range(nmpks):
                    [mz,ab,Ann] = f.readline().split('\t')
                    Ints.append(float(ab))
                    anns = Ann.split()[0].strip()[1:-1].split(",")[0] # multiple annotations on the line, separated by commas
                    for ann in [anns]: # cycle through each annotation
                        # Convert internal notation to start>extent
                        if ann[:3]=='Int': 
                            ann = "".join(ann.split('/')[:-1]) # Turn Int/{ann} into Int{ann}
                            hold = re.sub("[+-]", ',', ann).split(",") # [ann, neut/iso]
                            if seqInt.find(hold[0][3:])==-1: ERR_counter['int']+=1
                            # issue with internals starting at 0
                            if seqInt.find(hold[0][3:]) == 0:
                                # Find first uppercase match after 1st AA
                                start = seqInt[1:-1].upper().find(hold[0][3:].upper()) + 1
                            else: start = seqInt.find(hold[0][3:])
                            ann = 'Int%d>%d%s'%(start,
                                                len(hold[0][3:]),
                                                ann[len(hold[0]):]
                            )
                        if ann[-3:]=='ppm': ann = ann.split('/')[0] # get regular annotation
                        # DIC is a dictionary, ion->[mz,ab], for the current spectrum
                        if eval(config['mode']):#ann in dictionary.keys():# or ann[:3]=='Int':
                            if config['collect_internals'] and ann[:3]=='Int': 
                                intlst.append(ann)
                            elif config['collect_neutrals']:
                                hold = ann # ion-neutral
                                if '^' in hold:
                                    hold = hold.split('^')[0] # ion-neutral
                                elif 'i' in hold: 
                                    hold = "+".join(hold.split("+")[:-1]) # ion-neutral
                                hold = hold.split('-')
                                if len(hold)>1:
                                    hold2 = "-".join(hold[1:]) # neutral 
                                    neutlst.append(hold2) 
                                    # TMT+H
                                elif 'TMT+H' in ann:
                                    neutlst.append('H')
                            if config['collect_immoniums']:
                                if ann[0]=='I' and ann[:3]!='Int':
                                    immlst.append(ann.split('+')[0])
                            if config['collect_tmt']:
                                if ann[:3]=='TMT':
                                    tmtlst.append(ann.split('+')[0])
                            if config['write'] | config['write_stats']:
                                DIC[ann] = [float(mz), float(ab)]
                        elif config['collect_others']:
                            if ann in others.keys():
                                others[ann] += 1
                            else:
                                others[ann] = 1
                        
                # Write a streamlined dataset
                # - if statements limit the types of peptides I use
                if config['write'] | config['write_stats']:
                    mx = np.max(Ints) # intensities scaled between 0-1
                    modbool = [True if i in pep['modifications'] else False 
                               for i in mod_types]
                    if (
                        (len(seq)>=pep['min_length']) & 
                        (len(seq)<=pep['max_length']) & 
                        (ce>=pep['min_energy']) & 
                        (ce<=pep['max_energy']) & 
                        (charge>=pep['min_charge']) &
                        (charge<=pep['max_charge']) &
                        (False not in modbool)
                    ):
                        labels.append(label.strip())
                        if config['write']: h.write("%d "%g.tell())
                        if config['write']: g.write("NAME: %s|%s|%d|%.1f|%d\n"%(
                                seq,Mods,charge,ce,len(DIC)))
                        for a,b in DIC.items():
                            if config['write']: g.write('%s %d %.4f %.4f\n'%(
                                    a,dictionary[a],b[0],b[1]/mx
                                    ))
                            if config['write_stats']:
                                dic_counter[dictionary[a],0] += 1 # counts
                                dic_counter[dictionary[a],1] += b[1]/mx # sum intensity
        print('\r\033[K%d s'%(time()-startclock))

###############################################################################
############################# Writing to file #################################
###############################################################################
if not os.path.exists(curdir+'input_data/ion_stats/'): os.makedirs(curdir+'input_data/ion_stats')

if config['collect_labels'] and (config['write']|config['write_stats']):
    if not os.path.exists(curdir+'input_data/labels/'): os.makedirs(curdir+'input_data/labels')
    with open(curdir+'input_data/labels/'+"%s_labels.txt"%config['files'],'w') as f:
        f.write("\n".join(labels))
if config['collect_neutrals']:
    A,cntsa = np.unique(neutlst, return_counts=True)
    with open(curdir+'input_data/ion_stats/'+"neutral_counts.txt", 'w') as f:
        for a,b in zip(A,cntsa): f.write('%s %d\n'%(a,b))
if config['collect_modifications']:
    B,cntsb = np.unique(modlst, return_counts=True)
    with open(curdir+'input_data/ion_stats/'+"modification_counts.txt", 'w') as f:
        for a,b in zip(B,cntsb): f.write('%s %d\n'%(a,b))
if config['collect_internals']:
    C,cntsc = np.unique(intlst, return_counts=True)
    with open(curdir+'input_data/ion_stats/'+"internal_counts.txt", 'w') as f:
        for a,b in zip(C,cntsc): f.write('%s %d\n'%(a,b))
if config['collect_immoniums']:
    D,cntsd = np.unique(immlst, return_counts=True)
    with open(curdir+'input_data/ion_stats/'+"immonium_counts.txt", 'w') as f:
        for a,b in zip(D,cntsd): f.write('%s %d\n'%(a,b))
if config['collect_tmt']:
    E,cntse = np.unique(tmtlst, return_counts=True)
    with open(curdir+'input_data/ion_stats/'+"tmt_counts.txt", 'w') as f:
        for a,b in zip(E,cntse): f.write('%s %d\n'%(a,b))
if config['collect_others']:
    F,cntsf = np.unique(others, return_counts=True)
    with open(curdir+'input_data/ion_stats/'+'other_counts.txt', 'w') as f:
        for a,b in others.items(): f.write('%s %d\n'%(a,b))
if config['write_stats']:
    dic_counter[:,-1] = dic_counter[:,1] / np.maximum(dic_counter[:,0],1) # average intensity
    with open(curdir+'input_data/ion_stats/ion_stats_%s.txt'%config['files'],'w') as f:
        for a,b,c in zip(
                dictionary.keys(),
                dic_counter[:,0],
                dic_counter[:,1] / np.maximum(dic_counter[:,0],1)
        ):
            f.write('%22s %8d %.4f\n'%(a,b,c))
if config['write']: g.close()
if config['write']: h.close()
print("\n%d s"%(time()-Startclock))

# import os
# os.system('shutdown /s')
