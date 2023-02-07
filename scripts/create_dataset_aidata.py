"""
1. Survey dataset, collect sub-stats
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
files                ='test'
write                =True
write_stats          =False
mode                 ='ann in dictionary.keys()' #'ann in dictionary.keys()' or 'True'
combo                =False
collect_neutrals     =False
collect_internals    =False
collect_immoniums    =False
collect_modifications=False
collect_tmt          =False
collect_labels       =True
collect_others       =False
curdir               ="C:/Users/jsl6/Documents/Python Scripts/Pytorch/SpecPred/prosit/AIData8/"
itpath               ="./ion_types.txt"
nlpath               ="./neutral_losses.txt"
lcipath              ='./length_charge_iso.txt'
trfpath              ='./train_files.txt'
valfpath             ='./val_files.txt'
tefpath              ='./test_files.txt'
pepcrit              ='./peptide_criteria.txt'
modpath              ='./modifications.txt'
import numpy as np 
import sys
import re
from time import time
sys.path.append(curdir)
pep = {line.split()[0]:int(line.split()[1]) for line in open(pepcrit, 'r')}
pep['modifications'] = open(modpath,'r').read().split("\n")

###############################################################################
############################### Ion dictionary ################################
###############################################################################

it = {b:a for a,b in enumerate(open(itpath,'r').read().split("\n"))}
neut = {b:a for a,b in enumerate(['']+open(nlpath,'r').read().split("\n"))}
lci = {line.split()[0]:int(line.split()[1]) for line in open(lcipath,'r')}
mer = np.arange(1,lci['max_length'],1)
chars = ['']+['^'+str(i) for i in np.arange(1,lci['max_charge']+1,1)]
isotopes = ['']+['i' if i==1 else str(i)+'i' 
                 for i in np.arange(1,lci['max_isotope']+1,1)
                ]
# Create a dictionary for target indices
if combo:
    f = open("combo_dictionary.txt", "w")
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
                        # a-ions have no losses
                        if (ityp=='a') & (ni!=''):
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
    if not collect_internals:
        for i,line in enumerate(open(curdir+"input_data/ion_stats/internal_counts.txt","r")):
            if int(line.split()[1])>0: dictionary[line.split()[0]] = len(dictionary)
else:
    """Alternatively I could create a dictionary beforehand and read it in from
    file here. This could be useful if I get rid of all the ion types that are
    absent from the combination of train/val/test sets."""
    from utils import DicObj
    criteria = open(curdir+"input_data/ion_stats/criteria.txt","r").read().split("\n")
    D = DicObj(criteria=criteria)
    dictionary = D.dictionary
revdictionary = {n:m for m,n in dictionary.items()}

###############################################################################
############################# Main part of script #############################
###############################################################################

train_files = open(trfpath,'r').read().split("\n")
test_files = open(tefpath,'r').read().split("\n")
val_files = open(valfpath,'r').read().split("\n")
all_files = test_files+val_files+train_files
Files = (
         train_files if files=='train' else (
         val_files if files=='val' else (
         test_files if files=='test' else sys.exit('Choose either train/val/test')
         )))

if write:
    g = open(curdir+"dataset.txt", 'w')
    h = open(curdir+"fpos.txt", 'w')
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
                    if collect_modifications: modlst.append(typ)
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
                        if eval(mode):#ann in dictionary.keys():# or ann[:3]=='Int':
                            if collect_internals and ann[:3]=='Int': 
                                intlst.append(ann)
                            elif collect_neutrals:
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
                            if collect_immoniums:
                                if ann[0]=='I' and ann[:3]!='Int':
                                    immlst.append(ann.split('+')[0])
                            if collect_tmt:
                                if ann[:3]=='TMT':
                                    tmtlst.append(ann.split('+')[0])
                            if write | write_stats:
                                DIC[ann] = [float(mz), float(ab)]
                        elif collect_others:
                            if ann in others.keys():
                                others[ann] += 1
                            else:
                                others[ann] = 1
                        
                # Write a streamlined dataset
                # - if statements limit the types of peptides I use
                if write | write_stats:
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
                        if write: h.write("%d "%g.tell())
                        if write: g.write("NAME: %s|%s|%d|%.1f|%d\n"%(
                                seq,Mods,charge,ce,len(DIC)))
                        for a,b in DIC.items():
                            if write: g.write('%s %d %.4f %.4f\n'%(
                                    a,dictionary[a],b[0],b[1]/mx
                                    ))
                            if write_stats:
                                dic_counter[dictionary[a],0] += 1 # counts
                                dic_counter[dictionary[a],1] += b[1]/mx # sum intensity
        print('\r\033[K%d s'%(time()-startclock))

###############################################################################
############################# Writing to file #################################
###############################################################################

if collect_labels and (write|write_stats):
    with open("labels.txt",'w') as f:
        f.write("\n".join(labels))
if collect_neutrals:
    A,cntsa = np.unique(neutlst, return_counts=True)
    with open("neutral_counts.txt", 'w') as f:
        for a,b in zip(A,cntsa): f.write('%15s %7d\n'%(a,b))
if collect_modifications:
    B,cntsb = np.unique(modlst, return_counts=True)
    with open("modification_counts.txt", 'w') as f:
        for a,b in zip(B,cntsb): f.write('%15s %7d\n'%(a,b))
if collect_internals:
    C,cntsc = np.unique(intlst, return_counts=True)
    with open("internal_counts.txt", 'w') as f:
        for a,b in zip(C,cntsc): f.write('%15s %7d\n'%(a,b))
if collect_immoniums:
    D,cntsd = np.unique(immlst, return_counts=True)
    with open("immonium_counts.txt", 'w') as f:
        for a,b in zip(D,cntsd): f.write('%15s %7d\n'%(a,b))
if collect_tmt:
    E,cntse = np.unique(tmtlst, return_counts=True)
    with open("tmt_counts.txt", 'w') as f:
        for a,b in zip(E,cntse): f.write('%15s %7d\n'%(a,b))
if write_stats:
    dic_counter[:,-1] = dic_counter[:,1] / np.maximum(dic_counter[:,0],1) # average intensity
    with open(curdir+'input_data/ion_stats/ion_stats.txt','w') as f:
        for a,b,c in zip(
                dictionary.keys(),
                dic_counter[:,0],
                dic_counter[:,1] / np.maximum(dic_counter[:,0],1)
        ):
            f.write('%22s %8d %.4f\n'%(a,b,c))
if write: g.close()
if write: h.close()
print("\n%d s"%(time()-Startclock))

# import os
# os.system('shutdown /s')