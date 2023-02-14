# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 08:18:11 2022

@author: jsl6
"""
import re
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

class DicObj:
    def __init__(self,
                 seq_len = 40,
                 chlim = [1,8],
                 criteria_path='criteria.txt',
                 mass_path="masses.txt",
                 stats_path='ion_stats_train.txt',
                 mod_path='modifications.txt'
                 ):
        self.seq_len = seq_len
        self.chlim = chlim
        self.chrng = chlim[-1]-chlim[0]+1
        self.dic = {b:a for a,b in enumerate('ARNDCQEGHILKMFPSTWYVX')}
        self.revdic = {b:a for a,b in self.dic.items()}
        if os.path.exists(mod_path):
            self.mdic = {
                b:a+len(self.dic) 
                for a,b in enumerate(['']+open(mod_path).read().split("\n"))
            }
        else:
            self.mdic = {b:a+len(self.dic) for a,b in enumerate([
                '','Acetyl', 'Carbamidomethyl', 'Gln->pyro-Glu', 'Glu->pyro-Glu', 
                'Oxidation', 'Phospho', 'Pyro-carbamidomethyl', 'TMT6plex'])
            }
        self.revmdic = {b:a for a,b in self.mdic.items()}
        
        if os.path.exists(mass_path):
            self.mass = {line.split()[0]:float(line.split()[1]) for line in 
                          open(mass_path,'r')}
        else:
            # proteomicsresource.washington.edu/protocols06/masses.php
            # www.unimod.org/login.php?a=logout
            self.mass = {
                # Amino acids
                'A': 71.037113805,'R':156.101111050,'N':114.042927470,'D':115.026943065,
                'C':103.009184505,'Q':128.058577540,'E':129.042593135,'G': 57.021463735,
                'H':137.058911875,'I':113.084064015,'L':113.084064015,'K':128.094963050,
                'M':131.040484645,'F':147.068413945,'P': 97.052763875,'S': 87.032028435,
                'T':101.047678505,'W':186.079312980,'Y':163.063328575,'V': 99.068413945,
                # Neutral losses
                'NH2':16.0187,'NH3':17.0265+4.9e-5,'H2O':18.010565,'CO':27.994915,
                'C2H5NOS':91.009184,'CH2SH':46.9955+0.0458,'CH3SOH':63.99828544,
                'HPO3':79.966335,'H3PO4':97.9769,'H5PO5':115.987465,'H7PO6':133.99803,
                'TMT':229.17,'RP126':154.1221,'RP127N':155.1192,'RP127C':155.1254,
                'RP128N':156.1225,'RP128C':156.1287,'RP129N':157.1258,'RP129C':157.1322,
                'RP130N':158.1291,'RP130C':158.1356,'RP131':159.1325,
                # Modifications
                'Acetyl':42.010565,'Carbamidomethyl':57.021464,'Oxidation':15.994915,
                'Gln->pyro-Glu':-17.026549, 'Glu->pyro-Glu':-18.010565,'Phospho':79.966331,
                'Pyro-carbamidomethyl':39.994915,'CAM':57.021464,'TMT6plex':231.17747,
                # Isotopes
                'i':1.00727646688,'iso1':1.003,'iso2':1.002,
                # Ions, immoniums, et al.
                'a':-26.9871,'b':1.007276, 'p':20.02656, 'y':19.0184,
                'ICA':76.021545, 'IDA':88.039304, 'IDB':70.028793, 'IEA':102.054954,
                'IFA':120.080775,'IFB':91.054226, 'IHA':110.071273,'IHB':82.05255,
                'IHC':121.039639,'IHD':123.055289,'IHE':138.066188,'IHF':156.076753,
                'IIA':86.096425, 'IIC':72.080776, 'IKA':101.107324,'IKB':112.07569,
                'IKC':84.080776, 'IKD':129.102239,'IKE':56.049476, 'IKF':175.118952,
                'ILA':86.096426, 'ILC':72.080776, 'IMA':104.052846,'IMB':61.010647,
                'IMC':120.047761,'INA':87.055289, 'INB':70.02874,  'IPA':70.065126, 
                'IQA':101.070939,'IQB':56.049476, 'IQC':84.04439,  'IQD':129.065854,
                'IRA':129.113473,'IRB':59.060375, 'IRC':70.065126, 'IRD':73.076025, 
                'IRE':87.091675, 'IRF':100.086924,'IRG':112.086924,'IRH':60.055624,
                'IRI':116.070605,'IRJ':175.118952,'ISA':60.04439,  'ITA':74.06004, 
                'ITB':119.0814,  'IVA':72.080776, 'IVC':55.054227, 'IVD':69.033491,
                'IWA':159.091675,'IWB':77.038577, 'IWC':117.057301,'IWD':130.065126,
                'IWE':132.080776,'IWF':170.06004, 'IWH':142.065126,'IYA':136.07569,
                'IYB':91.054227, 'IYC':107.049141,
                'ICCAM':133.04301,
                'TMTpH':230.17,'TMT126':126.1277,'TMT127N':127.1248,'TMT127C':127.1311,
                'TMT128N':128.1281,'TMT128C':128.1344,'TMT129N':129.1315,
                'TMT129C':129.1378,'TMT130N':130.1348,'TMT130C':130.1411,'TMT131':131.1382
            }
        
        self.dictionary = {}
        criteria = (
            open(criteria_path,"r").read().split("\n") 
            if os.path.exists(criteria_path) else ['occurs>0']
        )
        self.make_dictionary(criteria, fn=stats_path)
        
        self.seq_channels = len(self.dic) + len(self.mdic)
        self.channels = len(self.dic) + len(self.mdic) + self.chrng + 1
        
        # Synonyms
        if 'Carbamidomethyl' in self.mdic.keys():
            self.mdic['CAM'] = self.mdic['Carbamidomethyl']
            self.revmdic[self.mdic['CAM']] = 'CAM'
        elif 'CAM' in self.mdic.keys():
            self.mdic['Carbamidomethyl'] = self.mdic['CAM']
            self.revmdic[self.mdic['Carbamidomethyl']] = 'Carbamidomethyl'
        if 'TMT6plex' in self.mdic.keys():
            self.mdic['TMT'] = self.mdic['TMT6plex']
            self.revmdic[self.mdic['TMT']] = 'TMT'
        elif 'TMT' in self.mdic.keys():
            self.mdic['TMT6plex'] = self.mdic['TMT']
            self.revmdic[self.mdic['TMT6plex']] = self.mdic['TMT6plex']
        
    def make_dictionary(self, 
                        criteria=['occurs>0'], 
                        fn="input_data/ion_stats/ion_stats_train.txt"):
        """
        Create an ion dictionary and store it in member variables 
        self.dictionary, self.revdictionary. Needs to read an ion statistics
        that contains 1) ion name, 2) ion occurences, 3) ion mean intensity.
        
        :param criteria: list of strings, each a python logical statement that
                         must be met in order to add ion to dictionary
        :param fn: path to ion statistics file
        
        """
        
        count=0
        for line in open(fn,'r'):
            # Make sure this agrees with pattern of input file
            [ion,occurs,meanint] = line.split()
            # convert entries
            occurs=int(occurs);meanint=float(meanint)
            if eval("(" + ")&(".join(criteria) + ")"):
                self.dictionary[ion] = count
                count+=1
        self.revdictionary = {b:a for a,b in self.dictionary.items()}
        self.dicsz = len(self.dictionary)
    
    def create_filter(self, criteria):
        """
        Create boolean filter vector based on criteria

        Parameters
        ----------
        criteria : string of python code to be evaluated

        Returns
        -------
        Boolean array where True are ions included and False are excluded

        """
        return [i for i,ion in enumerate(list(self.dictionary.keys())) 
                if eval(criteria)]
        
    def calcmass(self, seq, pcharge, mods, ion):
        """
        Calculating the mass of fragments

        Parameters
        ----------
        seq : Peptide sequence (str)
        pcharge : Precursor charge (int)
        mods : Modification string (str)
        ion : Ion type (str)

        Returns
        -------
        mass as a float

        """
        # modification
        Mstart = mods.find('(') if mods!='0' else 1 # EDIT 230121
        modamt = int(mods[0:Mstart])
        modlst = []
        if modamt>0:
            Mods = [re.sub("[()]",'',m).split(',') for m in 
                     mods[Mstart:].split(')(')]
            for mod in Mods:
                [pos,aa,typ] = mod # mod position, amino acid, and type
                modlst.append([int(pos), self.mass[typ]])
        
        # isotope
        isomass = self.mass['iso1'] if '+i' in ion else self.mass['iso2']
        if ion[-1]=='i': # evaluate isotope and set variable iso
            hold = ion.split("+")
            iso = 1 if hold[-1]=='i' else int(hold[-1][:-1])
            ion = "+".join(hold[:-1]) # join back if +CO was split
        else:
            iso = 0
        
        # If internal calculate here and return
        if ion[:3]=='Int':
            ion = ion.split('+')[0] if iso!=0 else ion
            ion = ion.split('-')
            nl = self.mass[ion[1]] if len(ion)>1 else 0
            [start,extent] = [int(z) for z in ion[0][3:].split('>')]
            modmass = sum([ms[1] for ms in modlst 
                           if ((ms[0]>=start)&(ms[0]<(start+extent)))
                          ])
            return (sum([self.mass[aa] for aa in seq[start:start+extent]]) - nl
                    + iso*isomass + self.mass['i'] + modmass)
        # if TMT, calculate here and return
        if ion[:3]=='TMT':
            ion = ion.split('+')[0] if iso!=0 else ion
            return self.mass[ion] + iso*isomass
        
        # product charge
        hold = ion.split("^") # separate off the charge at the end, if at all
        charge = 1 if len(hold)==1 else int(hold[-1]) # no ^ means charge 1
        # extent
        letnum = hold[0].split('-')[0];let = letnum[0] # ion type and extent is always first string separated by -
        num = int(letnum[1:]) if ((let!='p')&(let!='I')) else 0 # p type ions never have number
        
        # neutral loss
        nl=0
        hold = hold[0].split('-')[1:] # most are minus, separated by -
        """If NH2-CO-CH2SH, make the switch to C2H5NOS. Get rid of CO and CH2SH.""" 
        if len(hold)>0 and ('NH2' in hold[0]):
            mult = (int(hold[0][0]) 
                    if ((ord(hold[0][0])>=48) & (ord(hold[0][0])<=57)) else '')
            hold[0] = str(mult)+'C2H5NOS';del(hold[1],hold[1])
        for item in hold:
            if '+' in item: # only CO can be a +
                items = item.split('+') # split it e.g. H2O+CO
                nl-=self.mass[items[0]] # always minus the first
                nl+=self.mass[items[1]] # always plus the second
            else:
                if (ord(item[0])>=48) & (ord(item[0])<=57): # if there are e.g. 2 waters -> 2H2O 
                    mult = int(item[0])
                    item = item[1:]
                else:
                    mult = 1
                nl-=mult*self.mass[item]

        if let=='a':
            sm = sum([self.mass[aa] for aa in seq[:num]])
            modmass = sum([mod[1] for mod in modlst if num>mod[0]]) # if modification is before extent from n terminus
            return self.mass['i'] + (sm + modmass - self.mass['CO'] + nl + iso*isomass) / charge
        elif let=='b':
            sm = sum([self.mass[aa] for aa in seq[:num]])
            modmass = sum([mod[1] for mod in modlst if num>mod[0]]) # if modification is before extent from n terminus
            return self.mass['i'] + (sm + modmass + nl + iso*isomass) / charge
        elif let=='p':
            # p eq.: proton + (aa+H2O+mods+i)/charge
            sm = sum([self.mass[aa] for aa in seq])
            charge = int(pcharge) if ((charge==1)&('^1' not in ion)) else charge
            modmass = sum([mod[1] for mod in modlst]) # add all modifications
            return self.mass['i'] + (sm + self.mass['H2O'] + modmass + nl + iso*isomass) / charge
        elif let=='y':
            sm = sum([self.mass[aa] for aa in seq[-num:]])
            modmass = sum([mod[1] for mod in modlst if ((len(seq)-num)<=mod[0])]) # if modification is before extent from c terminus
            return self.mass['i'] + (sm + self.mass['H2O'] + modmass + nl + iso*isomass)/charge #0.9936 1.002
        elif let=='I':
            sm = self.mass[letnum]
            return (sm + iso*isomass) / charge
        else:
            return False

class LoadObj:
    def __init__(self, dobj, embed=False):
        self.D = dobj
        self.embed = embed
        self.channels = dobj.seq_channels if embed else dobj.channels
    
    def str2dat(self, string):
        """
        Turn a label string into its constituent 

        Parameters
        ----------
        string : label string in form {seq}/{charge}_{mods}_{ev}eV_NCE{nce}

        Returns
        -------
        Tuple of seq,mods,charge,ev,nce

        """
        seq,other = string.split('/')
        [charge,mods,ev,nce] = other.split('_')
        # Mstart = mods.find('(') if mods!='0' else 1 # EDIT 230121
        # modnum = int(mods[0:Mstart])
        # if modnum>0:
        #     modlst = [re.sub('[()]','',m).split(',') 
        #               for m in mods[Mstart:].split(')(')]
        #     modlst = [(int(m[0]),m[-1]) for m in modlst]
        # else: modlst = []
        return (seq,mods,int(charge),float(ev[:-2]),float(nce[3:]))
    
    def inptsr(self, info):
        """
        Create input(s) for 1 peptide

        Parameters
        ----------
        info : tuple of (seq,mod,charge,ev,nce)

        Returns
        -------
        out : List of a) tensor to model, b) charge float and/or c) ce float.
              Only outputs a) if not self.embed

        """
        (seq,mod,charge,ev,nce) = info
        output = torch.zeros((self.channels, self.D.seq_len), dtype=torch.float32)
        
        # Sequence
        output[:len(self.D.dic),:len(seq)] = torch.nn.functional.one_hot(
            torch.tensor([self.D.dic[o] for o in seq], dtype=torch.long),
            len(self.D.dic)
        ).T
        output[len(self.D.dic)-1, len(seq):] = 1.
        # PTMs
        Mstart = mod.find('(') if mod!='0' else 1 #EDIT 230118
        modamt = int(mod[0:Mstart]) #EDIT 230118
        output[len(self.D.dic)] = 1.
        if modamt>0:
            hold = [re.sub('[()]', '', n) for n in mod[Mstart:].split(")(")] #EDIT 230118
            for n in hold:
                [pos,aa,modtyp] = n.split(',')
                output[self.D.mdic[modtyp], int(pos)] = 1.
                output[len(self.D.dic), int(pos)] = 0.
        
        if self.embed:
            out = [output, float(charge), float(ev)]
        if not self.embed:
            output[self.D.seq_channels+int(charge)-1] = 1. # charge
            output[-1, :] = float(ev)/100. # ce
            out = [output]
        return out
    
    def input_from_file(self, fstarts, fn):
        """
        Create batch of model inputs from array of file starting positions and
        the filename.
        - If self.embed=True, then this function outputs charge and energy as 
        batch-size length arrays containing the their respective values for 
        each input. Otherwise the first output is just a single embedding tensor.
        
        :param fstarts: array of file postions for spectral labels to be loaded.
        :param fn: filename to be opened
        
        :return out: List of inputs to the model. Length is 3 if
                     self.embed=True, else length is 1.
        :return info: List of tuples of peptide data. Each tuple is ordered as
                      (seq,mod,charge,ev,nce).
        """
        if type(fstarts)==int: fstarts = [fstarts]

        bs = len(fstarts)
        outseq = torch.zeros((bs, self.channels, self.D.seq_len),
                             dtype=torch.float32)
        if self.embed:
            outch = torch.zeros((bs,), dtype=torch.float32)
            outce = torch.zeros((bs,), dtype=torch.float32)

        info = []
        with open(fn,'r') as fp:
            for m in range(len(fstarts)):
                fp.seek(fstarts[m])
                line = fp.readline()
                [seq,mod,charge,ev,nmpks] = line.split()[1].split("|")
                charge = int(charge)
                ev = float(ev[:-2])
                info.append((seq,mod,charge,ev,0)) # dummy 0 for nce
                out = self.inptsr(info[-1])
                outseq[m] = out[0]
                if self.embed:
                    outch[m] = out[1]
                    outce[m] = out[2]
        out = [outseq, outch, outce] if self.embed else [outseq]
        return out, info
    
    def input_from_str(self, strings):
        """
        Create batch of model inputs from list of string input labels. 
        - If self.embed=True, then this function outputs charge and energy as 
        batch-size length arrays containing the their respective values for 
        each input. Otherwise the first output is just a single embedding tensor.
        
        :param strings: List of input labels. All input labels must have the
                        form {seq}/{charge}_{mods}_{ev}eV_NCE{nce}
        
        :return out: List of inputs to the model. Length is 3 if
                     self.embed=True, else length is 1.
        :return info: List of tuples of peptide data. Each tuple is ordered as
                      (seq,mod,charge,ev,nce).
        """
        if (type(strings)!=list)&(type(strings)!=np.ndarray): 
            strings = [strings]
        
        bs = len(strings)
        outseq = torch.zeros(
            (bs, self.channels, self.D.seq_len), dtype=torch.float32
        )
        if self.embed:
            outch = torch.zeros((bs,), dtype=torch.float32)
            outce = torch.zeros((bs,), dtype=torch.float32)

        info = []
        for m in range(len(strings)):
            [seq,other] = strings[m].split('/')
            osplit = other.split("_") #TODO Non-standard label
            if len(osplit)==3: osplit+=['NCE0'] #TODO Non-standard label
            [charge,mod,ev,nce] = osplit#other.split('_') #TODO Non-standard label
            charge = int(charge);ev = float(ev[:-2]);nce = float(nce[3:])
            info.append((seq,mod,charge,ev,nce))
            out = self.inptsr(info[-1])
            outseq[m] = out[0]
            if self.embed:
                outch[m] = out[1]
                outce[m] = out[2]
        
        out = [outseq, outch, outce] if self.embed else [outseq]
        return out, info
    
    def target(self, fstart, fp, mint=0, return_mz=False):
        """
        Create target, from streamlined dataset, to train model on.
        
        :param fstart: array of file positions for spectra to be predicted.
        :param fp: filepointer to streamlined dataset.
        :param mint: minimum intensity to include in target spectrum.
        :param return_mz: whether to return the corresponding m/z values for
                          fragment ions.
        
        :return target: pytorch array of intensities for all ions in output
                        output space.
        :return moverz: pytorch array of m/z values corresponding to ions
                        present in target array. All zeros if return_mz=False.
        """
        target = torch.full(
            (len(fstart), self.D.dicsz), mint, dtype=torch.float32
        )
        moverz = (torch.zeros((len(fstart), self.D.dicsz), dtype=torch.float32) 
                  if return_mz else 0)
        for i,p in enumerate(fstart):
            fp.seek(p)
            nmpks = int(fp.readline().split()[1].split("|")[-1])
            for pk in range(nmpks):
                [d,I,mz,intensity] = fp.readline().split()
                target[i,int(I)] = float(intensity)
                if return_mz: moverz[i,int(I)] = float(mz)
        return target, moverz
    
    def root_intensity(self, ints, root=2):
        """
        Take the root of an intensity vector

        Parameters
        ----------
        ints : intensity vector
        root : root value

        Returns
        -------
        ints : return transformed intensity vector

        """
        if root==2:
            ints[ints>0] = torch.sqrt(ints[ints>0]) # faster than **(1/2)
        else:
            ints[ints>0] = ints[ints>0]**(1/root)
        return ints

    def add_ce(self, label, ceadd=0, typ='ev'):
        """
        Add to collision energy in label.

        Parameters
        ----------
        label : Input label.
        ceadd : Float value to add to existing collision energy.
        typ : Either add to "ev" or "nce"

        Returns
        -------
        Label with added collision energy

        """
        hold = label.split('_')
        if len(hold)<4: hold += ['NCE0'] #TODO Non-standard label
        if typ=='ev': hold[-2] = '%.1feV'%(float(hold[-2][:-2])+ceadd)
        elif typ=='nce': hold[-1] = 'NCE%.1f'%(float(hold[-1][3:])+ceadd)
        return "_".join(hold)

    def inp_spec_msp(self, 
                     fstart, 
                     fp,
                     mint=1e-10):
        """
        Input spectrum from MSP file. Works on 1 spectrum at a time.
        
        :param fstart: File starting positions for "Name:..." labels in msp.
        :param fp: file pointer to msp file
        :param mint: minimum intensity for including peaks
        
        :output label: spectrum label
        :output #2: tuple of (masses, intensities, ions)
        """
        fp.seek(fstart)
        label = fp.readline().split()[1]
        
        # Proceed to the peak list
        for _ in range(5):
            pos = fp.tell()
            line = fp.readline()
            if line[:9]=='Num peaks':
                fp.seek(pos)
                break
        npks = int(fp.readline().split()[2])
        masses = np.zeros((npks,));Abs = np.zeros((npks,));ions=[]
        # count=1
        for m in range(npks):
            line = fp.readline()
            # print(npks, count, line);count+=1
            spl = '\t' if '\t' in line else ' '
            [mass,ab,ion] = line.split(spl)
            masses[m] = float(mass)
            Abs[m] = float(ab)
            ions.append(ion.strip()[1:-1].split(',')[0])
        Abs /= np.max(Abs)
        sort = Abs>mint
        return label, (masses[sort],Abs[sort],np.array(ions)[sort])
    
    def FPs(self, filename, criteria):
        """
        Get file positions of spectrum labels in msp file
        
        :param filename: filepath+name of msp file
        :param criteria: string of python comparison statements to be 
                         evaluated. Use spectrum attributes "seq", "charge",
                         "ev", "nce", or "mods".
        
        :return poss: numpy array of file positions for labels meeting criteria
        """
        with open(filename,'r') as f:
            _ = f.read()
            end = f.tell()
            
            poss = []
            f.seek(0)
            pos = 0
            while pos<end:
                pos = f.tell()
                line = f.readline()
                # Prevent pos from blowing up (over end)
                if line=='\n':
                    pos = f.tell()
                    line = f.readline()
                if (line[:5]=='Name:') | (line[:5]=='NAME:'):
                    if criteria==None:
                        poss.append(pos)
                    else:
                        [seq,other] = line.split()[1].split('/')
                        otherspl = other.split('_') #TODO Non-standard label
                        if len(otherspl)<4: otherspl+=['NCE0'] #TODO Non-standard label
                        [charge,mods,ev,nce] = otherspl #TODO Non-standard label
                        charge = int(charge)
                        ev=float(ev[:-2])
                        nce = float(nce[3:])
                        if eval(criteria):
                            poss.append(pos)
                if line[:3]=='Num':
                    nmpks = int(line.split()[-1])
                    for _ in range(nmpks): _ = f.readline()
        return np.array(poss)
    
    def FPs_from_labels(self, query_labels, msp_filename):
        """
        Search for matching labels in msp file, return the file position

        Parameters
        ----------
        query_labels : list of spectrum labels to search for in the msp file.
        msp_filename : Filepath+name of msp file

        Returns
        -------
        out : list of filepositions for the query labels

        """
        with open(msp_filename, 'r') as f:
            _ = f.read()
            end = f.tell()
            
            out = {label:-1 for label in query_labels}
            f.seek(0)
            count = 0
            pos = 0
            while pos<end:
                pos = f.tell()
                line = f.readline()
                if line=='\n':
                    pos = f.tell()
                    line = f.readline()
                if line[:5].upper()=='NAME:':
                    if line.strip().split()[1] in out.keys():
                        out[line.strip().split()[1]] = pos
                        count+=1
                if count==len(query_labels):
                    pos=end
        return np.array(list(out.values()))
    
    def Pos2labels(self, fname, fposs=None):
        """
        Get spectrum labels from msp file
        
        Parameters
        ----------
        fname : msp filepath+name to get labels from.
        fposs : file positions of labels. If None then use FPs() to get them.

        Returns
        -------
        labels : list of spectrum labels

        """
        # if len(fposs)==0 or fposs==None: 
        #     fposs = self.FPs(fname, 
        #             '(len(seq)<self.D.seq_len) & (charge<self.D.chlim[-1])'
        #     )
        labels = []
        with open(fname, 'r') as f:
            for I, pos in enumerate(fposs):
                f.seek(pos)
                label = f.readline().split()[1]
                labels.append(label)
        return labels

class EvalObj(LoadObj):
    def __init__(self, 
                 config,
                 model_list, 
                 dobj, 
                 enable_gpu=False,
                 curpath='./',
                 rawdatapath='C:/Users/jsl6/MyDatasets/MassSpec/'
                 ):
        self.model_list = model_list
        embed = model_list[0].CEembed
        super().__init__(dobj, embed)
        self.enable_gpu = enable_gpu
        for model in model_list: model.eval()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        if enable_gpu:
            self.model_list = [model.to(self.device) for model in model_list]
        self.model = lambda inp: torch.cat(
            [model(inp)[0] for model in self.model_list], 
            0).mean(0)
        self.AM = model_list[0]
        
        # Filenames of experimental msp files
        self.dsets = config['dsets']
        # Filenames of predicted msp files
        self.dsetspred = {}
        # Labels
        self.lab = {}
        
        self.ppm = lambda theor, exp: 1e6*(exp[None] - theor[:,None])/theor[:,None]
        self.diff = lambda theor, exp: exp[None] - theor[:,None]
        self.ionstats = {
            line.split()[0]:[int(line.split()[1]), float(line.split()[2])] 
            for line in open(curpath+'input_data/ion_stats/ion_stats_train.txt', 'r')
        }
        # proinds = np.loadtxt(path+'input_data/ion_stats/proinds.txt').astype('int')
        # self.prosit_filter = np.zeros(len(dobj.dictionary), dtype='int') 
        # self.prosit_filter[proinds]=1
    
    def add_labeldic(self, dset):
        """
        Get spectrum labels from dset and add it to member self.lab

        :param dset: dataset namne to add labels for
        """
        self.lab[dset] = {
            lab:I for I, lab in enumerate(
                self.Pos2labels(self.dsets[dset]['msp'], self.dsets[dset]['pos']))
        }
    
    def add_prosit(self):
        """
        Place any code here for adding prosit libraries for comparison
        """
        
        path = 'C:/Users/jsl6/Documents/Python Scripts/Pytorch/SpecPred/prosit/AIData6/'
        print("fnms['mabps'] = '.../NISTmAb20190711_Libtest_20221021.msp'")
        self.fnms['mabps'] = self.fnms['mab']
        
        print("fnmspred['mabps'] = '.../mab_PrositLib.msp'")
        self.fnmspred['mabps']  = path+'predictions/PrositLib/mab_PrositLib.msp'
        self.fnmspred['mab'] = self.fnmspred['mabps']
        
        print("fnms['valuniqps'] = '.../ValidUniq2022418_edit.msp'")
        self.fnms['valuniqps'] = self.fnms['valuniq']
        
        print("fnmspred['valuniqps'] = '.../valuniq_PrositLib.msp'")
        self.fnmspred['valuniqps'] = path+'predictions/PrositLib/valuniq_PrositLib.msp'
        self.fnmspred['valuniq'] = self.fnmspred['valuniqps']
        
        # Have to search mab msp file by labels in prosit library. Criteria in
        # FPs() will not work because prosit server treats '(ox)' as 4 in sequence
        # length.
        with open(self.fnmspred['mabps'], 'r') as f:
            labels = [line.split()[1] for line in f.read().split('\n') if line[:5]=='Name:']
        print("fpos['mabps'] = FPs_from_labels(labels, fnms['mab'])")
        self.fpos['mabps'] = self.FPs_from_labels(labels, self.fnms['mabps'])
        
        print("fpospred['mabps'] = FPs(fnmspred['mabps'],...)")
        self.fpospred['mabps'] = self.FPs(self.fnmspred['mabps'], None)
        
        with open(self.fnmspred['valuniqps'], 'r') as f:
            labels = [line.split()[1] for line in f.read().split("\n") if line[:5]=='Name:']
        print("fpos['valuniqps'] = FPs_from_labels(labels, fnms['valuniq'])")
        self.fpos['valuniqps'] = self.FPs_from_labels(labels, self.fnms['valuniqps'])
        
        print("fpospred['valuniqps'] = FPs(fnmspred['valuniqps'],None)")
        self.fpospred['valuniqps'] = self.FPs(self.fnmspred['valuniqps'], None)
    
    def lst2spec(self, mzlist, ablist):
        """
        Turn 2 lists, mz and abundance, into 2xN array
        
        :param mzlist: 1D list of masses
        :param ablist: 1D list of abundances
        
        :output: 2xN numpy array of [mz, abundances]
        """
        return np.concatenate([[mzlist],[ablist]],0)
    
    def match(self, mz1, mz2, thr=[15,20,25], spl=[800,1200], typ='ppm'):
        """
        Find matches between 2 peaks lists' m/z values
        - All non-matches will be classifiedfalse positives for mz1, and false 
           negatives for mz2

        Parameters
        ----------
        mz1 : First vector of peak m/z's', preferably the predicted peaks
        mz2 : Second vector of peak m/z's, preferably the experimental peaks'
        thr : Thresholds, under which peaks are considered matched. There
               should be 1 more value than length of 'spl'. The default is 
               [15,20,25].
        spl : m/z values on which to split dataset for different thresholds.
               Splits will be under first value, between middle values, and 
               over the top value. The default is '[800,1200]'.
        typ : Distance metric between peaks to be matched. Use either absolute 
               difference ('abs'), or ppm. The default is 'ppm'.

        Returns
        -------
        TP : Tuple of true positive indices for [0] mz1 peaks and [1] mz2 peaks
        FP1 : False positives indices for mz1 peaks
        FN2 : False negatives indices for mz2 peaks

        """
        delta = self.ppm(mz1,mz2) if typ=='ppm' else self.diff(mz1,mz2)
        TP1=[];TP2=[];FP1=[];FN2=[]
        for i,s in enumerate(thr):
            # smaller than first, between middle, over the top
            split = (mz1>spl[i-1] if i==(len(thr)-1) else 
                     (mz1<spl[i] if i==0 else ((mz1>spl[i-1])&(mz1<spl[i])) ) )
            S = np.where(split)[0]
            tp1 = S[np.where(abs(delta[split]).min(1) < s)[0]]
            tp2 = abs(delta[tp1]).argmin(1)
            # assert(sum((diff<thr).sum(1)>1)==0)
            fp1 = S[np.where(abs(delta[split]).min(1) > s)[0]]
            split = (mz2>spl[i-1] if i==(len(thr)-1) else 
                     (mz2<spl[i] if i==0 else ((mz2>spl[i-1])&(mz2<spl[i])) ) )
            fn2 = np.where(split)[0][np.where(abs(delta[:,split]).min(0) > s)[0]]
            TP1.append(tp1);TP2.append(tp2);FP1.append(fp1);FN2.append(fn2)
        TP1 = np.concatenate(TP1);TP2 = np.concatenate(TP2)
        FP1 = np.concatenate(FP1);FN2 = np.concatenate(FN2)
        return (TP1,TP2), FP1, FN2
    
    def CosineScore(self, peaks1, peaks2, pions=None, rions=None, thr=[15,20,25],
                    tbthr=2e-2, root=2, mxpks=400, remove_p=True, return_css=False):
        """
        Calculate the cosine similarity between 2 peak lists
        - sum(peaks1[TP]*peaks2[TP]) / norm(peaks1) / norm(peaks2)
        
        Parameters
        ----------
        peaks1 : 2xN array; first row are m/z, second row are abundances
        peaks2 : 2xN array; first row are m/z, second row are abundances
        thr : Maximum absolute ppm between matched peaks.
        tbthr : tiebreaker threshold
        root : take the root of intensity vectors
        mxpks : maximum number of intense experimental peaks to include
        remove_p : remove p-ions from scoring
        return_css : return arrays of TP,FP,FN indices that were included in
                     score
        
        Returns
        -------
        cs : Single float for cosine score
        (TP,FP,FN) : Tuple of TP, FP, and FN (following match()).
                     - Indices are that of original peaks1 and peaks2, i.e.
                       following mxpks filter and match() function. 
        """
        
        def RmP(ions, isp=False):
            # Remove all p-ions
            return (
                np.array([True if ion[0]=='p' else False for ion in ions]) if isp else
                np.array([False if ion[0]=='p' else True for ion in ions])    
                    ) if len(ions)>0 else []
        
        def tiebreak(peaks1, peaks2, TP1, TP2, thr):
            """
            Must get rid of multiple predicted peaks (peaks1) that match a
             single experimental peak (peaks2).

            Parameters
            ----------
            peaks1 : 2xN predicted peaks [mz;ab]
            peaks2 : 2xM experimental peaks  [mz;ab]
            TP1 : Original true positive indices for predicted peaks
            TP2 : Original true positive indices for experimental peaks

            Returns
            -------
            peaks1 : peaks1 without the updated mz and ab values, and without
                      the duplicate peaks.
            TP1 : New, smaller, true positive indices for predicted peaks
            TP2 : New, smaller, true positive indices for experimental peaks

            """
            # unique experimental TP indices, and the counts for each
            uniqs, i, cnts = np.unique(TP2, return_index=True, return_counts=True)
            if len(TP2)==len(uniqs): return peaks1, TP1, TP2#, []
            multi_inds = uniqs[np.where(cnts>1)[0]] # all duplicate TP2 values
            dels=[]
            for ind in multi_inds:
                # indices of TP2 array that match its duplicate value (which is in turn an index for the peaks2 array)
                nest_inds = np.where(TP2==ind)[0]
                expmz = peaks2[0][TP2[nest_inds]]
                predmz = peaks1[0][TP1[nest_inds]]
                # Make sure duplicates are below tiebreaker threshold
                inds2 = np.where(abs(predmz-expmz)<thr)[0]
                if len(inds2)==0:
                    for m in nest_inds[np.argsort(abs(predmz-expmz))[1:]]: dels.append(m)
                else:
                    I = TP1[nest_inds[inds2]] # Values of TP1 for duplicates
                    # use m/z of most intense duplicate predicted peak
                    highintarg = I[peaks1[1][I].argmax()]
                    peaks1[0][I[0]] = peaks1[0][highintarg]
                    # use sum of intensities of duplicate predicted peaks for single resulting peak
                    smint = peaks1[1][I].sum()
                    peaks1[1][I[0]] = smint
                    # Changed the values for duplicate peak, collect indices of others to get rid of
                    for m in nest_inds[np.where(abs(predmz-expmz)>thr)[0]]: dels.append(m)
                    for m in nest_inds[inds2[1:]]: dels.append(m)
            #global_dels = TP1[dels]
            TP1 = np.delete(TP1, dels)
            TP2 = np.delete(TP2, dels)
            assert(len(TP2)==len(uniqs))
            return peaks1, TP1, TP2#, global_dels
        
        def LowMz(mz1, mz2, maxmz=300, thr=3e-3, thr2=4e-3, ions=[]):
            """
            Special criterion for very low mz, using absolute difference

            Parameters
            ----------
            mz1 : mzs for predicted peaks
            mz2 : mzs for experimental peaks
            maxmz : Maximum m/z, below which criterion will be applied. 
                    The default is 300.
            thr : Threshold for maximum absolute value difference. The default 
                  is 3e-3.
            thr2 : Threshold for monoisotopic peaks. The default is 4e-3.

            Returns
            -------
            Indices of true positives to remove

            """
            
            lt = mz1<maxmz
            diff = abs(mz1-mz2)[lt]
            neg = np.array([d > thr2 if '+i' in ion else d > thr 
                            for ion,d in zip(ions[lt],diff)])
            return np.where(lt)[0][neg] if len(neg)>0 else neg
        
        def root_intensity(ints, root=2):
            if root==2:
                ints[ints>0] = np.sqrt(ints[ints>0])
            else:
                ints[ints>0] = ints[ints>0]**(1/root)
            return ints
        
        # Top {mxpks} experimental peaks
        if peaks2.shape[1]>mxpks:
            toppks = np.sort(np.argsort(peaks2[1])[-mxpks:])
            peaks2 = peaks2[:,toppks]
        else: toppks = np.arange(peaks2.shape[1])
        # Match peaks
        TP, FP, FN = self.match(peaks1[0], peaks2[0], thr, typ='ppm')
        TP1,TP2 = TP # unpack TP so that TP1 and TP2 can be further operated on
        fp,fn = FP,FN # same for FP and FN
        TP = (TP1,toppks[TP2]) # Put it back together for output purposes (don't return operated TP)
        # Remove duplicate TP2's
        # - i.e. 2 very close predicted peaks that match the same experimental peak
        peaks1, TP1, TP2 = tiebreak(peaks1, peaks2, TP1, TP2, tbthr)
        # Apply low m/z threshold
        dels = LowMz(peaks1[0][TP1], peaks2[0][TP2], ions=pions[TP1])
        if len(dels)>0:
            fp = np.sort(np.append(fp, TP1[dels])) # add nonTPs to FP
            fn = np.sort(np.append(fn, TP2[dels])) # add nonTPs to FN
            TP1 = np.delete(TP1, dels) # remove TPs
            TP2 = np.delete(TP2, dels) # remove TPs
        # Root intensities
        pred = root_intensity(peaks1[1], root=root)
        exp = root_intensity(peaks2[1], root=root)
        # - Remove p-ions
        rmptp = ([] if len(TP1)==0 else 
                 (RmP(pions[TP1])&RmP(rions[toppks][TP2]) 
                  if remove_p else 
                  np.array(len(TP1)*[True]))
                 )
        rmallpred = RmP(pions) if remove_p else np.array(len(pions)*[True])
        rmallexp  = RmP(rions[toppks]) if remove_p else np.array(len(toppks)*[True])
        cs = (
              sum(pred[TP1[rmptp]]*exp[TP2[rmptp]]) 
              / (np.linalg.norm(pred[rmallpred])+1e-10) 
              / (np.linalg.norm(exp[rmallexp])+1e-10)
              )
        assert(cs<=1)
        
        return2 = (
            ((TP1[rmptp],toppks[TP2[rmptp]]),fp[RmP(pions[fp])],toppks[fn[RmP(rions[fn])]]) 
            if return_css else 
            (TP,FP,toppks[FN])
            )
        return cs, return2
    
    def filter_fake(self, pepinfo, masses, ions):
        """
        Filter out the ions which cannot possibly occur for the peptide being
         predicted.

        Parameters
        ----------
        pepinfo : tuple of (sequence, mods, charge, ev, nce). Identical to
                   second output of str2dat().
        masses: array of predicted ion masses
        ions : array or list of predicted ion strings.

        Returns
        -------
        Return a numpy boolean array which you can use externally to select
         indices of m/z, abundance, or ion arrays

        """
        (seq,mods,charge,ev,nce) = pepinfo
        
        # modification
        # modlst = []
        # Mstart = mods.find('(') if mods!='0' else 1 # EDIT 230121
        # modamt = int(mods[0:Mstart])
        # if modamt>0:
        #     Mods = mods[Mstart:].split(')(') # )( always separates modifications
        #     for mod in Mods:
        #         [pos,aa,typ] = re.sub('[()]', '', mod).split(',') # mod position, amino acid, and type
        #         modlst.append([int(pos), self.D.mass[typ]])
        
        filt = []
        for ion in ions:
            ext = (len(seq) if ion[0]=='p' else 
                   (int(ion[1:].split('-')[0].split('+')[0].split('^')[0])
                    if (ion[0] in ['a','b','y']) else 0)
                   )
            a = True
            if "Int" in ion:
                [start,ext] = [
                    int(j) for j in 
                    ion[3:].split('+')[0].split('-')[0].split('>')[:2]
                ]
                # Do not write if the internal extends beyond length of peptide
                if (start+ext)>=len(seq): a = False
            if (
                (ion[0] in ['a','b','y']) and 
                (int(ion[1:].split('-')[0].split('+')[0].split('^')[0])>(len(seq)-1))
                ):
                # Do not write if a/b/y is longer than length-1 of peptide
                a = False
            if ('H3PO4' in ion) & ('Phospho' not in mods):
                # Do not write Phospho specific neutrals for non-phosphopeptide
                a = False
            if ('CH3SOH' in ion) & ('Oxidation' not in mods):
                a = False
            if ('CH2SH' in ion) & ('Carbamidomethyl' not in mods):
                a = False
            if ('IPA' in ion) & ('P' not in seq):
                a = False
            filt.append(a)
        # Qian says all masses must be >60 and <1900
        return (np.array(filt)) & (masses>60)
    
    def theor_spec(self, label):
        """
        Create a theoretical spectrum (only m/z values) using ion dictionary
         in the dictionary object D

        Parameters
        ----------
        label : spectrum label {seq}/{charge}_{mods}_{ev}eV_NCE{nce}

        Returns
        -------
        masses : list of m/z values, sorted in ascending order
        ions : corresponding numpy array of ion strings

        """
        pepinfo = self.str2dat(label)
        seq,mods,charge,ce = pepinfo
        
        ions = np.array(list(self.D.dictionary.keys()))
        masses = np.array([self.D.calcmass(seq,charge,mods,ion) for ion in ions])
        
        sort = masses.argsort()
        masses = masses[sort]
        ions = ions[sort]
        
        filt = self.filter_fake(pepinfo, masses, ions)
        return masses[filt], ions[filt]
    
    def predict_spectrum_new(self,
                             label,
                             mint=5e-4,
                             maxnorm=True,
                             rm_fake=True,
                             rm_lowmz=False,
                             minmz = 0,
                             prosit_inds=False
                             ):
        """
        Predict a spectrum from input of string/peptide label.
        - Predicts only 1 spectrum at a time

        Parameters
        ----------
        label : Peptide label of form {seq}/{charge}_{mods}_{ev}eV_NCE{nce}
        mint : Minimum relative abundance/intensity, above which predicted
                peaks are included. The default is 1e-3.
        maxnorm : Boolean to normalize spectrum by setting maximum intensity to
                   1. The default is True.
        rm_fake : Boolean to filter out impossible ions. The default is True.
        rm_lowmz : Boolean to remove all peaks with m/z below minmz. The 
                    default is False.
        minmz : Minimum m/z value, above which peaks are included. Only used if
                 rm_lowmz=True. The default is 0.
        prosit_inds : Zero out all non-prosit indices (b/y, <=^3, <=length 30) 
                       in prediction. The default is False.

        Returns
        -------
        1. Tuple of (m/z values, abundance values, ion strings)
        2. Tuple of (sequence, modifications, charge, eV, NCE)

        """
        intor, pepinfo = self.input_from_str(label)
        (seq,mods,charge,ev,nce) = pepinfo[0]
        
        with torch.no_grad():
            if self.enable_gpu: intor = [m.to(self.device) for m in intor]
            pred = self.model(intor)
        pred = (
            pred.squeeze().detach().cpu().numpy()*self.prosit_filter 
            if prosit_inds else 
            pred.squeeze().detach().cpu().numpy()
        )
        if maxnorm: pred /= pred.max()
        piboo = (pred>mint)
        rdinds = np.where(piboo)[0]
        pions = np.array([self.D.revdictionary[ind] for ind in rdinds])
        pints = np.array(pred[piboo])
        pmass = np.array([self.D.calcmass(seq,charge,mods,ion) for ion in pions])
        sort = np.argsort(pmass)
        pmass = pmass[sort]
        pints = pints[sort]
        pions = pions[sort]
        
        if rm_lowmz:
            filt = pmass>minmz
            pmass = pmass[filt]
            pints = pints[filt]
            pions = pions[filt]
        if rm_fake:
            filt = self.filter_fake(pepinfo[0], pmass, pions)
            pmass = pmass[filt]
            pints = pints[filt]
            pions = pions[filt]
        return (pmass,pints,pions), (seq,mods,charge,ev,nce)
    
    def _inp_spec(self, index=0, dset='valuniq', typ='raw'):
        """
        Input spectrum from MSP file. Works on input of spectrum index and 
        dataset name (dset)
        - easier to use than inp_spec_raw(file_pos, file_pointer)
        
        :param index: index number of spectrum in dataset "dset"
        :param dset: name of dataset
        :param typ: input from "raw" msp or "pred" msp
        
        :return label: spectrum label
        :return specdata: tuple of spectrum (masses, abundances, annotations)
        """
        fnm = self.dsetspred[dset]['msp'] if typ=='pred' else self.dsets[dset]['msp']
        pos = self.dsetspred[dset]['pos'] if typ=='pred' else self.dsets[dset]['pos']
        with open(fnm, 'r') as fp:
            label, specdata = self.inp_spec_msp(pos[index], fp)
        return label, specdata
    
    def _write_msp(self, 
                   specdata,
                   specinfo,
                   comment=None,
                   mint=5e-4, 
                   maxnorm=True,
                   cecorr=0,
                   cetyp='ev',
                   fn="out.msp"
        ):
        
        """
        Print prediction to file in msp format. This subroutine is called by
        self.write_msp().
        
        Parameters
        ----------
        specdata : should be list of tuples of (masses, intensities, ion_names)
        specinfo : should be list of tuples of (seq, mods, charge, ev)
        comment : should be a string with any additional comment info
        mint : minimum intensity to include predictions
        maxnorm : set most intense peak to 1.0
        cecorr : collision energy correction
        cetyp : units for collision energy
        fn : output filename
        
        """
        
        def ModSeq(seq, mods):
            """
            Parameters
            ----------
            seq : Capitalized sequence
            mods : PTM string

            Returns
            -------
            seqcase : Sequence with lowercase modified aa's 
            
            """
            Mstart = mods.find('(') if mods!='0' else 1
            modamt = int(mods[0:Mstart])
            Mods = ([re.sub("[()]",'',m).split(',') for m in 
                     mods[Mstart:].split(')(')] 
                    if modamt>0 else [])
            
            # seqcase will be written in the internal style, i.e. lowercase PTMs
            seqcase =  list(seq)
            for mod in Mods:
                # [pos,aa,typ] = [int(mod[0]),mod[1],mod[2]]
                pos = int(mod[0])
                seqcase[pos] = seqcase[pos].lower()
            seqcase = "".join(seqcase)
            return seqcase
        
        if comment==None: comment = len(specdata)*['']
        with open(fn, "w") as f:
            for i in range(len(specinfo)):
                (mass, ints, ions) = specdata[i]
                (seq,mods,charge,ev,nce) = specinfo[i]
                Seq = ModSeq(seq, mods)
                EV = ev;NCE = nce
                if cetyp=='ev': EV-=cecorr
                elif cetyp=='nce': NCE-=nce-cecorr
                
                Parent = self.D.calcmass(seq,charge,mods,'p')
                MW = Parent*charge # or just use the Parent mass?
                
                # Write header lines
                label = '%s/%d_%s_%.1feV_NCE%.1f'%(seq,charge,mods,EV,NCE)
                f.write("Name: %s\n"%label) # EDIT 230116
                f.write('MW: %.4f\n'%MW)
                f.write('Comment: Single Pep=Tryptic Collision_energy=%.1f '%ev)
                f.write('Mods=%s Fullname=R.%s.L Charge=%d Parent=%.4f %s\n'%(
                    mods,seq,charge,Parent,comment[i])
                )
                
                # Write peaks
                f.write('Num peaks: %d\n'%len(mass))
                for pknum, (mass,ab,ion) in enumerate(zip(*specdata[i])):
                    
                    # Convert internal to normal notation
                    if "Int" in ion:
                        [start,ext] = [
                        int(j) for j in 
                        ion[3:].split('+')[0].split('-')[0].split('>')[:2]
                        ]
                        back = ion[len(str(start))+len(str(ext))+4:] # if anything was tacked on to internal
                    # Convert Internal to normal notation, else use original ion annotation
                    ion2 = ('Int/%s/%d'%(Seq[start:start+ext]+back,start) 
                            if 'Int' in ion else ion)
                    ab = int(ab*10000) # save memory EDIT 230115
                    f.write('%.4f\t%d\t\"%s\"\n'%(mass,ab,ion2)) # EDIT 230115
                f.write('\n')
    
    def write_msp(self,
                  inp,
                  cecorr=0,
                  comments=None,
                  print_every=None,
                  outfn='out.msp'
                  ):
        """
        Generate msp file for list of peptide labels, or from an existing msp
        file.

        Parameters
        ----------
        inp : input can be list of peptide labels or dataset name string
        cecorr : collision energy correction
        comments: list of strings to add to end of comments entry in msp
        print_every: Integer to print out intermediate results every {} steps.
                     If None, only print out final file
        outfn : output filename
        
        """
        
        if type(inp)==list:
            if comments != None: assert(len(inp)==len(comments)) # EDIT 230118
            specdata = []
            specinfo = []
            last = 0
            for count, label in enumerate(inp):
                print("\r%d/%d"%(count+1, len(inp)), end='')
                out = self.predict_spectrum_new(self.add_ce(label, cecorr))
                specdata.append(out[0])
                specinfo.append(out[1])
                # cecorr subtracts from the specinfo eV when writing Name entry in msp
                if (print_every is not None): # EDIT 230122
                    if ((count%print_every)==0) & (count!=0):
                        Com = None if comments==None else comments[last:count+1] # EDIT 230118
                        self._write_msp(
                            specdata, specinfo, cecorr=cecorr, 
                            comment=Com, 
                            fn=outfn+'_%0000007d_%0000007d'%(last,count)
                        )
                        specdata = []
                        specinfo = []
                        last = count
            if print_every==None:
                self._write_msp(specdata, specinfo, cecorr=cecorr, 
                                comment=comments, fn=outfn
                )
            else:
                Com = None if comments==None else comments[last:count+1] # EDIT 230118
                self._write_msp(specdata, specinfo, cecorr=cecorr, comment=Com, 
                                fn=outfn+'_%d'%count
                )
                
        elif type(inp)==str:
            assert inp in self.dsets.keys(), "inp not in dsets.keys()"
            if self.dsets[inp]['pos']=='None':
                print("Searching for file positions in %s"%inp)
                Pos = self.FPs(
                    self.dsets[inp]['msp'], 
                    '(len(seq)<self.D.seq_len)&(charge<self.D.chlim[-1])'
                )
            else:
                Pos = np.loadtxt(self.dsets[inp]['pos'])
            # Get data on raw spectra
            # - cecorr adds to the label in inp_spec_msp
            with open(self.dsets[inp]['msp'], 'r') as fp:
                out = [self.inp_spec_msp(pos, fp) for pos in Pos]
            labels = [o[0] for o in out]
            rawdata = [o[1] for o in out]
            comment = [];Specdata = [];Specinfo = [];CS=0
            for a, label in enumerate(labels):
                print("\r%5d/%5d"%(a+1, len(labels)),end='')
                # Get prediction data
                specdata, specinfo = self.predict_spectrum_new(
                    self.add_ce(label, cecorr), 
                    rm_lowmz=True, 
                    minmz=min(rawdata[a][0])-0.1
                )
                Specdata.append(specdata)
                Specinfo.append(specinfo)
                # Get cosine score between prediction and raw
                pred_peaks = self.lst2spec(specdata[0], specdata[1])
                raw_peaks = self.lst2spec(rawdata[a][0], rawdata[a][1])
                pions = specdata[-1]
                rions = rawdata[a][-1]
                cs, diffs = self.CosineScore(
                    pred_peaks, raw_peaks, pions=pions, rions=rions
                )
                comment.append("Cosine_score: %.3f"%cs)
                CS+=cs
            # cecorr subtracts from the specinfo eV when writing Name entry in msp
            # - eV is the same as predicted in Comment entry
            self._write_msp(Specdata, Specinfo, cecorr=cecorr, 
                            comment=comment, fn=outfn
            )
            print("\nmean(CS) = %.3f"%(CS/a))
            
    def mirrorplot(self,
                   index=0,
                   rawdset='valuniq',
                   predict=True,
                   cecorr=0,
                   labelsearch=True,
                   maxnorm=True,
                   prosit_inds=False,
                   maxraw=None, # lessen the intense peaks, pick up low peaks
                   save=False):
        """
        Plot interactive mirror plot for spectrum in your fnms dataset

        Parameters
        ----------
        index : Spectrum index in your dataset.
        rawdset : Dataset name from self.fnms.
        predict : Predict a spectrum with the model (True) or load a previous
                  prediction (False) from a generated library.
        cecorr : Collision energy correction to add to a predicted spectrum.
        labelsearch : If True, search for the raw label in predicted library.
        maxnorm : Normalize max intensity to 1 if True
        prosit_inds : Only plot the prosit ions in your prediction.
        maxraw : Set most intense peaks to maxraw and re-normalize.
                 - Picks up the intensity of the smaller peaks.
        Returns
        -------
        None.

        """
        
        plt.close('all')
        # self.model.eval()
        # self.model.to("cpu")
        
        # Get raw data
        with open(self.dsets[rawdset]['msp'], 'r') as g:
            Pos = np.loadtxt(self.dsets[rawdset]['pos'])
            label, (rawmz,rawab,rawion) = self.inp_spec_msp(
                Pos[index], g
            )
            label = self.add_ce(label, cecorr)
            if maxraw != None:
                rawab[rawab>maxraw] *= maxraw
                rawab /= maxraw
        # Get predicted data
        if predict==True: # use model to predict spectrum
            (pmz,pab,pions), (seq,mods,charge,ev,nce) = self.predict_spectrum_new(
                label, maxnorm=maxnorm, prosit_inds=prosit_inds
            )
        else: # predicted spectrum is from existing msp file
            # find the label on the fly. Requires same filename as raw dataset. 
            # Use e.g. mab, valuniq, etc.    
            if labelsearch:
                pos = self.FPs_from_labels([label], self.fnmspred[rawdset])[0]
                assert pos!=-1, "Label not found."
            # use precalculated positions in predicted msp. Use e.g. mabps, valuniqps
            else: 
                pos = self.fpospred[rawdset][index]
            with open(self.fnmspred[rawdset],'r') as g:
                label, (pmz,pab,pions) = self.inp_spec_msp(pos, g)
            [seq,other] = label.split('/')
            [charge,mods,ev,nce] = other.split('_')
            charge = int(charge)
            ev = float(ev[:-2])
            nce = float(nce[3:])
        
        fig,ax = plt.subplots()
        fig.set_figwidth(15)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        
        ax.vlines(rawmz, ymin=0, ymax=rawab, linewidth=1, color='red')
        ax.vlines(pmz, ymin=-pab, ymax=0, linewidth=1, color='blue')
        
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([-1.1,1.1])
        ax.set_xlim([0,2000])
        ax.set_xticks(np.arange(0,ax.get_xlim()[1],500))
        ax.set_xticks(np.arange(0,ax.get_xlim()[1],100), minor=True)
        
        predpeaks = self.lst2spec(pmz,pab)
        rawpeaks = self.lst2spec(rawmz,rawab)
        sim, (TP,FP,FN) = self.CosineScore(predpeaks, rawpeaks, pions, rawion)
        ax.set_title(
        "Seq: %s(%d); Charge: +%d; eV: %.1f; NCE: %.1f; Mod: %s; Sim=%.3f"%(
            seq, len(seq), charge, ev, nce, mods, sim)
        )
        
        if save:
            fig.savefig("C:/Users/jsl6/Desktop/mirroplot%d.jpg"%(index))
            plt.close()
    
    def plotAM(self, 
               label,
               layer=0,
               cols=4,
               maxmap=16,
               trunc=False, 
               save=False):
        """
        Plot the attention maps for a single predicted peptide.

        Parameters
        ----------
        label : Input peptide label to be predicted
        layer : Layer to plot attention heads. If <0 then it plots all layers
                for a single head dimension.
        cols : How many columns to have in the figure
        maxmap : Maximum attention maps to plot for a layer. If None, plot all.
        trunc : If True, truncate attention map to non-null amino acids.
        save : Save figure to file
        """
        samples, info = self.input_from_str([label])
        (seq,mod,charge,ev) = info[0]
        with torch.no_grad():
            out1,out2,FMs = self.AM(samples)
        length = len(seq)
        seq += (self.D.seq_len-length)*'X'
        
        plt.close('all')
        if layer>-1:
            maxmaps = FMs[layer].shape[-1] if maxmap==None else maxmap
            rows = maxmaps//cols
            fig,axes = plt.subplots(rows, cols)
            if rows==1: axes = np.expand_dims(axes,0)
            fig.text(0.05,0.95,"%s(%d)"%(seq[:length],length))
            map_index = np.random.choice(np.arange(FMs[layer].shape[-1]), maxmaps)
            for m in range(rows):
                for n in range(cols):
                    axes[m,n].imshow(
                        FMs[layer][0, ..., map_index[m*cols+n]].detach().numpy()
                    )
                    axes[m,n].xaxis.set_ticks(np.arange(40))
                    axes[m,n].xaxis.set_ticklabels(list(seq), fontsize=7)
                    axes[m,n].yaxis.set_ticks(np.arange(40))
                    axes[m,n].yaxis.set_ticklabels(list(seq), fontsize=7)
                    if trunc:
                        axes[m,n].set_xlim([0,length-1])
                        axes[m,n].set_ylim([length-1,0])
                    # axes[m,n].invert_yaxis()
                    axes[m,n].xaxis.tick_top()
        else:
            rows = len(FMs)
            fig, axes = plt.subplots(rows, cols)
            fig.text(0.05,0.95,"%s(%d)"%(seq[:length],length))
            for m in range(rows):
                for n in range(cols):
                    axes[m,n].imshow(FMs[m][0, ..., n].detach().numpy())
                    if trunc:
                        axes[m,n].set_xlim([0,length-1])
                        axes[m,n].set_ylim([length-1,0])

        if save:
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.set_figwidth(20)
            fig.set_figheight(12)
            fig.tight_layout()
            fig.savefig("C:/Users/jsl6/Desktop/hold.jpg")
