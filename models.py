# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:14:51 2021

@author: jsl6
"""
verbose=False # print out intialization statistics for various tensors.

import torch
from torch import nn
import numpy as np

# NOTE: Target sigma
# sigma(a(0,siga).dot(b(0,sigb))) = siga*sigb*(dot_dim_len)**0.5
# try to aim for small std on residual (~ <=0.1) with sig_inp=1

"""
Initializers for various model tensors
- Very ad hoc for head(16,16,64), embedsz=256. If these parameters change,
  may need to change the initializers somewhat for optimal training.
"""
nm = 0.03
initEmb = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, nm)
initPos = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, nm)
initQ = lambda shps: (
    nn.init.normal_(torch.empty(shps), 0.0, shps[0]**-0.5*shps[1]**-0.25)
)
initK = lambda shps: (
    nn.init.normal_(torch.empty(shps), 0.0, shps[0]**-0.5*shps[1]**-0.25)
)
initL = lambda shps, eltargstd=0.5: (
    nn.init.normal_(torch.empty(shps), 0.0, eltargstd*shps[0]**-0.5)
)
initW = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, 0.01)
initV = lambda shps, wtargstd=0.012, sigWw=1, seq_len=40: (
    nn.init.normal_(torch.empty(shps), 0.0, 
    wtargstd**-1 * 
    shps[-1]**-0.5 * 
    shps[0]**-0.5 * 
    seq_len**-0.5 * 
    sigWw**-1)
)
initO = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, nm)
initFFN = lambda shps, stdin=1, targ=1: (
    nn.init.normal_(torch.empty(shps), 0.0, targ*shps[0]**-0.5**stdin**-1)
)
initProj = lambda shps: nn.init.normal_(torch.empty(shps), 0.0, shps[0]**-0.5)
initFin = lambda tsr: nn.init.normal_(tsr, 0.0, .1)
# initFin = lambda tsr: nn.init.xavier_normal_(tsr)

class TalkingHeads(nn.Module):
    def __init__(self,
                 in_ch,
                 dk,
                 dv,
                 hv,
                 hk=None,
                 h=None,
                 drop=0,
                 out_dim=None
                 ):
        """
        Talking heads attention - Same as multi-headed attention, but for two 
        linear transformations amongst head dimensions before and after softmax.

        Parameters
        ----------
        in_ch : Dimension 1 of incoming tensor to the layer
        dk : Number of channels for queries and keys tensors
        dv : Number of channels in values tensor, projecting from softmax layer
        hv : Heads in values tensor
        hk : Heads in queries and keys tensors. None defaults to hv value
        h : Heads in softmax layer. None defaults to hv value
        drop : Dropout rate for residual layer
        out_dim : Channels in the output. None defaults to in_ch

        """
        super(TalkingHeads, self).__init__()
        self.dk = dk
        self.dv = dv
        self.hv = hv
        self.hk = hv if hk==None else hk
        self.h = hv if h==None else h
        self.out_dim = in_ch if out_dim==None else out_dim
        self.shortcut = (nn.Linear(in_ch, self.out_dim, bias=False) 
                         if self.out_dim!=in_ch else 
                         nn.Identity()
        )
        
        self.Wq = nn.Parameter(initQ((in_ch, dk, self.h)), requires_grad=True)
        #self.bq = nn.Parameter(torch.zeros(dk, self.h), requires_grad=True)
        self.Wk = nn.Parameter(initK((in_ch, dk, self.h)), requires_grad=True)
        #self.bk = nn.Parameter(torch.zeros(dk, self.h), requires_grad=True)
        self.Wv = nn.Parameter(initV((in_ch, dv, self.h)), requires_grad=True)
        #self.bv = nn.Parameter(torch.zeros(dv, self.h), requires_grad=True)
        self.alphaq = nn.Parameter(
            nn.init.constant_(torch.empty(1,), 2.), requires_grad=True
        ) # 2.
        self.alphak = nn.Parameter(
            nn.init.constant_(torch.empty(1,), 2.), requires_grad=True
        ) # 2.
        self.alphav = nn.Parameter(
            nn.init.constant_(torch.empty(1,), 2.), requires_grad=True
        ) # 2.
        
        self.Wl = nn.Parameter(initL((self.hk, self.h)), requires_grad=True)
        self.Ww = nn.Parameter(initW((self.h, self.hv)), requires_grad=True)
        
        self.Wo = nn.Parameter(initO((dv*self.h, self.out_dim)), requires_grad=True)
        #self.bo = nn.Parameter(torch.zeros(self.out_dim), requires_grad=True)
        self.drop = nn.Identity() if drop==0 else nn.Dropout(drop)
        
        if verbose:
            print("Wq=%f"%self.Wq.std())
            print("Wk=%f"%self.Wk.std())
            print("Wv=%f"%self.Wv.std())
            print("Wl=%f"%self.Wl.std())
            print("Ww=%f"%self.Ww.std())
            print("Wo=%f"%self.Wo.std())
    def forward(self, inp, mask, test=True):
        """
        Talking heads forward function

        Parameters
        ----------
        inp : Input tensor on which to perform attention. 
              inp.shape = (bs, in_ch, seq_len)
        mask : Vector that masks out null amino acids from softmax
               calculation. Set to zeros tensor if masking is undesired.
               mask.shape = (bs, seq_len)
        test : Set to True if you want to calculate and return intermediate
               activation statistics, False to run faster during training.

        Returns
        -------
        Returns inp + residual(inp). Best to initialize so that residual is
        very small.
        
        activations : Means and standard deviations of intermediate tensors.
                      Useful for diagnosing instabilities, especially in J layer.
        W : Attention maps, if you want to visualize them.

        """
        Q = torch.sigmoid(self.alphaq)*torch.einsum('abc,bde->adce', inp, self.Wq)
        K = torch.sigmoid(self.alphak)*torch.einsum('abc,bde->adce', inp, self.Wk)
        V = torch.sigmoid(self.alphav)*torch.einsum('abc,bde->adce', inp, self.Wv)  
        
        J = torch.einsum('abcd,abed->aced', Q, K)
        EL = torch.einsum('abcd,de->abce', J, self.Wl) - mask[:, None, :, None]
        W = torch.softmax(EL, dim=2) # %1 zeros
        U = torch.einsum('abcd,de->abce', W, self.Ww)
        O = torch.einsum('abcd,aecd->abed', U, V)
        O = O.reshape(O.shape[0], -1, self.dv*self.hv)
        resid = self.drop(torch.einsum('abc,cd->adb', O, self.Wo))
        
        if test:
            Q_ = Q.mean();Q__ = Q.std();K_ = K.mean();K__ = K.std()
            V_ = V.mean();V__ = V.std();J_ = J.mean();J__ = J.std()
            EL_ = EL.mean();EL__ = EL.std();FM = W.max(2)[0].mean()
            U_ = U.mean();U__ = U.std();O_ = O.mean();O__ = O.std()
            resid_ = resid.mean();resid__ = resid.std()
            activations = (Q_, Q__, K_, K__, V_, V__, J_, J__, EL_, EL__, FM, 
                           U_, U__, O_, O__, resid_, resid__)
        else:
            activations = ()
        
        INP = self.shortcut(inp.transpose(-1,-2)).transpose(-1,-2)
        return INP + resid, activations, W

class FFN(nn.Module):
    def __init__(self,
                 in_ch,
                 units=None,
                 embed=None,
                 drop=0
                 ):
        """
        Feed forward network
        
        :param in_ch: Dimension 1 of incoming tensor.
        :param units: Intermediate units projected to before ReLU. If None,
                      then defaults to in_ch.
        :param embed: Incoming units for for tensor adding charge and 
                      collision energy embedding before ReLU. If None then no
                      embedding is added in between linear transofrmations.
        :param drop: Dropout rate for residual layer.
        
        """
        super(FFN, self).__init__()
        units = in_ch if units==None else units
        self.embed = embed
        self.W1 = nn.Parameter(initFFN((in_ch, units), 1, 1))
        self.W2 = nn.Parameter(initFFN((units, in_ch), 1, 0.1))
        if self.embed is not None:
            self.chce =  nn.Linear(self.embed, units)
        self.drop = nn.Identity() if drop==0 else nn.Dropout(drop)
        if verbose:
            print("FFN=%f"%self.W1.std())
    def forward(self, inp, embinp=None, test=True):
        """
        Feed forward network forward function.

        Parameters
        ----------
        inp : Input tensor. inp.shape = (bs, in_ch, seq_len)
        embinp : Embedded input of charge and collision energy fourier features.
                 embinp.shape = (bs, embed)
        test : Set to True if you want to calculate and return intermediate
               activation statistics, False to run faster during training.

        Returns
        -------
        Returns inp + residual(inp)
        
        activations : Means and standard deviations of intermediate tensors.
                      Useful for diagnosing instabilities.

        """
        emb = 0 if self.embed==None else self.chce(embinp)[...,None]
        resid1 = torch.relu(torch.einsum('abc,bd->adc', inp, self.W1) + emb)
        resid2 = torch.einsum('abc,bd->adc', resid1, self.W2)
        resid2 = self.drop(resid2)
        
        if test:
            resid1_ = resid1.mean();resid1__ = resid1.std()
            resid2_ = resid2.mean();resid2__ = resid2.std()
            activations = (resid1_, resid1__, resid2_, resid2__)
        else:
            activations = ()
            
        return inp + resid2, activations

class TransBlock(nn.Module):
    def __init__(self,
                 hargs,
                 fargs
                 ):
        """
        Transformer block that implements attention and FFN layers with
        BatchNorm1d following each layer.
        
        :param hargs: Attention layer positional arguments.
        :param fargs: FFN layer positional arguments.
        """
        
        super(TransBlock, self).__init__()
        units1 = hargs[0] if hargs[-1]==None else hargs[-1]
        self.norm1 = nn.BatchNorm1d(units1)
        self.head = TalkingHeads(*hargs)
        self.norm2 = nn.BatchNorm1d(units1)
        self.ffn = FFN(*fargs)
    def forward(self, inp, embed=None, mask=None, test=True):
        """
        Forward call for TransBlock

        Parameters
        ----------
        inp : Input tensor. inp.shape = (bs, in_ch, seq_len)
        embed : Optional embedding tensor for charge and collision energy, fed 
                into FFN layer.
        mask : Optional mask for attention layer.
        test : Optional boolean to return intermediate activations from both
               the attention and ffn layers.

        Returns
        -------
        out : Output following Attention-BatchNorm1d-FFN-BatchNorm1d
        out2+out3 : tuple of intermediate activation statistics.
        FM : Feature/attention maps from attention layer.

        """
        out, out2, FM = self.head(inp, mask, test)
        out = self.norm1(out)
        out, out3 = self.ffn(out, embinp=embed, test=test)
        out = self.norm2(out)
        return out, out2+out3, FM

class FlipyFlopy(nn.Module):
    def __init__(self,
                 in_ch=40,
                 seq_len=40,
                 out_dim=20000,
                 embedsz=256,
                 blocks=9,
                 head=(64,64,4),
                 units=None,
                 drop=0,
                 filtlast=512,
                 mask=False,
                 CEembed=False,
                 CEembed_units=256,
                 pos_type='learned',
                 device='cpu'
                 ):
        """
        Talking heads (Making Flippy Floppy) attention model

        Parameters
        ----------
        in_ch : Number of input channels for input embedding of at least 1)
                amino acid sequence, 2) modifications, 3-optional) charge,
                4-optional) collision energy.
        seq_len : Maximum sequence length for peptide sequence.
        out_dim : Output units for 1d output vector.
        embedsz : Channels for first linear projection, and likely running width
                  of network throughout.
        blocks : Number of TransBlocks to include in depth of network.
        head : Tuple of positional arguments for attention layer.
        units : Channels for first linear projection in FFN layer
        drop : Dropout rate to use in attention and FFN layers
        filtlast : Channels in penultimate layer.
        mask : Optional mask for attention head.
        CEembed : Option for embedding charge and energy with fourier features.
        CEembed_units : Number of fourier features to expand both charge and
                        energy by, before concatenation and feeding to FFN.
        pos_type : Either 'learned' positional embedding, otherwise fourier 
                   feature embedding.
        device : Device to run on.

        """
        super(FlipyFlopy, self).__init__()
        units = embedsz if eval(units)==None else units
        
        self.mask = mask
        self.dev=device
        
        self.embed = nn.Parameter(initEmb((in_ch, embedsz)), requires_grad=True)
        if pos_type=='learned':
            self.pos = nn.Parameter(initPos((embedsz, seq_len)), requires_grad=True)
        else:
            pos = (
                np.arange(seq_len)[:,None] * 
                np.exp(-np.log(1000) * 
                       np.arange(embedsz//2)/(embedsz//2)
                )[None]
            )
            self.pos = nn.Parameter(
                torch.tensor(
                    np.concatenate([np.cos(pos),np.sin(pos)], axis=-1).T[None], 
                    dtype=torch.float32
                ), requires_grad=False
            )
        self.embed_norm = nn.BatchNorm1d(embedsz)
        
        self.CEembed = CEembed
        self.cesz = CEembed_units
        ffnembed = 2*CEembed_units if CEembed else None
        if CEembed:
            self.denseCH = nn.Linear(self.cesz, self.cesz)
            self.denseCE = nn.Linear(self.cesz, self.cesz)
        
        head = (embedsz,)+tuple(head)+(None,None,drop,None)
        self.main = nn.ModuleList([
            TransBlock(head, (embedsz, units, ffnembed, drop)) 
            for _ in range(blocks)]
        )
        
        self.Proj = nn.Parameter(initProj((embedsz, filtlast)), requires_grad=True)
        self.ProjNorm = nn.BatchNorm1d(filtlast)
        self.final = nn.Sequential(nn.Linear(filtlast, out_dim), nn.Sigmoid())
        initFin(self.final[0].weight)
        nn.init.zeros_(list(self.final.parameters())[1])
        
        if verbose:
            print("Embed: %f"%self.embed.std())
            print("Pos: %f"%self.pos.std())
            print("Proj: %f"%self.Proj.std())
            print("Final: %f"%list(self.final.parameters())[0].std())
    
    def embedCE(self, ce, embedsz, freq=10000.):
        """
        Generating fourier features for either charge and energy or positional
        embedding.

        Parameters
        ----------
        ce : Charge, energy, or positional value
        embedsz : Number of features to expand ce by.
        freq: Frequency parameter.

        Returns
        -------
        bs x embedsz fourier feature tensor.

        """
        # ce.shape = (bs,)
        embed = (
            ce[:,None] * 
            torch.exp(
                -np.log(freq) * 
                torch.arange(embedsz//2, device=ce.device)/(embedsz//2)
            )[None]
        )
        return torch.cat([torch.cos(embed),torch.sin(embed)],axis=1)      
    
    def total_params(self, trainable=True):
        """
        Count the total number of model parameters.

        Parameters
        ----------
        trainable : Only count trainable parameters.

        """
        if trainable:
            print(sum([np.prod(m.shape) for m in list(self.parameters()) 
                       if m.requires_grad==True]))
        else:
            print(sum([np.prod(m.shape) for m in list(self.parameters())]))
    def forward(self, inp, test=True):
        """
        Forward call for model

        Parameters
        ----------
        inp : List of input embedding tensors. 
              inp[0].shape = (bs, in_ch, seq_len)
              optional charge vector -> inp[1].shape = (bs,)
              optional energy vector -> inp[2].shape = (bs,)
        test : Set to True if you want to calculate and return all intermediate
               activation tensor statistics throughout depth of network.

        Returns
        -------
        1D mass spectrum intensities for fragments in the dictionary.
        
        lst : Return 'blocks' number of tuples of activation statistics.
        lst2 : Return intermdiate attention maps for plotting (perhaps)

        """
        # inp = (inp, inpch, inpev)
        if self.CEembed:
            # unpack input
            [inp, inpch, inpce] = inp
            
            ch_embed = nn.functional.silu(
                self.denseCH(self.embedCE(inpch, self.cesz, 100))
            )
            ce_embed = nn.functional.silu(
                self.denseCE(self.embedCE(inpce, self.cesz))
            )
            embed = torch.cat([ch_embed,ce_embed],-1)
        else:
            inp = inp[0]
            embed = None
        
        # inp.shape: bs, in_ch, seq_len
        mask = (1e5*inp[:,20] 
                if self.mask else 
                torch.zeros((inp.shape[0], inp.shape[-1]), device=inp.device)
        )
        
        out = torch.einsum('abc,bd->adc', inp, self.embed) # bs, embedsz, seq_len
        out += self.pos
        out = self.embed_norm(out)
        lst = []
        lst2 = []
        for layer in self.main:
            out, out2, FM = layer(out, embed, mask, test)
            lst.append(out2)
            lst2.append(FM)
        out = torch.relu(self.ProjNorm(torch.einsum('abc,bd->adc', out, self.Proj))) # bs, filtlast, seq_len
        #lst.append(out)
        return self.final(out.transpose(-1,-2)).mean(dim=1), lst, lst2
