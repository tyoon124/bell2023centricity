#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:50:27 2023

@author: tyoon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.integrate import trapz
import seaborn as sns

def calcs2pls(r,gr,rho,rc=np.inf):
    ''' Calculate the two-body excess entropy '''
    gr = gr[r<=rc]
    r = r[r<=rc]
    I1 = gr*np.log(gr)
    I2 = gr-1
    I1[gr==0]=0
    I = I1 - I2
    s2pls = trapz(I*2*np.pi*rho*r**2,x=r)
    return s2pls

import warnings
warnings.filterwarnings("ignore")

# s2 calculation.
# Since there is a discontinuity in the discrete potential result,
# we need to extrapolate the rdf to the discontinuity.
# Here, only the rdf data for L=2.5 case is given.
for L in [2.5]:
    ds = range(5,65,5)
    for d in ds:
        rs = []
        delrs = []
        rr = []
        grs = []
        for setnum in [1,2,3]:
            addr = 'grdata/set%d/L%.3f/d%d' % (setnum,L,d)
            filename2 = addr + '/gr.npz'
            locals().update(np.load(filename2))
            rr.append(r)
            grs.append(gr)
        rr = np.mean(np.array(rr),axis=0)
        grs = np.mean(np.array(grs),axis=0)
        
        rr = rr - rr[grs==np.max(grs)] + 1
        
        grs1 = grs[np.logical_and(rr>=1,rr<L)]
        rr1 = rr[np.logical_and(rr>=1,rr<L)]
        
        grs2 = grs[rr>=L]
        rr2 = rr[rr>=L]
        #grs1F = savgol_filter(grs1,51,3)
        
        window_length = int(len(rr1)*0.3)        
        if window_length % 2 == 0:
            window_length += 1
            
        rrgrs1F = savgol_filter(rr1*grs1,window_length,2) # L=1.25
        
        #window_length = int(len(rr2)*0.06)
        window_length = int(10)
        if window_length % 2 == 0:
            window_length += 1
            
        rrgrs2F = savgol_filter(rr2*grs2,window_length,2)
        
        s1 = UnivariateSpline(rr1,rrgrs1F,s=0)
        s2 = UnivariateSpline(rr2,rrgrs2F,s=0)
        
        rr0 = np.linspace(0,1,1000)
        rr1 = np.linspace(1,L,50000)
        rr2 = np.linspace(L,np.max(rr),50000)
        
        rrgrs0F = np.zeros((len(rr0),))
        rrgrs1F = s1(rr1)
        rrgrs2F = s2(rr2)
        
        rrgrsF = np.concatenate((rrgrs0F,rrgrs1F,rrgrs2F))
        rrF = np.concatenate((rr0,rr1,rr2))
        
        grsF = rrgrsF/rrF
        grsF[0] = 0
        plt.plot(rrF,grsF)
        s2 = calcs2pls(rrF,grsF,rho=d/100)
        ss2 = calcs2pls(rr,grs,rho=d/100)
        print(s2)
        plt.xlim([0,4])
    plt.ylim([-0.02,4])
    plt.show()
