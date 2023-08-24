#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:35:20 2023

@author: tyoon
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


''' Example 1: Critical Temperature Estimation (Square-well fluids) '''
data = pd.read_excel('EstimateCriticality.ods')

cm = 1/2.54
fig, ax = plt.subplots(3,2,dpi=150,figsize=(16*cm,16*cm),gridspec_kw={'hspace':0.45,'wspace':0.4})
ax = ax.flatten()

for counter, L in enumerate([1.25,1.375,1.5,1.75,2,2.5]):
    data2 = data[np.abs(data['Well Width']-L)<1e-5]
    temperatures = np.unique(data2['Temperature'].to_numpy())
    slopes = []
    for T in temperatures:        
        D = data2['Density'][np.abs(data2['Temperature']-T)<1e-5].to_numpy()
        P = data2['Pressure'][np.abs(data2['Temperature']-T)<1e-5].to_numpy()
        D = D[np.argsort(D)]
        P = P[np.argsort(D)]    
        par, var = np.polyfit(D,P,3,cov=True)
        uncertainty = np.sqrt(np.diag(var))
        xinflect = -par[1]/(3*par[0])
        xinflectU = xinflect*np.sqrt((uncertainty[1]/par[1])**2 + (uncertainty[0]/par[0])**2)
        slope = 3*par[0]*xinflect**2 + 2*par[1]*xinflect + par[2]
        slopes.append(slope)        
    ax[counter].plot(slopes,temperatures,'o',markersize=5,color='k',markerfacecolor='None',markeredgewidth=0.75)    
    ax[counter].set_ylim([np.round(np.min(temperatures)*0.8,1),np.round(np.max(temperatures)*1.15,1)])
    ylims = ax[counter].get_ylim()
    #ax[counter].set_yticks(np.arange(ylims[0],ylims[1],1))
    p,cov = np.polyfit(slopes,temperatures,1,cov=True)
    U = np.sqrt(np.diag(cov))
    ax[counter].plot(slopes,np.polyval(p,slopes),color='r',linewidth=0.75)
    ax[counter].text(0.95,0.05,r'$T_\mathrm{crit}=%.3f\pm%.3f$' % (p[1],U[1]),
                     horizontalalignment='right',fontsize=8,
                     transform=ax[counter].transAxes)
    ax[counter].set_xlabel(r'$(\partial p/\partial\rho)_\mathrm{T}$')
    ax[counter].set_ylabel(r'$T$')

plt.show()

''' Example 2: Critical Density Estimation (Argon, using the pressrun data from LAMMPS simulations) '''
drs = np.arange(0.1,2.1,.1)
Trs = [1.00]
Dcexpinkgm3 = 536 # kg/m3
mw = 39.948 # g/mol
Na = 6.02214076e23
num = 2048
Dcexp = Dcexpinkgm3 * 1000 / 39.948 * Na / 1e30 # particles per angstrom
Tc_exp = 158.597

fig, ax = plt.subplots(dpi=150,figsize=(8*cm,8*cm),gridspec_kw={'hspace':0.45,'wspace':0.4})

for Tr in Trs:
    pressures = []
    for dr in drs:
        pp = []
        for setnum in range(1,4):
            filename = 'argon-Example/set'+str(setnum)+'/dr%.2f/pressrun.out' % (dr)
            with open(filename,'r') as infile:
                for line in infile:
                    continue
            pp.append(float(line.split()[-1]))
        pressures.append(np.mean(pp))
    pressures = np.array(pressures)
    ax.plot(drs*Dcexpinkgm3,pressures,'o',color='k',markersize=5,markerfacecolor='w',markeredgewidth=0.75)
    parm = np.polyfit(drs[np.logical_and(drs>0.6,drs<1.4)],pressures[np.logical_and(drs>0.6,drs<1.4)],3)
    yy = np.polyval(parm,np.linspace(0.1,2.1,1000))
    rhocrit = -parm[1]/(3*parm[0])
    ax.text(0.95,0.05,r'$\rho_\mathrm{crit}=%.2f$ kg/m$^3$' % float(rhocrit*Dcexpinkgm3),
            horizontalalignment='right',
            transform=ax.transAxes)
    ax.plot(rhocrit*Dcexpinkgm3,np.polyval(parm,rhocrit),'o',color='r',markersize=5,label='Critical Point')
ax.legend(frameon=False)
