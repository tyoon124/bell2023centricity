#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:46:41 2023

@author: tyoon
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

def r_squared(y, y_hat):
    y_bar = y.mean()
    ss_tot = ((y-y_bar)**2).sum()
    ss_res = ((y-y_hat)**2).sum()
    return 1 - (ss_res/ss_tot)


cm = 1/2.54
color = sns.color_palette()
fig, ax = plt.subplots(1,2,figsize=(18*cm,6*cm),dpi=1000,gridspec_kw={'wspace':0.7})
mws = [4.002602, 20.1797, 39.948, 83.798, 131.293]
ws = [-0.390,0.000,0.000,0.000,0.000]
scrits = [0.3174,0.9224,0.9948,0.9728,0.9840]
ax[0].annotate('He',(mws[0],scrits[0]),xytext=(mws[0],scrits[0]-0.1),fontsize=8)
ax[0].annotate('Ne',(mws[1],scrits[1]),xytext=(mws[1],scrits[1]-0.1),fontsize=8)
ax[0].annotate('Ar',(mws[2],scrits[2]),xytext=(mws[2],scrits[2]-0.1),fontsize=8)
ax[0].annotate('Kr',(mws[3],scrits[3]),xytext=(mws[3],scrits[3]-0.1),fontsize=8)
ax[0].annotate('Xe',(mws[4],scrits[4]),xytext=(mws[4],scrits[4]-0.1),fontsize=8)
l1, = ax[0].plot(mws,scrits,'o',color=color[1],markerfacecolor=color[1])
ax[0].set_ylim([0,1.1])
ax2 = ax[0].twinx()
l2, = ax2.plot(mws,ws,'s',color='k',markerfacecolor='None',markeredgewidth=0.75)
ax2.set_ylim([-0.6,0.058])
ax[0].hlines(1.0,0,1000,linestyle='--',linewidth=0.75,color=color[0])
ax[0].set_xlim([0,150])
ax[0].set_xlabel('M [g/mol]')
ax[0].set_ylabel(r'$s_\mathrm{c}^+\equiv-s_\mathrm{res,c}/k_\mathrm{B}$')
ax2.set_ylabel(r'$\omega$')
ax[0].text(-0.2,1,'A',
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax[0].transAxes,
        weight='bold')
plt.legend([l1,l2],["$s_\mathrm{c}^+$","$\omega$"],frameon=False,
           loc='lower right')

# CoolProp
from CoolProp.CoolProp import PropsSI
materials = ['1-Butene',
             'Acetone',
             'Ammonia',
             'Argon',
             'Benzene',
             'CarbonDioxide',
             'CarbonMonoxide',
             'CarbonylSulfide',
             'CycloHexane',
             'CycloPropane',
             'Cyclopentane',
             'D4',
             'D5',
             'D6',
             'Deuterium',
             'Dichloroethane',
             'DiethylEther',
             'DimethylCarbonate',
             'DimethylEther',
             'Ethane',
             'Ethanol',
             'EthylBenzene',
             'Ethylene',
             'EthyleneOxide',
             'Fluorine',
             'HFE143m',
             'HeavyWater',
             'Helium',
             'Hydrogen',
             'HydrogenChloride',
             'HydrogenSulfide',
             'IsoButane',
             'IsoButene',
             'Isohexane',
             'Isopentane',
             'Krypton',
             'MD2M',
             'MD3M',
             'MD4M',
             'MDM',
             'MM',
             'Methane',
             'Methanol',
             'MethylLinoleate',
             'MethylLinolenate',
             'MethylOleate',
             'MethylPalmitate',
             'MethylStearate',
             'Neon',
             'Neopentane',
             'Nitrogen',
             'NitrousOxide',
             'Novec649',
             'OrthoDeuterium',
             'OrthoHydrogen',
             'Oxygen',
             'ParaDeuterium',
             'ParaHydrogen',
             'Propylene',
             'Propyne',
             'R11',
             'R113',
             'R114',
             'R115',
             'R116',
             'R12',
             'R123',
             'R1233zd(E)',
             'R1234yf',
             'R1234ze(E)',
             'R1234ze(Z)',
             'R124',
             'R1243zf',
             'R125',
             'R13',
             'R134a',
             'R13I1',
             'R14',
             'R141b',
             'R142b',
             'R143a',
             'R152A',
             'R161',
             'R21',
             'R218',
             'R22',
             'R227EA',
             'R23',
             'R236EA',
             'R236FA',
             'R245ca',
             'R245fa',
             'R32',
             'R365MFC',
             'R40',
             'R404A',
             'R407C',
             'R41',
             'R410A',
             'R507A',
             'RC318',
             'SES36',
             'SulfurDioxide',
             'SulfurHexafluoride',
             'Toluene',
             'Water',
             'Xenon',
             'cis-2-Butene',
             'm-Xylene',
             'n-Butane',
             'n-Decane',
             'n-Dodecane',
             'n-Heptane',
             'n-Hexane',
             'n-Nonane',
             'n-Octane',
             'n-Pentane',
             'n-Propane',
             'n-Undecane',
             'o-Xylene',
             'p-Xylene',
             'trans-2-Butene'
             ]
Tcs = []
Pcs = []
Dcs = []
omegas = []
scrits = []
Bvirials = []
for material in materials:
    try:
        material = material.strip()
        Tcrit = PropsSI('Tcrit',material)
        Pcrit = PropsSI('Pcrit',material)
        Dcrit = PropsSI('RHOCRIT',material)        
        scrits.append(-PropsSI('SMOLAR_RESIDUAL','T',Tcrit,'P',Pcrit,material)/8.314)
        Tcs.append(Tcrit)
        Pcs.append(Pcrit)
        Dcs.append(Dcrit)
        omegas.append(PropsSI('ACENTRIC',material))
        Bvirials.append(PropsSI('BVIRIAL','T',Tcrit,'P',Pcrit,material))
    except ValueError:
        continue

ax[1].plot(omegas,scrits,'o',markersize=4,markeredgecolor=color[0],markerfacecolor='w',markeredgewidth=0.5)
ax[1].set_xlim([-0.5,1.25])
ax[1].set_ylim([0,3.25])
p,cov = np.polyfit(omegas,scrits,1,cov=True)
xx = np.linspace(np.min(omegas),np.max(omegas),100)
yy = np.polyval(p,xx)
ax[1].plot(xx,yy,color=color[1],linewidth=0.5)
ax[1].text(-0.2,1,'B',
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax[1].transAxes,
        weight='bold')
ax[1].set_xlabel(r'$\omega$')
ax[1].set_ylabel(r'$s_\mathrm{c}^+$')

yhats = np.polyval(p,omegas)
print(r_squared(np.array(scrits),yhats))
ax[1].text(0.65,0.25,'$R^2=$'+str(np.round(r_squared(np.array(scrits),yhats),2)))
