#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 23:13:03 2023

@author: tyoon
"""

import numpy as np

module = __import__('3Sastry-Bieshaar')

# We first load the trajectory data.
locals().update(np.load('HS-d40-trajectory.npz'))

# Then, use the Sastry algorithm to calculate the probe-accessible volume.
L = BoxL[0]

Pinss = []
for traj in trajectories:
    result = module.CavityVol(traj,L)
    Pinss.append(result.void/result.vol)
    print('Probe-accessible volume fraction:',result.void/result.vol)

print('Dimensionless excess chemical potential from Sastry algorithm:',-np.log(np.mean(Pinss)))

# For the comparison, we use the Carnahan-Starling Equation of States to calculate the excess chemical potential.
density = 0.40
mureshs = module.mureshs(density,diameter=1.0)
print('Dimensionless excess chemical potential from Carnahan-Starling EoS:',mureshs)



