#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 22:04:30 2023

@author: tyoon
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import islice

from scipy.special import factorial
import random
import string

# Use stilinger algorithm for making the liquid cluster
def boxit(xs,Ls):
    ''' Reducing a vector xs into a periodic box 
    of size Ls '''
    xs, Ls = np.array(xs), np.array(Ls)    
    
    xs = np.mod(xs,Ls)
    for i in range(3):
        xs[xs[:,i]<0,i] += Ls[i]
    
    return xs

def diffvec(xs1, xs2, Ls):
    ''' Difference between vectors xs1 and xs2 
    in a box size of Ls, considering the PBC '''
    xs1 = np.array(xs1)
    xs2 = np.array(xs2)
    Ls = np.array(Ls)
    
    ds = xs2 - xs1
    ds = boxit(ds,Ls)
    for i in range(3):
        ds[ds[:,i] > Ls[i]/2,i] -= Ls[i]
    
    return ds

def StillingerCluster(traj,BoxL,threshold):
    ''' Stillinger criterion cluster for detecting phases'''
    ids = np.arange(traj.shape[0],dtype=int) # number of particles
    G = nx.Graph()    
    for each_atom in ids:        
        d = np.linalg.norm(diffvec(traj[each_atom,:],traj,BoxL),axis=1)
        ind = np.where(np.logical_and(d>0,d<threshold))[0]
        neighbors = ids[ind]        
        for each_neighbor in neighbors:            
            G.add_edge(each_atom,each_neighbor)
    clusters = list(nx.connected_components(G))
    clustersize = np.array([len(cluster) for cluster in clusters])
    return clusters, clustersize

def profile(z,lb,ub,z0,d):
    return 1/2*(lb+ub) - 1/2*(lb-ub)*np.tanh(2*(z-z0)/d)


# First, import the towhee movie file.
ls = [1.25,1.375,1.5,1.7,1.75,1.8,1.9,2,2.25,2.5]
for l in ls:
    addr = 'Your/Simulation/File/Address'
    
    atomnum = 1000
    traj = []
    Ls = []
    with open(addr+'/towhee_movie','r') as infile:
        count = 0
        pos = []
        for line in infile:        
            line = line.split()
            if len(line) == 3:
                line = [float(i) for i in line]
                if len(Ls) < 3:
                    Ls.append(line)
                else:
                    continue
            if len(line) == 5:
                if line[-1].isdigit():
                    line = [float(i) for i in line]
                    line = line[:3]
                    pos.append(line)
            if len(pos) >= 1000:
                traj.append(pos)
                pos = []
    traj = np.array(traj)
    Ls = np.array(Ls)
    np.savez(addr+'/traj.npz',traj=traj,Ls=Ls)

LAMBDAs = [1.5]
for LAMBDA in LAMBDAs:
    addr = 'Your/Simulation/File/Address'
    locals().update(np.load(addr+'/traj.npz'))
    Ls = [Ls[0,0],Ls[1,1],Ls[2,2]]    
    zz = np.linspace(0,Ls[2]/2,100)
    dz = np.diff(zz)[0]
    zbins = (zz[1:]+zz[:-1])/2
    totals = np.zeros((len(zbins)),)
    traj = traj[100:]
    for counter,pos in enumerate(traj):
        clusters,_ = StillingerCluster(pos,Ls,LAMBDA)
        Mcluster = []
        for each in clusters:        
            if len(each) > len(Mcluster):
                Mcluster = list(each)
        center = np.mean(pos[Mcluster,:],axis=0)
        
        for i in range(3):
            pos[:,i] -= center[i]
        
        for i in range(3):
            pos[pos[:,i]<-Ls[i]/2,i] += Ls[i]
        
        for i in range(3):
            pos[pos[:,i]>Ls[i]/2,i] -= Ls[i]
        
        # Now, we assume that the box lengths are from -Ls/2 to Ls/2.
        # Make the z histogram.
        counts,_ = np.histogram(np.abs(pos[:,2]),bins=zz)
        totals += counts
    densities = totals / traj.shape[0] / (dz*Ls[0]**2) / 2
    
    plt.plot(zbins,densities,'o')
    popt,pcov = curve_fit(profile,zbins,densities)
    xx = np.linspace(0,Ls[2]/2,1000)
    yy = profile(xx,*popt)
    plt.plot(xx,yy)
    print(popt)
    plt.show()

# =============================================================================
# 2PMD
# =============================================================================

# Read Argon artoms and make a initial data file for LAMMPS.
addr = 'Your/Simulation/File/Address'
data = np.genfromtxt(addr+'Ar2048.xyz',skip_header=2)
data = data[:,1:]
with open(addr+'data.initial','w') as out:
    out.write('LAMMPS data file\n\n')
    out.write('2048 atoms\n')
    out.write('1 atom types\n\n')
    out.write('-20 20 xlo xhi\n')
    out.write('-20 20 ylo yhi\n')
    out.write('-80 80 zlo zhi\n\n')
    out.write('Masses\n\n1 39.948\n\nAtoms # atomic\n\n')
    for i in range(data.shape[0]):
        out.write('%d 1 %2.6f %2.6f %2.6f\n' % (i+1,data[i,0],data[i,1],data[i,2]))
        

""" # Read Krypton artoms and make a initial data file for LAMMPS. """
addr = 'Your/Simulation/File/Address'
import pandas as pd
data = np.genfromtxt(addr+'Kr2048.xyz',skip_header=2)
data = data[:,1:]
with open(addr+'data.initial','w') as out:
    out.write('LAMMPS data file\n\n')
    out.write('2048 atoms\n')
    out.write('1 atom types\n\n')
    out.write('-21 21 xlo xhi\n')
    out.write('-21 21 ylo yhi\n')
    out.write('-90 90 zlo zhi\n\n')
    out.write('Masses\n\n1 83.798\n\nAtoms # atomic\n\n')
    for i in range(data.shape[0]):
        out.write('%d 1 %2.6f %2.6f %2.6f\n' % (i+1,data[i,0],data[i,1],data[i,2]))
        
# Now, read the lammps trajectory and make the npz file.
def ReadLAMMPS(filename):
    traj = []
    L = []
    with open(filename,'r') as infile:
        for line in infile:
            if 'ITEM: NUMBER OF ATOMS' in line:
                num = int(next(infile))
                break
    with open(filename,'r') as infile:
        for line in infile:
            if 'ITEM: BOX BOUNDS' in line:
                boxs = list(islice(infile,3))
                for boxL in boxs:
                    boxL = boxL.split()
                    boxL = [float(i) for i in boxL]
                    L.append(boxL)
                L = np.array(L)
                break
    with open(filename,'r') as infile:
        for line in infile:
            if 'ITEM: ATOMS id' in line:
                pos = list(islice(infile,num))
                q = []
                for atom in pos:
                    atom = atom.split()
                    atom = [float(i) for i in atom]
                    q.append(atom[:-3])
                q = np.array(q)
                traj.append(q)
    traj = np.array(traj)
    
    return traj, L

material = 'Krypton'
filename = 'Your/Simulation/File/Address/%s/2PMD/dump.lammpstrj' % material
traj, L = ReadLAMMPS(filename)
filename = 'Your/Simulation/File/Address/%s/2PMD/traj.npz' % material
np.savez(filename,traj=traj,L=L)

mw = 83.798
Na = 6.02214e23
material = 'Krypton'
filename = 'Your/Simulation/File/Address/%s/2PMD/traj.npz' % material
locals().update(np.load(filename))
Ls = np.array([L[0,1],L[1,1],L[2,1]])
zz = np.linspace(0,Ls[2],100)
dz = np.diff(zz)[0]
zbins = (zz[1:]+zz[:-1])/2
totals = np.zeros((len(zbins)),)
traj = traj[100:,:,1:]
for counter,pos in enumerate(traj):
    pos = boxit(pos,Ls*2)       
    pos[:,2] = pos[:,2] - Ls[2]
    # Now, we assume that the box lengths are from -Ls/2 to Ls/2.
    # Make the z histogram.
    counts,_ = np.histogram(np.abs(pos[:,2]),bins=zz)
    totals += counts
densities = totals / traj.shape[0] / (dz*(2*Ls[0])**2) / 2
densities = densities * mw * 1e24 / Na # g/cm3
plt.plot(zbins,densities,'o')
popt,pcov = curve_fit(profile,zbins,densities)
xx = np.linspace(0,Ls[2],1000)
yy = profile(xx,*popt)
plt.plot(xx,yy)
print(popt)
plt.show()