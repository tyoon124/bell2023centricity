#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:42:30 2023

@author: tyoon
"""

import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import string

''' The user needs to install the cavity-vol-PBC program to use this code. '''

class Widom:
    
    def __init__(self,traj,L,cutoff,trials,
                 temperature,
                 model = 'HS',
                 LAMBDA = 0.0):
        ''' Insert hard test particle(s) "trials" times to the given trajectory data.
        Calculate the exclusion probability.'''
        
        self.kb = 1.0
        
        # Move the box to satisfy the PBC.        
        self.temperature = temperature
        self.kbT = self.kb*self.temperature
        self.LAMBDA = LAMBDA
        if traj.shape[1] > 3:
            self.traj = traj[:,1:4]
        self.traj = self.boxit(self.traj,L)
        
        self.results = np.zeros((trials,))
        for i in range(trials):
            pos = np.random.random(3)*L
            d = np.linalg.norm(diffvec(pos,self.traj,L),axis=1)
            poteng = np.zeros((len(d),))                
            poteng[d<1-1.0e-10] = np.inf
            if model == 'SW':
                poteng[d>=1.0-1.0e-10] = -1.0
                poteng[d>=LAMBDA] = 0.0
            poteng = np.sum(poteng)
            result = np.exp(-poteng/self.kbT)
            self.results[i] = result        
            
    def boxit(self,xs,Ls):
        ''' Reducing a vector xs into a periodic box 
        of size Ls '''
        xs, Ls = np.array(xs), np.array(Ls)
        xs = np.mod(xs,Ls)
        xs[xs < 0] += Ls
        
        return xs

    def diffvec(self, xs1, xs2, Ls):
        ''' Difference between vectors xs1 and xs2 
        in a box size of Ls, considering the PBC '''
        xs1 = np.array(xs1)
        xs2 = np.array(xs2)
        Ls = np.array(Ls)
        
        ds = xs2 - xs1
        ds = self.boxit(ds,Ls)
        ds[ds > Ls/2] = ds[ds > Ls/2] - Ls
        
        return ds

def boxit(xs,Ls):
    ''' Reducing a vector xs into a periodic box 
    of size Ls '''
    xs, Ls = np.array(xs), np.array(Ls)
    xs = np.mod(xs,Ls)
    xs[xs < 0] += Ls
    
    return xs

def diffvec(xs1, xs2, Ls):
    ''' Difference between vectors xs1 and xs2 
    in a box size of Ls, considering the PBC '''
    xs1 = np.array(xs1)
    xs2 = np.array(xs2)
    Ls = np.array(Ls)
    
    ds = xs2 - xs1
    ds = boxit(ds,Ls)
    ds[ds > Ls/2] = ds[ds > Ls/2] - Ls
    
    return ds

def loadXMLFile(filename):
    #Check if the file is compressed or not, and 
    if (os.path.splitext(filename)[1][1:].strip() == "bz2"):
        import bz2
        f = bz2.BZ2File(filename)
        doc = ET.parse(f)
        f.close()
        return doc
    else:
        return ET.parse(filename)
    
    
def mureshs(density,diameter=1.0):
    eta = np.pi/6*density*diameter**3
    return (8*eta - 9*eta**2 +3*eta**3)/(1-eta)**3

def sresHS(density,diameter=1.0):
    ''' Calculate splus of hard-sphere model.'''      
    eta = density*np.pi/6*diameter**3
    splus = eta*(4-3*eta)/(1-eta)**2
    return -splus

def pHS(density,temperature,diameter=1.0):
    ''' Calculate the pressure of hard-sphere model given the density and temp'''
    kB = 1    
    eta = density*np.pi/6*diameter**3
    p = kB*temperature*density * (1 + eta + eta**2 - eta**3) / (1-eta)**3
    return p

def pCS(density,temperature,diameter=1.0):
    ''' Calculate the pressure of hard-sphere model given the density and temp'''
    kB = 1    
    eta = density*np.pi/6*diameter**3
    p = kB*temperature*density * (1 + eta + eta**2 - eta**3) / (1-eta)**3
    return p

def vresHS(density,temperature,diameter=1.0):
    ''' Calculate the residual volume '''
    volume = density**(-1)
    p = pCS(density,temperature,diameter=diameter)
    kB = 1
    vid = kB*temperature/p
    vres = volume - vid
    return vres

def PinsHS(density):
    ''' Calculate the insertion probability in hard-sphere systems '''
    diameter = 1
    eta = (np.pi/6)*density*diameter**3
    mur = (8*eta-9*eta**2+3*eta**3)/(1-eta)**3
    Pins = np.exp(-mur)
    return Pins

def zHS(density,diameter=1.0):
    eta = (np.pi/6)*density*diameter**3
    Zhs = (1+eta+eta**2-eta**3)/(1-eta)**3
    return Zhs


#A helpful function to load compressed or uncompressed XML files
   
def ReaddynamOHS(ntraj,nmol,addr,option=True):
    trajectories = np.zeros((ntraj+1,nmol,4))
    for i in range(ntraj+1):        
        filename = addr+'Production/config'+str(i)+'.xml'
        XMLDoc = loadXMLFile(filename)
        # Box length
        BoxL = XMLDoc.find('.//Simulation')
        BoxL = BoxL.find('SimulationSize')
        BoxLs = [float(BoxL.get("x")),
                 float(BoxL.get("y")),
                 float(BoxL.get("z"))]
        PtTags = XMLDoc.findall(".//Pt")
        traj = np.zeros((nmol,4))        
        for counter,PtElement in enumerate(PtTags):
            PosTag = PtElement.find('P')
            pos = [float(PosTag.get("x")),
                   float(PosTag.get("y")),
                   float(PosTag.get("z"))]
            traj[counter,1:] = pos
            
        traj = np.array(traj,dtype=float)
        traj[:,0] = range(1,traj.shape[0]+1)
        ''' Periodic boundary condition '''
        trajectories[i,:,:] = traj  
        
    np.savez(addr+'traj.npz',trajectories=trajectories,
             BoxL=BoxLs)
    #if option:
    #    os.system('rm -rf '+addr+'Production')
    
    return trajectories, BoxLs

""" Implement the Voronoi-Widom method """

class CavityVol:
    
    ''' wrapper for Cavity-Volume-PBC '''
    
    def __init__(self,traj,L,radii=None,Probe_radius=0.5):
        self.traj = traj        
        self.L = L
        if radii is None:
            self.radii = np.ones((traj.shape[0]))
        else:
            self.radii = radii
        self.Probe_radius = Probe_radius
        self.basisdir = '/home/tyoon/cavity-volumes-pbc/simulations/'
        self.basisdir += ''.join(random.choices(string.ascii_lowercase, k=10))+'/'
        os.makedirs(self.basisdir)
        self.WrapPBC()
        self.Writetraj()
        self.Writecontrol()
        self.Run()
        self.Read()
        os.system('rm -rf '+self.basisdir)
    
    def boxit(self,xs,Ls):
        ''' Reducing a vector xs into a periodic box 
        of size Ls '''
        xs, Ls = np.array(xs), np.array(Ls)
        xs = np.mod(xs,Ls)
        xs[xs < 0] += Ls
        
        return xs
        
    def WrapPBC(self):
        ''' Wrap coordinates '''
        for i in [1,2,3]:
            self.traj[:,i] = self.boxit(self.traj[:,i],self.L)
        
    def Writetraj(self):
        header = '#testing system\n'+str(self.traj.shape[0])                
        with open(self.basisdir+'data.gro','w') as out:
            out.write('test particle\n')
            out.write('{0:5d}\n'.format(self.traj.shape[0]))        
            for num in range(self.traj.shape[0]):
                out.write('{:>5d}{:<5.5s}{:>5.5s}'.format(num,'a','unit'))
                out.write('{:>5d}{:8.3f}{:8.3f}{:8.3f}\n'.format(num,self.traj[num,1],
                                                                 self.traj[num,2],self.traj[num,3]))
            out.write("{:10.5f} {:9.5f} {:9.5f}\n".format(self.L,self.L,self.L))
                
        np.savetxt(self.basisdir+'data.rad',self.radii,fmt='%2.5f',header=header,comments='')
    
    def Writecontrol(self):
        filename = self.basisdir+'input.inp'
        with open(filename,'w') as infile:
            infile.write('#############################Cavity Volume PBC############################\n')
            infile.write('data.gro - input trajectory file with coordinates\n')
            infile.write('1 1 1 - first /last /step - cycle over configurations\n')
            infile.write('data.rad - file with atomic radii\n')
            infile.write('##########################################################################\n')
            infile.write('0 - trajecory file format selector (0: Gromacs xtc, gro, g96, 1: text)\n')
            infile.write('0.5  - box thickening shell width [0, 1] for pbc triangulation\n')
            infile.write('##########################################################################\n')
            infile.write('##########################################################################\n')
            infile.write('0.5 - input radii scale factor (final R = scale * Ri + Rprobe)\n')
            infile.write(str(self.Probe_radius)+' - probe radius\n')
            infile.write('##########################################################################\n')
            infile.write('1cavout_info.dat - information output file (! - do not write)\n')
            infile.write('0 - be verbose: print information to the screen (0/1)\n')
            infile.write('##########################################################################\n')
            infile.write('!2cavout_asurf.dat - contributions of atoms to voids surface\n')
            infile.write('!3cavout_volumes.dat - all encountered void volumes for distribution cal\n')
            infile.write('!4cavout_evfract.dat - empty volume fraction in each conf (totV/boxV)\n')
            infile.write('##########################################################################\n')
            infile.write('!5cavout_3dmesh.stl - file for writing 3D graphics mesh of cavities\n')
            infile.write('2 - 3D surface subdivision parameter')
        
    def Run(self):
        os.system('cd '+self.basisdir+' && '+
                  '/home/tyoon/cavity-volumes-pbc/build/cavity_volumes_pbc input.inp')
    
    def Read(self):
        with open(self.basisdir+'1cavout_info.dat','r') as infile:
            for line in infile:
                if 'box volume' in line:
                    self.vol = float(line.split()[-1])
                if 'voids total volume' in line:
                    self.void = float(line.split()[-1])
        self.exclude = self.vol-self.void
       
class Bieshaar:
    
    def __init__(self,traj,L,cutoff,trials,
                 temperature,
                 model = 'HS',
                 LAMBDA = 0.0):
        ''' Insert hard test particle(s) "trials" times to the given trajectory data.
        Calculate the exclusion probability.
        Do the Voronoi tessellation and free volume calculation for boosting the calculation'''
        
        self.kb = 1.0
        
        # Move the box to satisfy the PBC.        
        self.temperature = temperature
        self.kbT = self.kb*self.temperature
        self.LAMBDA = LAMBDA
        if traj.shape[1] > 3:
            self.traj = traj[:,1:4]
        self.traj = self.boxit(self.traj,L)
        
        # First, use the cavity volume calculation for calculating the insertion probability.
        Cavity = CavityVol(traj,L,radii=None,Probe_radius=0.5)
        self.Pins = 1 - Cavity.exclude/Cavity.vol
        
        # Then, do the Voronoi tessellation for obtaining the insertion points.
        self.basisdir = '/home/tyoon/voro++-0.4.6/temporary/'
        self.basisdir += ''.join(random.choices(string.ascii_lowercase, k=10))+'/'
        os.makedirs(self.basisdir)
        filename = self.basisdir+'pre'
        with open(filename,'w') as outfile:
            for i in range(traj.shape[0]):
                outfile.write('%d %.5f %.5f %.5f\n' % (i+1,traj[i,0],traj[i,1],traj[i,2]))
        os.system('cd '+self.basisdir+' && voro++ -c %P -p -o 0 '+str(L)+' 0 '+str(L)+' 0 '+str(L)+' '+filename)
        with open(filename+'.vol','r') as infile:
            neighbors = []
            for line in infile:
                line = line.split()
                for each in line:
                    each = each.replace('(','')
                    each = each.replace(')','')
                    each = each.split(',')
                    each = [float(i) for i in each]
                    neighbors.append(each)
        os.system('cd '+self.basisdir+' && voro++ -c %P -p -o 0 '+str(L)+' 0 '+str(L)+' 0 '+str(L)+' '+filename) # This is for calculating the distance.
        with open(filename+'.vol','r') as infile:
            distances= []
            for line in infile:
                line = line.split()                                
                for each in line:
                    each = each.replace('(','')
                    each = each.replace(')','')
                    each = each.split(',')
                    each = [float(i) for i in each]
                    distances.append(np.linalg.norm(each))
        os.system('rm -rf '+self.basisdir)
        # Now, remove all vertices that are too close
        vertices = np.array(neighbors)
        distances = np.array(distances)
        vertices = vertices[distances>=1]
                
        np.random.shuffle(vertices)
        vertices = vertices[:trials,:]
        self.results = np.zeros((trials,))
        for i in range(vertices.shape[0]):
            pos = vertices[i,:]
            d = np.linalg.norm(diffvec(pos,self.traj,L),axis=1)
            poteng = np.zeros((len(d),))                
            if any(d<1-1.0e-10):
                continue
            else:
                if model == 'SW':
                    poteng[d>=1.0-1.0e-10] = -1.0
                    poteng[d>=LAMBDA] = 0.0
                poteng = np.sum(poteng)
                result = np.exp(-poteng/self.kbT)
                self.results[i] = result        
            
    def boxit(self,xs,Ls):
        ''' Reducing a vector xs into a periodic box 
        of size Ls '''
        xs, Ls = np.array(xs), np.array(Ls)
        xs = np.mod(xs,Ls)
        xs[xs < 0] += Ls
        
        return xs

    def diffvec(self, xs1, xs2, Ls):
        ''' Difference between vectors xs1 and xs2 
        in a box size of Ls, considering the PBC '''
        xs1 = np.array(xs1)
        xs2 = np.array(xs2)
        Ls = np.array(Ls)
        
        ds = xs2 - xs1
        ds = self.boxit(ds,Ls)
        ds[ds > Ls/2] = ds[ds > Ls/2] - Ls
        
        return ds