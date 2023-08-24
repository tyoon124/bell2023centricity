#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:18:44 2023

@author: tyoon
"""

import numpy as np, scipy.integrate
from scipy.special import factorial
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas
import os
from scipy.optimize import brentq
import string
import random
from itertools import islice

k_B = 1.380649e-23 # [J/K]
hbar = 1.054571817e-34 # [J s]
u = 1.66053906660e-27 # [kg]
N_A = 8.314462618/k_B # [J/(mol*K)]

# Convert the trajectory data into numpy array.
def ReadLAMMPS(filename):
    traj = []
    with open(filename,'r') as infile:
        for line in infile:
            if 'ITEM: NUMBER OF ATOMS' in line:
                num = int(next(infile))
                break
    with open(filename,'r') as infile:
        for line in infile:
            if 'ITEM: BOX BOUNDS' in line:
                L = next(infile).split()
                L = [float(i) for i in L]
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

class CavityVol:
    
    ''' wrapper for Cavity-Volume-PBC '''
    
    def __init__(self,traj,L,radii=None,Probe_radius=0.5):
        self.traj = traj        
        self.L = L
        if radii is None:
            self.radii = np.ones((traj.shape[0]))
        else:
            self.radii = radii*np.ones((traj.shape[0]))
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
            d = np.linalg.norm(self.diffvec(pos,self.traj,L),axis=1)
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
    
def exp(x):
    """
    if isinstance(x, np.ndarray) and len(x) > 0 and isinstance(x[0], pymcx.MultiComplex):
        return np.array([x_.exp() for x_ in x])
    else:
        return np.exp(x)
    """
    return np.exp(np.float128(x))

class TangToennies:
    def __init__(self, **params):
        self.__dict__.update(**params)

    def add_recursive(self):
        """ 
        Add the C values by the recurrence relation if they are not provided 
        """
        for n in [6, 7, 8]:
            self.C[2*n] = self.C[2*n-6]*(self.C[2*n-2]/self.C[2*n-4])**3

    def potTT(self, R):
        """
        Return the Tang-Toennies potential V/kB in K as a function of R in nm
        """
        out = self.A*exp(self.a1*R + self.a2*R**2 + self.an1/R + self.an2/R**2)
        bR = self.b*R
        contribs = []
        for n in range(3, self.nmax+1):
            bracket = 1-exp(-bR)*sum(
                [bR**k/factorial(k) for k in range(0, 2*n+1)])
            contribs.append(self.C[2*n]*bracket/R**(2*n))
        out -= sum(contribs)
        return out

    def pot(self, R):
        """
        Return the potential V/kB in K as a function of R in nm

        Also apply the correction function at small separations
        """
        R = np.array(R, ndmin=1) # to array
        out = self.potTT(R)
        mask = R < self.Rcutoff*self.Repsilon
        out[mask] = self.tildeA/R[mask]*exp(-self.tildea*R[mask])
        return out

    def potprimeTT(self, R):
        """
        Return the derivative of the potential V/kB in K with respect to 
        position as a function of R in nm 
        """
        R = np.array(R, ndmin=1) # to array
        v = self.a1*R+self.a2*R**2+self.an1/R+self.an2/R**2
        vprime = (self.a1+2*self.a2*R-self.an1/R**2-2*self.an2/R**3)
        out = self.A*exp(v)*vprime
        summer = 0
        for n in range(3, self.nmax+1):
            bsum = sum([(self.b*R)**k/factorial(k) for k in range(0, 2*n+1)])
            bsumprime = sum(
                [self.b**k*k*R**(k-1)/factorial(k) for k in range(0, 2*n+1)])
            b = 1-exp(-self.b*R)*bsum
            bprime = -exp(-self.b*R)*bsumprime +self.b*exp(-self.b*R)*bsum
            summer += -2*n*self.C[2*n]/R**(2*n+1)*b + self.C[2*n]/R**(2*n)*bprime
        out -= summer
        return out

    def potprime(self, R):
        """
        Return the derivative of the potential V/kB in K with respect to 
        position as a function of R in nm 
        """
        R = np.array(R, ndmin=1) # to array
        out = self.potprimeTT(R)
        mask = R < self.Rcutoff*self.Repsilon
        out[mask] = -self.tildeA*exp(-self.tildea*R[mask])*(
            1/R[mask]**2 + self.tildea/R[mask]
            )
        return out

    def potprime2TT(self, R):
        """
        Return the second derivative of the potential V/kB in K with respect to 
        position as a function of R in nm 
        """
        R = np.array(R, ndmin=1) # to array
        v = self.a1*R+self.a2*R**2+self.an1/R+self.an2/R**2
        vprime = self.a1+2*self.a2*R-self.an1/R**2-2*self.an2/R**3
        vprime2 = 2*self.a2+2*self.an1/R**3+6*self.an2/R**4
        out = self.A*exp(v)*(vprime2 + vprime**2)
        summer = 0
        for n in range(3, self.nmax+1):
            bsum = sum([(self.b*R)**k/factorial(k) for k in range(0, 2*n+1)])
            bsumprime = sum(
                [self.b**k*k*R**(k-1)/factorial(k) for k in range(0, 2*n+1)])
            bsumprime2 = sum(
                [self.b**k*k*(k-1)*R**(k-2)/factorial(k) for k in range(0, 2*n+1)])
            b = 1-exp(-self.b*R)*bsum
            bprime = -exp(-self.b*R)*bsumprime +self.b*exp(-self.b*R)*bsum
            bprime2 = (-exp(-self.b*R)*bsumprime2 
                +2*self.b*exp(-self.b*R)*bsumprime -self.b**2*exp(-self.b*R)*bsum)
            summer += (-4*n*self.C[2*n]/R**(2*n+1)*bprime 
                       +(2*n)*(2*n+1)*self.C[2*n]/R**(2*n+2)*b 
                       +self.C[2*n]/R**(2*n)*bprime2)
        out -= summer
        return out

    def potprime2(self, R):
        """
        Return the second derivative of the potential V/kB in K with respect 
        to position as a function of R in nm 

        Also includes the small separation correction
        """
        R = np.array(R, ndmin=1) # to array
        out = self.potprime2TT(R)
        mask = R < self.Rcutoff*self.Repsilon
        Rm = R[mask]
        out[mask] = self.tildeA*exp(-self.tildea*Rm)*(
            2/Rm**3 + 2*self.tildea/Rm**2 + (Rm*self.tildea)**2/Rm**3)
        return out

    def potprime3TT(self, R):
        """
        Return the third derivative of the Tang-Toennies potential V/kB in K 
        with respect to position as a function of R in nm 
        """
        R = np.array(R, ndmin=1) # to array
        v = self.a1*R+self.a2*R**2+self.an1/R+self.an2/R**2
        vprime = self.a1+2*self.a2*R-self.an1/R**2-2*self.an2/R**3
        vprime2 = 2*self.a2+2*self.an1/R**3+6*self.an2/R**4
        vprime3 = -6*self.an1/R**4-24*self.an2/R**5
        out = self.A*exp(v)*(vprime3 + 3*vprime*vprime2 + vprime**3)
        summer = 0
        for n in range(3, self.nmax+1):
            bsum = sum([(self.b*R)**k/factorial(k) for k in range(0, 2*n+1)])
            bsumprime = sum(
                [self.b**k*k*R**(k-1)/factorial(k) for k in range(0, 2*n+1)])
            bsumprime2 = sum(
                [self.b**k*k*(k-1)*R**(k-2)/factorial(k) for k in range(0, 2*n+1)])
            bsumprime3 = sum(
                [self.b**k*k*(k-1)*(k-2)*R**(k-3)/factorial(k) for k in range(0, 2*n+1)])
            b = 1-exp(-self.b*R)*bsum
            bprime = exp(-self.b*R)*(-bsumprime +self.b*bsum)
            bprime2 = exp(-self.b*R)*(
                -bsumprime2 
                +2*self.b*bsumprime -self.b**2*bsum)
            bprime3 = (exp(-self.b*R)*(
                -bsumprime3 +2*self.b*bsumprime2 -self.b**2*bsumprime) 
                -self.b*exp(-self.b*R)*(-bsumprime2 +2*self.b*bsumprime 
                -self.b**2*bsum))
            summer += (-4*n*self.C[2*n]/R**(2*n+1)*bprime2
                       +4*n*(2*n+1)*self.C[2*n]/R**(2*n+2)*bprime 
                       + (2*n)*(2*n+1)*self.C[2*n]/R**(2*n+2)*bprime 
                       - (2*n)*(2*n+1)*(2*n+2)*self.C[2*n]/R**(2*n+3)*b 
                        + self.C[2*n]/R**(2*n)*bprime3
                        - 2*n*self.C[2*n]/R**(2*n+1)*bprime2
                        )
        out -= summer
        return out

    def potprime3(self, R):
        """
        Return the third derivative of the potential V/kB in K with respect 
        to position as a function of R in nm 

        Also includes the small separation correction
        """
        R = np.array(R, ndmin=1) # to array
        out = self.potprime3TT(R)
        mask = R < self.Rcutoff*self.Repsilon
        Rm = R[mask]
        out[mask] = -self.tildeA*exp(-self.tildea*Rm)*(
            6/Rm**4 + 6*self.tildea/Rm**3 
            +3*(Rm*self.tildea)**2/Rm**4 
            +(Rm*self.tildea)**3/Rm**4)
        return out

    def fit_tildes(self, R):
        pot = self.potTT(R)
        dpot = self.potprimeTT(R)
        def objective(tildes):
            tildeA, tildea = tildes 
            val = tildeA/R*exp(-tildea*R)
            deriv = tildeA*(-tildea*1/R*exp(-tildea*R) + -1/R**2*exp(-tildea*R))
            residues = [val-pot, deriv-dpot]
            return np.array(residues)
        res = scipy.optimize.differential_evolution(
            lambda x: (objective(x)**2).sum(), 
            bounds=[(1e4,1e8),(1e1,1e2)],disp=
            True
            )
        print(res)
        return res.x

NeonTT = TangToennies(
    A =     0.402915058383e+08,
    a1 =   -0.428654039586e+02,
    a2 =   -0.333818674327e+01,
    an1 =  -0.534644860719e-01,
    an2 =   0.501774999419e-02,
    b =     0.492438731676e+02,
    nmax =  8,
    C = {
        6:  0.440676750157e-01,
        8:  0.164892507701e-02,
        10: 0.790473640524e-04,
        12: 0.485489170103e-05,
        14: 0.382012334054e-06,
        16: 0.385106552963e-07
    },
    tildeA = 2.36770343e+06, # Fit in this work
    tildea = 3.93124973e+01, # Fit in this work
    Rcutoff = 0.4,
    mass_rel = 20.1797,
    Repsilon = 0.30894556,
    key = 'Bich-MP-2008-Ne',
    doi = '10.1080/00268970801964207'
)

ArTT = TangToennies(
    A =    4.61330146e7,
    a1 =  -2.98337630e1,
    a2 =  -9.71208881,
    an1 =  2.75206827e-2,
    an2 = -1.01489050e-2,
    b =    4.02517211e1,
    nmax = 8,
    C = {
        6: 4.42812017e-1,
        8: 3.26707684e-2,
        10: 2.45656537e-3,
        12: 1.88246247e-4,
        14: 1.47012192e-5,
        16: 1.17006343e-6
    },
    tildeA=9.36167467e5,
    tildea=2.15969557e1,
    epsilonkB=143.123,
    Repsilon=0.376182,
    Rcutoff=0.4,
    sigma = 0.335741,
    mass_rel = 39.948,
    key = 'Vogel-MP-2010-Ar',
    doi = '10.1080/00268976.2010.507557'
)

KrTT = TangToennies(
    A =    0.3200711798e8,
    a1 =  -0.2430565544e1  *10,
    a2 =  -0.1435536209    *10**2,
    an1 = -0.4532273868    /10,
    an2 =  0,
    b =    0.2786344368e1  *10,
    nmax = 8,
    C = {
        6: 0.8992209265e6  /10**6,
        8: 0.7316713603e7  /10**8,
        10: 0.7835488511e8 /10**10
    },
    tildeA = 0.8268005465e7 /10,
    tildea = 0.1682493666e1 *10,
    epsilonkB = 200.8753,
    Repsilon = 4.015802     /10,
    Rcutoff = 0.3,
    mass_rel = 83.798,
    key = 'Jaeger-JCP-2016-Kr',
    doi = '10.1063/1.4943959'
)
KrTT.add_recursive()

XeTT = TangToennies(
    A = 0.579317071e8,
    a1 = -0.208311994e1   *10,
    a2 = -0.147746919     *10**2,
    an1 = -0.289687722e1  /10,
    an2 = 0.258976595e1   /10**2,
    b = 0.244337880e1     *10,
    nmax = 8,
    C = {
        6:  0.200298034e7 /10**6, 
        8:  0.199130481e8 /10**8,
        10: 0.286841040e9 /10**10
    },
    tildeA = 4.18081481e+06, # Fit in this work
    tildea = 2.38954061e+01, # Fit in this work
    Rcutoff = 0.3,
    Repsilon = 4.37798    /10,
    mass_rel = 131.293,
    key = 'Hellmann-JCP-2017-Xe',
    doi = '10.1063/1.4994267'
)
XeTT.add_recursive()
# print(XeTT.fit_tildes(XeTT.Repsilon*XeTT.Rcutoff))

def diffassert(val,thresh, reference=''):
    if val > thresh:
        print(val, thresh, reference)
        assert(val < thresh)

def get_potvals (Rmin_m, Rmax_m, N, *, pot):
    R_m = np.linspace(Rmin_m, Rmax_m, 5000)
    R_nm = R_m*1e9
    V = pot.pot(R_nm)*k_B # [J]
    Vprime = pot.potprime(R_nm)*k_B*1e9 # [J/m]
    Vprime2 = pot.potprime2(R_nm)*k_B*1e9**2 # [J/m^2]
    Vprime3 = pot.potprime3(R_nm)*k_B*1e9**3 # [J/m^3]
    potvals = {
        'R / m': R_m,
        'V': V,
        'Vprime': Vprime,
        'Vprime2': Vprime2,
        'Vprime3': Vprime3,
        'mass_rel': pot.mass_rel
    }
    return potvals






"""
# First, calculate the entropy of ab initio Ar utilizing the Deiters-Hoheisel method.
# For this, we need to calculate the second virial coefficient of Ar at the low-density limit.
# V is in J!
"""

yy0 = get_potvals(0.02e-10,60e-10,1000000,pot=ArTT)
r0 = yy0['R / m']*1e10 # A
phi0 = yy0['V'] # J
phi0 = phi0*6.241509074461E+18 # J to eV
kb = 8.617333262e-5 # eV/K
T = 158.597 # Tcrit
kbT = kb*T
integrand = (np.exp(-phi0/kbT)-1)*r0**2
B2 = -2*np.pi*np.trapz(integrand,r0) # in A3

dc_exp = 536 # Argon, kg/m3
drs = np.arange(0.1,2.2,0.1)
densities = drs*dc_exp # in kgm3
densities_kgm3 = np.copy(densities)
mass = 39.948 # Ar
densities = densities / mass / 1e30 * 6.02214e23 * 1000 # A-3
   
pressures = []
energies = []
for dr in drs:
    ps = []
    es = []
    for setnum in range(1,4):
        addr = 'argon-Example/set'+'%d/dr%.2f' % (setnum,dr)+'/'
        if setnum == 1 and np.abs(dr-1.4)<=1e-5:
            continue
        with open(addr+'pressrun.out','r') as infile:
            for line in infile:
                continue
        ps.append(float(line.split()[-1]))
        with open(addr+'energyrun.out','r') as infile:
            for line in infile:
                continue
        es.append(float(line.split()[-1]))
    pressures.append(np.mean(ps))
    energies.append(np.mean(es))
pressures = np.array(pressures) # bar
pressures = pressures*1e5 # Pa
energies = np.array(energies)/2048 # eV per molecule
energies = energies/kbT # dimensionless energy
plt.plot(densities_kgm3,pressures/1e5,'o')
plt.show()
pp = np.polyfit(densities_kgm3[np.logical_and(drs>0.6,drs<1.4)],
                pressures[np.logical_and(drs>0.6,drs<1.4)]/1e5,3)
dc_sim = -pp[1]/(3*pp[0])
dr_sim = drs*dc_exp/dc_sim

eexcs = energies - 3/2 # residual energy
kbJK = 1.380649e-23 # J/K
kbTJ = kbJK*T
Zs = pressures/(densities*1e30*kbTJ)
hexcs = eexcs + Zs - 1 # dimensionless residual enthalpy
integrand = (Zs-1)/densities
integrand = np.insert(integrand, 0, B2)
densities = np.insert(densities, 0, 0)

s = UnivariateSpline(densities,integrand)
gexcs = []
for counter,density in enumerate(densities[1:]):
    dd = np.arange(0,density+1e-7,1e-7)
    yy = s(dd)
    gres = np.trapz(yy,dd) + Zs[counter] - 1
    gexcs.append(gres)

sexcs = hexcs - gexcs
splsaimd = -sexcs
s = UnivariateSpline(drs*dc_exp/dc_sim,splsaimd,s=0)
print(s(1))
plt.plot(drs,splsaimd,'o')
plt.show()

"""
# Second, calculate the entropy of ab initio Ar (repulsive version) using the Deiters-Hoheisel method.
"""
delta = 1e-6#A
r_min = 0.02 # A
r_max = 14 # A

yy1 = get_potvals(r_min*1e-10,r_max*1e-10,5000,pot=ArTT)
r1 = yy1['R / m']*1e10 # A
phi1 = yy1['V'] # J
phi1 = phi1*6.241509074461E+18
phimin = phi1[phi1==np.min(phi1)]
rm = r1[np.where(phi1==np.min(phi1))[0][0]]

yy0 = get_potvals(0.02e-10,60e-10,1000000,pot=ArTT)
r0 = yy0['R / m']*1e10 # A
phi0 = yy0['V'] # J
phi0 = phi0*6.241509074461E+18 # J to eV
phi0 = phi0 - phimin
phi0[r0>=rm] = 0

kb = 8.617333262e-5 # eV/K
kbT = kb*T
integrand = (np.exp(-phi0/kbT)-1)*r0**2
B2 = -2*np.pi*np.trapz(integrand,r0) # in A3

drs = np.arange(0.1,2.2,0.1)
densities = drs*dc_exp # in kgm3
densities = densities / mass / 1e30 * 6.02214e23 * 1000 # A-3
   
pressures = []
energies = []
for dr in drs:
    ps = []
    es = []
    for setnum in range(1,4):
        addr = 'argon-Example/Repulsion/set'+'%d/dr%.2f' % (setnum,dr)+'/'
        if setnum == 1 and np.abs(dr-1.3)<1e-5:
            continue
        with open(addr+'pressrun.out','r') as infile:
            for line in infile:
                continue
        ps.append(float(line.split()[-1]))
        with open(addr+'energyrun.out','r') as infile:
            for line in infile:
                continue
        es.append(float(line.split()[-1]))
    pressures.append(np.mean(ps))
    energies.append(np.mean(es))
pressures = np.array(pressures) # bar
pressures = pressures*1e5 # Pa
energies = np.array(energies)/2048 # eV per molecule
energies = energies/kbT # dimensionless energy
eexcs = energies - 3/2 # residual energy
kbJK = 1.380649e-23 # J/K
kbTJ = kbJK*T
Zs = pressures/(densities*1e30*kbTJ)
hexcs = eexcs + Zs - 1 # dimensionless residual enthalpy
integrand = (Zs-1)/densities
integrand = np.insert(integrand, 0, B2)
densities = np.insert(densities, 0, 0)

s = UnivariateSpline(densities,integrand)
gexcs = []
for counter,density in enumerate(densities[1:]):
    dd = np.arange(0,density+1e-7,1e-7)
    yy = s(dd)
    gres = np.trapz(yy,dd) + Zs[counter] - 1
    gexcs.append(gres)

sexcs = hexcs - gexcs
splsrep = -sexcs
plt.plot(drs,splsrep,'o')
plt.show()




"""
# Third, calculate the effective hard-sphere diameter of the repulsive version.
# I will use the hard-sphere entropy for calculating the effective hard-sphere diameter.
"""

def splusHS(eta):
    '''Calculate splus of hard-sphere model.'''
    #eta = density*np.pi/6*diameter**3
    splus = eta*(4-3*eta)/(1-eta)**2
    return splus

# find the interval.    
eta0 = np.linspace(0,0.7,100000)
splsHS0 = splusHS(eta0)
#plt.plot(eta0,splsHS0)
diameters = []
for counter,density in enumerate(densities[1:]):
    splrep = splsrep[counter]
    #plt.plot(eta0,splsHS0-splrep)
    a = eta0[splsHS0-splrep<0][-1]
    b = eta0[splsHS0-splrep>0][0]
    eta = brentq(lambda eta: splusHS(eta)-splrep,a,b)
    diameter = (eta/density/(np.pi/6))**(1/3)
    diameters.append(diameter)

"""
# Fourth, calculate the free volume based on the hard-sphere diameter for dissecting the entropy.
# This example requires trajectory data, which should be generated and cannot be uploaded to GitHub. (Large size)


diameter = np.mean(diameters)
for counter,dr in enumerate(np.arange(0.1,2.2,0.1)):    
    Pinss = []
    for setnum in range(1,4):
        addr = 'argon-Example/set%d/dr%.2f/' % (setnum,dr)        
        filename = addr+'/trajectory.npz'        
        if os.path.isfile(filename):
            locals().update(np.load(filename))                
            for pos in traj:
                Cavity = CavityVol(pos,L[1],radii=diameter,Probe_radius=diameters[counter]/2)
                Pins = 1 - Cavity.exclude/Cavity.vol
                Pinss.append(Pins)            

for counter,dr in enumerate(np.arange(0.1,2.2,0.1)):    
    Pinss = []
    for setnum in range(1,4):
        addr = 'argon-Example/Repulsion/set%d/dr%.2f/' % (setnum,dr)        
        filename = addr+'/trajectory.npz'        
        if os.path.isfile(filename):
            locals().update(np.load(filename))                
            for pos in traj:
                Cavity = CavityVol(pos,L[1],radii=diameter,Probe_radius=diameters[counter]/2)
                Pins = 1 - Cavity.exclude/Cavity.vol
                Pinss.append(Pins)            
"""