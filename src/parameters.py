# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:01:00 2023

@author: danpa
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numba import njit
import numba
from numba.types import float64,int64,int32,complex128
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
from scipy import special
import joblib
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
import h5py
import time
import os
import subprocess

@njit
def hoppingInter_namkoshino(dR,kval):


    ez = [0,0,1]#;%unit vector along z axis
    d = la.norm(dR)
    d0 = 3.34 #interlayer spacing
    a = 2.529
    a0=a/np.sqrt(3)
    r0_pi = 0.184*a   #decay length of transfer integral

    Vpp_sigma0 = 0.48 #;%7.75792; 
    Vpp_pi0 = -2.7 #;%-3; 
   
 
    Vpp_sigma = Vpp_sigma0*np.exp(-(d-d0)/r0_pi) #; % here is the change of terms and interlayer distance is involved     
    Vpp_pi = Vpp_pi0*np.exp(-(d-a0)/r0_pi) #; % here is the change of terms
    Ezz = (dot(dR,ez)/d)**2*Vpp_sigma + (1-(dot(dR,ez)/d)**2)*Vpp_pi

    valmat = np.exp(1j*dot(dR,kval))*np.array([Ezz])


    return valmat  #*eV_per_hart

@njit
def hoppingIntra_namkoshino(dR,kval):

    ez = [0,0,1]#;%unit vector along z axis
    d = la.norm(dR)
    d0 = 3.34 #interlayer spacing
    a = 2.529
    a0=a/np.sqrt(3)
    r0_pi = 0.184*a   #decay length of transfer integral

    Vpp_sigma0 = 0.48 #;%7.75792; 
    Vpp_pi0 = -2.7 #;%-3; 
   
 
    Vpp_sigma = Vpp_sigma0*np.exp(-(d-d0)/r0_pi) #; % here is the change of terms and interlayer distance is involved     
    Vpp_pi = Vpp_pi0*np.exp(-(d-a0)/r0_pi) #; % here is the change of terms
    Ezz = (dot(dR,ez)/d)**2*Vpp_sigma + (1-(dot(dR,ez)/d)**2)*Vpp_pi

    valmat = np.exp(1j*dot(dR,kval))*np.array([Ezz])


    return valmat  #*eV_per_hart

@njit
def hoppingInter(dR,kval):
#CHEBYSHEV POLYNOMIAL EXPANSION FOR INTERLAYER MATRIX ELEMENT, POPOV AND
#ALSENOY 

    kval = np.transpose(kval)
    ang_per_bohr=0.529177 # [Anstroms/Bohr radius]
    eV_per_hart=27.2114 # [eV/Hartree]
    dR = dR/ang_per_bohr
    kval = kval*ang_per_bohr
    dRn = dR/(la.norm(dR))

    
    l = dRn[0]
    m = dRn[1]
    n = dRn[2]
    r = la.norm(dR)
  
    #%boundaries for polynomial
    aa=1. #[Bohr radii]
    b=10. #[Bohr radii]
    y = (2*r-(b+aa))/(b-aa)
    
   # %chebyshev polynomials
    #T = np.polynomial.chebyshev.chebval(y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    #%potential coefficients
    Css_sigma = np.array([-0.5286482, 0.4368816, -0.2390807, 0.0701587,
                            0.0106355, -0.0258943, 0.0169584, -0.0070929,
                            0.0019797, -0.000304])
    Csp_sigma = np.array([0.3865122, -0.2909735, 0.1005869, 0.0340820,
                            -0.0705311, 0.0528565, -0.0270332, 0.0103844,
                            -0.0028724, 0.0004584])
    Cpp_sigma = np.array([0.1727212, -0.0937225, -0.0445544, 0.1114266,
                            -0.0978079, 0.0577363, -0.0262833, 0.0094388,
                            -0.0024695, 0.0003863])
    Cpp_pi = np.array([-0.3969243, 0.3477657, -0.2357499, 0.1257478,
                             -0.0535682, 0.0181983, -0.0046855, 0.0007303,
                            0.0000225, -0.0000393])
    Vss_sigma =  chebval(y, Css_sigma) 
    Vsp_sigma =  chebval(y, Csp_sigma) 
    Vpp_sigma =  chebval(y, Cpp_sigma) 
    Vpp_pi =  chebval(y, Cpp_pi) 
    
    #Vss_sigma = Vss_sigma -Css_sigma[0]/2
    Vss_sigma -= Css_sigma[0]/2
    Vsp_sigma -= Csp_sigma[0]/2
    Vpp_sigma  -= Cpp_sigma[0]/2
    Vpp_pi -= Cpp_pi[0]/2
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi  #; %Changing only this as only
    
    
    valmat = np.exp(1j*dot(dR,kval))*np.array([Ezz])

    return valmat*eV_per_hart



@njit
def overlapInter(dR,kval):
    
#CHEBYSHEV POLYNOMIAL EXPANSION FOR INTERLAYER MATRIX ELEMENT, POPOV AND
#ALSENOY 
    ang_per_bohr=0.529177 # [Anstroms/Bohr radius]
    eV_per_hart=27.2114 # [eV/Hartree]
    dR = dR/ang_per_bohr
    kval = kval*ang_per_bohr
    kval = np.transpose(kval)
    #dR = D - Rs #%% d vector in the paper
    #spacinge = 1e-5
    #dRn = dR/(la.norm(dR)+spacinge)
    dRn = dR/(la.norm(dR))
    
    l = dRn[0]
    m = dRn[1]
    n = dRn[2]
    r = la.norm(dR)
  
    #%boundaries for polynomial
    aa=1. #[Bohr radii]
    b=10. #[Bohr radii]
    y = (2*r-(b+aa))/(b-aa)
    
    #overlap matrix coefficient (No units mentioned)
    Css_sigma=np.array([0.4524096, -0.3678693, 0.1903822, -0.0484968,
                            -0.0099673, 0.0153765, -0.0071442, 0.0017435,
                            -0.0001224, -0.0000443])
    Csp_sigma=np.array([-0.3509680, 0.2526017, -0.0661301, -0.0465212,
                            0.0572892, -0.0289944, 0.0078424, -0.0004892,
                            -0.0004677, 0.0001590])
    Cpp_sigma=np.array([-0.0571487, -0.0291832, 0.1558650, -0.1665997,
                            0.0921727, -0.0268106, 0.0002240, 0.0040319,
                            -0.0022450, 0.0005596])
    Cpp_pi=  np.array([0.3797305, -0.3199876, 0.1897988, -0.0754124,
                            0.0156376, 0.0025976, -0.0039498, 0.0020581,
                            -0.0007114, 0.0001427])
    Vss_sigma =  chebval(y, Css_sigma) 
    Vsp_sigma =  chebval(y, Csp_sigma) 
    Vpp_sigma =  chebval(y, Cpp_sigma) 
    Vpp_pi =  chebval(y, Cpp_pi) 
    
    #Vss_sigma = Vss_sigma -Css_sigma[0]/2
    Vss_sigma -= Css_sigma[0]/2
    Vsp_sigma -= Csp_sigma[0]/2
    Vpp_sigma  -= Cpp_sigma[0]/2
    Vpp_pi -= Cpp_pi[0]/2
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi  #; %Changing only this as only

    valmat = np.array([Ezz]) * np.exp(1j*dot(dR,kval))

    return valmat*eV_per_hart

@njit
def hoppingIntra(dR,kval):
    #CHEBYSHEV POLYNOMIAL EXPANSION FOR INTERLAYER MATRIX ELEMENT, POPOV AND
#ALSENOY 
    ang_per_bohr=0.529177 # [Anstroms/Bohr radius]
    eV_per_hart=27.2114 # [eV/Hartree]
    dR = dR/ang_per_bohr
    kval = kval*ang_per_bohr
    kval = np.transpose(kval)
    #dR = D - Rs #%% d vector in the paper
    #spacinge = 1e-5
    #dRn = dR/(la.norm(dR)+spacinge)
    dRn = dR/(la.norm(dR))
    
    l = dRn[0]
    m = dRn[1]
    n = dRn[2]
    r = la.norm(dR)
  
    #%boundaries for polynomial
    aa=1. #[Bohr radii]
    b=7. #[Bohr radii]
    y = (2.*r-(b+aa))/(b-aa)
    
    #%potential coefficients
    Css_sigma = np.array([-0.4663805, 0.3528951, -0.1402985, 0.0050519,
                            0.0269723, -0.0158810, 0.0036716, 0.0010301,
                            -0.0015546, 0.0008601])
    Csp_sigma = np.array([0.3395418, -0.2250358, 0.0298224, 0.0653476,
                            -0.0605786, 0.0298962, -0.0099609, 0.0020609,
                            0.0001264, -0.0003381])
    Cpp_sigma = np.array([0.2422701, -0.1315258, -0.0372696, 0.0942352,
                            -0.0673216, 0.0316900, -0.0117293, 0.0033519,
                            -0.0004838, -0.0000906])
    Cpp_pi = np.array([-0.3793837, 0.3204470, -0.1956799, 0.0883986,
                            -0.0300733, 0.0074465, -0.0008563, -0.0004453,
                            0.0003842, -0.0001855])

    Vss_sigma =  chebval(y, Css_sigma) 
    Vsp_sigma =  chebval(y, Csp_sigma) 
    Vpp_sigma =  chebval(y, Cpp_sigma) 
    Vpp_pi =  chebval(y, Cpp_pi) 

    
    #Vss_sigma = Vss_sigma -Css_sigma[0]/2
    Vss_sigma -= Css_sigma[0]/2
    Vsp_sigma -= Csp_sigma[0]/2
    Vpp_sigma  -= Cpp_sigma[0]/2
    Vpp_pi -= Cpp_pi[0]/2
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi  #; %Changing only this as only
    # %unit vector along z axis is involved
     #%Ezz = ((dot(dR,ez)/d)^2)*Vpp_sigma + (1-(dot(dR,ez)/d)^2)*Vpp_pi;
    
    valmat = np.exp(1j*dot(dR,kval))*np.array([Ezz])
    return valmat*eV_per_hart

@njit
def overlapIntra(dR,kval):
    #CHEBYSHEV POLYNOMIAL EXPANSION FOR INTERLAYER MATRIX ELEMENT, POPOV AND
#ALSENOY 
    ang_per_bohr=0.529177 # [Anstroms/Bohr radius]
    eV_per_hart=27.2114 # [eV/Hartree]
    dR = dR/ang_per_bohr
    kval = kval*ang_per_bohr
    kval = np.transpose(kval)
    #dR = D - Rs #%% d vector in the paper
    #spacinge = 1e-5
    #dRn = dR/(la.norm(dR)+spacinge)
    dRn = dR/(la.norm(dR))
    
    l = dRn[0]
    m = dRn[1]
    n = dRn[2]
    r = la.norm(dR)
  
    #%boundaries for polynomial
    aa=1. #[Bohr radii]
    b=7. #[Bohr radii]
    y = (2.*r-(b+aa))/(b-aa)
    
    Css_sigma=np.array([0.4728644, -0.3661623, 0.1594782, -0.0204934,
                            -0.0170732, 0.0096695, -0.0007135, -0.0013826,
                            0.0007849, -0.0002005])
    Csp_sigma=np.array([-0.3662838, 0.2490285, -0.0431248, -0.0584391,
                            0.0492775, -0.0150447, -0.0010758, 0.0027734,
                            -0.0011214, 0.0002303])
    """
            When compared to "Construction of tight binding like potentials ...
            Applications to carbon" paper by Porezag, the Sppsigma Spppi tables
            HAVE BEEN SWAPPED. This is because that paper INCORRECTLY LABELED
            (swapped) those elements in the table and the corresponding plot.
            For proof of this look at the "Transferable density functional
            tight binding for carbon ..." by Cawkwell.
        """
    Cpp_sigma=np.array([-0.1359608, 0.0226235, 0.1406440, -0.1573794,
                            0.0753818, -0.0108677, -0.0075444, 0.0051533,
                            -0.0013747, 0.0000751])
    Cpp_pi=   np.array([0.3715732, -0.3070867, 0.1707304, -0.0581555,
                            0.0061645, 0.0051460, -0.0032776, 0.0009119,
                            -0.0001265, -0.000227])
    Vss_sigma =  chebval(y, Css_sigma) 
    Vsp_sigma =  chebval(y, Csp_sigma) 
    Vpp_sigma =  chebval(y, Cpp_sigma) 
    Vpp_pi =  chebval(y, Cpp_pi) 

    
    #Vss_sigma = Vss_sigma -Css_sigma[0]/2
    Vss_sigma -= Css_sigma[0]/2
    Vsp_sigma -= Csp_sigma[0]/2
    Vpp_sigma  -= Cpp_sigma[0]/2
    Vpp_pi -= Cpp_pi[0]/2
    Ezz = n**2*Vpp_sigma + (1-n**2)*Vpp_pi  #; %Changing only this as only
    # %unit vector along z axis is involved
     #%Ezz = ((dot(dR,ez)/d)^2)*Vpp_sigma + (1-(dot(dR,ez)/d)^2)*Vpp_pi;
    valmat = np.array([Ezz]) * np.exp(1j*dot(dR,kval))
    return valmat*eV_per_hart

@njit
def dot(x1,x2):
    sum=0
    for i in range(len(x1)):
        sum += x1[i] * x2[i]
    return sum

@njit
def chebval(x, c, tensor=True):
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2*x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x

@njit
def dot(x1,x2):
    sum=0
    for i in range(len(x1)):
        sum += x1[i] * x2[i]
    return sum
@njit
def diag(arr):
    ndim = len(arr)
    new_arr = np.zeros((ndim,ndim),dtype=complex128)
    for i in range(ndim):
        new_arr[i,i] = arr[i]
    return arr

@njit
def wrap_disp(r1,r2, cell):
    """Wrap positions to unit cell."""
    RIJ=np.zeros(3)
    d = 1000
    drij=np.zeros(3)
    
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            pbc=[i,j,0]
            
            #RIJ = r2 + np.matmul(pbc,cell)  - r1
            RIJ[0] = r2[0] + pbc[0]*cell[0,0] + pbc[1]*cell[1,0] +\
                    pbc[2]*cell[2,0] - r1[0]
            
            RIJ[1] = r2[1] + pbc[0]*cell[0,1] + pbc[1]*cell[1,1] + \
                pbc[2]*cell[2,1] - r1[1]
            
            RIJ[2] = r2[2] + pbc[0]*cell[0,2] + pbc[1]*cell[1,2] + \
                pbc[2]*cell[2,2] - r1[2]
            if np.linalg.norm(RIJ)<d:
                d = np.linalg.norm(RIJ)
                drij = RIJ.copy()

    
    return drij

          
@njit
def gen_ham_popov(xyz, cell, layer_tags,use_hoppingInter,use_hoppingIntra,
            use_overlapInter,use_overlapIntra, rcut_inplane=3.6, rcut_interlayer=5.29,kval=np.array([0,0,0])):
    """ get pz only tight binding energy spectrum from a given ase.atoms object. Note**
    to use interlayer hoppings, atoms_obj.symbols must contain differing symbols
    for atoms in different layers. At least one hopping function must be provided
    energies in eV
    
    atoms_obj (ase.atoms object) atoms object to calculate energy for. can be 
    from a lammps dump or data file. Or generated using flatgraphene module
    
    hoppingIntra,overlapIntra,hoppingInter,overlapInter (function) slater koster function for
    tight binding parameters. must take in positions of each atom, and kval
    
    r_cut_interlayer(float): interlayer cutoff radius
    
    r_inplance (float): inplane cutoff radius
    
    kval (3x1 array): point in kspace to calculate energy at
    
    """
    periodicR1 = cell[0,:]
    periodicR2 = cell[1,:]
    periodicR3 = cell[2,:]
    V = dot(periodicR1,np.cross(periodicR2,periodicR3))
    b1 = 2*np.pi*np.cross(periodicR2,periodicR3)/V
    b2 = 2*np.pi*np.cross(periodicR1,periodicR3)/V
    b3 = 2*np.pi*np.cross(periodicR1,periodicR2)/V
    kval = kval[0]*b1 + kval[1]*b2 + kval[2]*b3
    natoms = np.shape(xyz)[0]
    Es_C =  -13.7388 # eV 
    Ep_C = -5.2887  # eV
    EnergiesOfCarbon = [Es_C, Ep_C, Ep_C, Ep_C]
    if use_hoppingIntra:
        test_val = hoppingIntra(np.array([3.,3.,3.]),np.array([0,0,0]))
    elif use_hoppingInter:
        test_val = hoppingInter(np.array([3.,3.,3.]),np.array([0,0,0]))
    else:
        print("add hopping function")
    orbs_per_atom= test_val.shape[0]
    #add in code to take care of multiple orbitals at same atom
    Hmatrix = np.array([np.complex128(x) for x in range(0)],dtype=complex128)
    H_row = np.array([np.int64(x) for x in range(0)],dtype=int64)
    H_col = np.array([np.int64(x) for x in range(0)],dtype=int64)
    Smatrix = np.array([np.complex128(x) for x in range(0)],dtype=complex128)
    S_row = np.array([np.int64(x) for x in range(0)],dtype=int64)
    S_col = np.array([np.int64(x) for x in range(0)],dtype=int64)
    for i in range (0,natoms, 1): #%going to each atom one by one
        ri = xyz[i,:]
        curr_tag=layer_tags[i]
        for j in range (i, natoms, 1): 
            rj = xyz[j,:]
            if i == j: #insert self energies on diagonal
                Hmatrix = np.append(Hmatrix,Ep_C)
                
                H_row = np.append(H_row,i)
                H_col = np.append(H_col,j)
                
                if use_overlapInter or use_overlapIntra:
                    Smatrix= np.append(Smatrix,1)
                    S_row= np.append(S_row,i)
                    S_col = np.append(S_col,j)
                continue
            disp = wrap_disp(ri,rj,cell)
            dist = np.linalg.norm(disp)
            #INPLANE INTERACTION,USING R_CUT TO CHOOSE NEIGHBOR
            if curr_tag == layer_tags[j] and use_hoppingIntra: #or insert tolerance
                if (dist)<rcut_inplane:
                    hopMat = hoppingIntra(disp,kval)
                    Hmatrix = np.append(Hmatrix,hopMat)
                    H_row = np.append(H_row,i)
                    H_col = np.append(H_col,j)
                    Hmatrix = np.append(Hmatrix,hopMat.conj().T)
                    H_row= np.append(H_row,j)
                    H_col = np.append(H_col,i)
                    if use_overlapIntra:
                        overMat = overlapIntra(disp,kval)
                        Smatrix = np.append(Smatrix,overMat)
                        S_row = np.append(S_row,i)
                        S_col = np.append(S_col,j)
                        Smatrix = np.append(Smatrix,overMat.conj().T)
                        S_row = np.append(S_row,j)
                        S_col = np.append(S_col,i)
            #INTERLAYER INTERACTION, NOT APPLYING R_CUT, RIGIDLY CHOOSING NEIGHBOR       
            elif use_hoppingInter:
                
                if dist < rcut_interlayer:
                     
                    hopMat = hoppingInter(disp,kval)
                    Hmatrix = np.append(Hmatrix,hopMat)
                    H_row = np.append(H_row,i)
                    H_col = np.append(H_col,j)
                    Hmatrix = np.append(Hmatrix,hopMat.conj().T)
                    H_row= np.append(H_row,j)
                    H_col = np.append(H_col,i)
                    if use_overlapIntra:
                        overMat = overlapIntra(disp,kval)
                        Smatrix = np.append(Smatrix,overMat)
                        S_row = np.append(S_row,i)
                        S_col = np.append(S_col,j)
                        Smatrix = np.append(Smatrix,overMat.conj().T)
                        S_row = np.append(S_row,j)
                        S_col = np.append(S_col,i)

    return Hmatrix,H_row,H_col,Smatrix,S_row,S_col