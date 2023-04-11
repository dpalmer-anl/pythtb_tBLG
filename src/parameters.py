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