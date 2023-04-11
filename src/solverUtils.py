# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:12:11 2022

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

def ase_arrays(atoms_obj):
    xyz= atoms_obj.positions
    layer_tags=list(atoms_obj.symbols)
    natoms = xyz.shape[0]
    cell=atoms_obj.get_cell()
        
    return np.array(xyz,dtype=np.double),np.array(cell,dtype=np.double),np.array(layer_tags,dtype=np.str_)
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
def gen_ham(xyz, cell, layer_tags,use_hoppingInter,use_hoppingIntra,
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


#@njit
def _nicefy_eval(eval):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=eval.real
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    return eval
#@njit
def _nicefy_eig(eval,eig):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=eval.real
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    eig=eig[:,args]
    return (eval,eig)



def sol_ham(ham,overlap,k,eig_vectors,dev_num,efermi=-4.511911038906458,
            use_cupy=False,sparse=True):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        # check that matrix is hermitian
        #if np.real(np.max(ham-ham.T.conj()))>1.0E-9:
        #    raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        #solve matrix 
        if not sparse:
            
            if use_cupy:
                import cupy
                with cupy.cuda.Device(dev_num):
                    ham = cupy._cupyx.scipy.sparse.csr_matrix(ham)
                    ham = ham.toarray() #cupy.asarray(ham)
                    LM_eval,eigvec = cupy.linalg.eigh(ham)
                    LM_eval = cupy.asnumpy(LM_eval)
                    eigvec = cupy.asnumpy(eigvec)
            else:
                #find k eigenvalues around fermi level
                ham  = ham.toarray()
                LM_eval,eigvec = np.linalg.eigh(ham) #,b=overlap)
            eval,eigvec = _nicefy_eig(LM_eval,eigvec)
            del ham
            del overlap
            
            return eval,eigvec
        else:
            if use_cupy:
                import cupy
                import cupyx.scipy.sparse.linalg as cpl
                with cupy.cuda.Device(1):
                    ham = cupy._cupyx.scipy.sparse.csr_matrix(ham)
                    ham -= efermi*cupy.eye(cupy.shape(ham)[0]) #shift invert
                    LM_eval,eigvec = cpl.eigsh(ham,k=k,which='LM')
                    LM_eval = cupy.asnumpy(LM_eval)
                    eigvec = cupy.asnumpy(eigvec)
            else:
               
               LM_eval,eigvec = spspla.eigsh(ham,k=k,which='LM',sigma=efermi)
            eval,eigvec = _nicefy_eig(LM_eval,eigvec)
            del ham
            del overlap
            return eval,eigvec

def get_bands_func(xyz, cell, layer_tags,use_hoppingInter,use_hoppingIntra,
            use_overlapInter,use_overlapIntra, rcut_inplane, 
            rcut_interlayer,k_list,k,use_cupy,sparse,dev_num,fobj=None):
    
    if use_hoppingIntra:
        test_val = hoppingIntra(np.array([3.,3.,3.]),np.array([0,0,0]))
    elif use_hoppingInter:
        test_val = hoppingInter(np.array([3.,3.,3.]),np.array([0,0,0]))
        
    orbs_per_atom = np.shape(test_val)[0]
    def func_to_return(ind):
        kval = k_list[ind,:]
        Hmatrix,H_row,H_col,Smatrix,S_row,S_col = gen_ham(xyz, cell, layer_tags,use_hoppingInter,use_hoppingIntra,
            use_overlapInter,use_overlapIntra, rcut_inplane=rcut_inplane, 
                  rcut_interlayer=rcut_interlayer,kval=kval)
        
        ham =  csr_matrix((Hmatrix,(H_row,H_col)),shape=(xyz.shape[0]*orbs_per_atom,xyz.shape[0]*orbs_per_atom))
        overlap =  csr_matrix((Smatrix,(S_row,S_col)),shape=(xyz.shape[0]*orbs_per_atom,xyz.shape[0]*orbs_per_atom))
        # solve Hamiltonian
        eval, evec = sol_ham(ham,overlap,k,True,dev_num,use_cupy=use_cupy,sparse=sparse)
        if fobj != None:
            if os.path.exists(fobj+'/kp_'+str(kval)+".hdf5"):
                subprocess.call("rm -f \'"+fobj+'/kp_'+str(kval)+".hdf5\'",shell=True)
            with h5py.File(fobj+'/kp_'+str(kval)+".hdf5", 'a') as f:
                group = f.create_group(str(kval))
                dset1 = group.create_dataset('eigvals', data=eval)
                dset2 = group.create_dataset('eigvecs', data=evec)
            subprocess.call('echo '+str(kval).replace(']','').replace('[','')
                            +" >> "+os.path.join(fobj,'kpoints.calc')
                            ,shell=True)

        return (eval,evec)
    return func_to_return


def solve_all(xyz, cell, layer_tags,k_list,use_hoppingInter,use_hoppingIntra,
            use_overlapInter,use_overlapIntra, num_eigvals,use_cupy,
            sparse,dev_num,rcut_inplane=3.6, rcut_interlayer=5.29,output_name=None,restart=True):
    all_kpoints = k_list.copy()
    nsta = len(layer_tags)
    if use_hoppingIntra:
        test_val = hoppingIntra(np.array([3.,3.,3.]),np.array([0,0,0]))
    elif use_hoppingInter:
        test_val = hoppingInter(np.array([3.,3.,3.]),np.array([0,0,0]))
    else:
        print("add hopping function")
  
    orbs_per_atom= 1 #np.shape(test_val)[0]
    norb = int(len(layer_tags) * orbs_per_atom)
    #num_eigvals = 30 #int(eigfrac * norb)
    if not sparse:
        num_eigvals = norb
    
    if output_name!=None:
        if restart:
            if not os.path.exists(output_name):
                os.mkdir(output_name)
            if os.path.exists(os.path.join(output_name,'kpoints.calc')):
                kcalc = np.loadtxt(os.path.join(output_name,'kpoints.calc'))
                use = slices_inarray(k_list,kcalc,axis=0,invert=True)
                k_list = k_list[use,:]
                if np.shape(k_list)[0] == 0:
                    print("all kpoints already calculated")
                    exit()
                    
        else:
            if not os.path.exists(output_name):
                os.rmdir(output_name)
            os.mkdir(output_name)
        
    nkp=len(k_list) # number of k points
    ret_eval = np.zeros((num_eigvals,nkp))
    #    indices are [orbital,band,kpoint,spin]
    eigvect = np.zeros((norb,num_eigvals,nkp),dtype=complex)
    band_func = get_bands_func(xyz, cell, layer_tags,use_hoppingInter,
            use_hoppingIntra,use_overlapInter,use_overlapIntra, rcut_inplane,
              rcut_interlayer,k_list,num_eigvals,use_cupy,sparse,dev_num,fobj=output_name)
    
    #change return to read hdf5 files instead
    if not use_cupy:
        number_of_cpu = joblib.cpu_count()
        output = Parallel(n_jobs=number_of_cpu)(delayed(band_func)(i) for i in range(nkp))
        for i in range(nkp):
            ret_eval[:,i] = output[i][0]
            eigvect[:,:,i] = output[i][1]
    else:
        #try to parallelize this loop
        for i in range(nkp):
            tmp_evals,tmp_evec = band_func(i)
            ret_eval[:,i] = tmp_evals 
            eigvect[:,:,i] = tmp_evec 
    
    # ret_eval= output[0]
    # eigvect = output[1]
    # indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
    return ret_eval, eigvect

def slices_inarray(check_array,test_array,axis=0,invert=False):
    #axis to iterate over
    nk = np.shape(check_array)[0]
    isin = np.full(nk,False)
    for i in range(nk):
        for j in range(np.shape(test_array)[0]):
            tmpval = np.allclose(check_array[i,:],test_array[j,:])
            if tmpval:
                isin[i] = True
    if invert:
        isin =  [not elem for elem in isin]
    return isin

def get_bands(atoms,k_list,use_hoppingInter=False,use_hoppingIntra=False,
            use_overlapInter=False,use_overlapIntra=False,num_eigvals=30,
            use_cupy=False,sparse=True,output_name=None):
    ngpu = 6
    nkp = np.shape(k_list)[0]
    xyz,cell,layer_tags = ase_arrays(atoms)
    for i in range(nkp-1):
        dev_num = i%ngpu
        if i==nkp-1:
            k_use = k_list[int(ngpu*i):]
        else:
            k_use = k_list[int(ngpu*i):int(ngpu*(i+1))]
        ret_eval,eigvect = solve_all(xyz, cell, layer_tags,k_list,\
                use_hoppingInter,use_hoppingIntra,\
                use_overlapInter,use_overlapIntra,num_eigvals,use_cupy,sparse,
                dev_num,output_name=output_name)
    
    return ret_eval,eigvect
    
