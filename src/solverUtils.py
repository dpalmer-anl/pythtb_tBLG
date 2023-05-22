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
import copy
from parameters import gen_ham_popov
import shutil

class solver(object):
    def __init__(self,model):
        self._model=model
    
    def sol_ham(self,ham,overlap,dev_num):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        # check that matrix is hermitian
        #solve matrix 
        
        if not self._model.solve_dict['sparse']:
            
            if self._model.solve_dict['cupy']:
                import cupy
                with cupy.cuda.Device(dev_num):
                    s = cupy.cuda.Stream(non_blocking = True)
                    with s:
                        ham = cupy._cupyx.scipy.sparse.csr_matrix(ham)
                        ham = ham.toarray() #cupy.asarray(ham)
                        LM_eval,eigvec = cupy.linalg.eigh(ham)
                        eval = cupy.asnumpy(LM_eval)
                        eigvec = cupy.asnumpy(eigvec)
            else:
                #find k eigenvalues around fermi level
                ham  = ham.toarray()
                eval,eigvec = np.linalg.eigh(ham) #,b=overlap)
            eval,eigvec = _nicefy_eig(eval,eigvec)
            del ham
            del overlap
            
            return eval,eigvec
        else:
            efermi = self._model.solve_dict['fermi energy']
            k=self._model.solve_dict["num states"]
            if self._model.solve_dict['cupy']:
                import cupy
                import cupyx.scipy.sparse.linalg as cpl
                with cupy.cuda.Device(dev_num):
                    s = cupy.cuda.Stream(non_blocking = True)
                    with s:
                        ham = cupy._cupyx.scipy.sparse.csr_matrix(ham)
                        ham -= efermi*cupy.eye(cupy.shape(ham)[0]) #shift invert
                        LM_eval,eigvec = cpl.eigsh(ham,k=k,which='LM')
                        eval = cupy.asnumpy(LM_eval)
                        eigvec = cupy.asnumpy(eigvec)
            else:
               
               eval,eigvec = spspla.eigsh(ham,k=k,which='LM',sigma=efermi)
            eval,eigvec = _nicefy_eig(eval,eigvec)
            del ham
            del overlap
            return eval,eigvec
    
    def gen_ham(self,kval):
        xyz,cell,layer_tags = ase_arrays(self._model.atoms)
        if self._model.parameters=='popov':
            use_hoppingInter= True
            use_hoppingIntra = True
            use_overlapInter = False
            use_overlapIntra = False
            Hmatrix,H_row,H_col,Smatrix,S_row,S_col = \
                gen_ham_popov(xyz, cell, layer_tags,use_hoppingInter,use_hoppingIntra,
                    use_overlapInter,use_overlapIntra,kval=kval)
        else:
            print("only tblg parameters implementation available currently is popov")
            exit()
        return Hmatrix,H_row,H_col,Smatrix,S_row,S_col
    
    def get_PMF(self):
        cell = self.atoms.get_cell()
        periodicR1 = cell[0,:]
        periodicR2 = cell[1,:]
        periodicR3 = cell[2,:]
        V = np.dot(periodicR1,np.cross(periodicR2,periodicR3))
        b1 = 2*np.pi*np.cross(periodicR2,periodicR3)/V
        b2 = 2*np.pi*np.cross(periodicR3,periodicR1)/V
        b3 = 2*np.pi*np.cross(periodicR1,periodicR2)/V
        kval =2/3*b1 + 1/3*b2 #evaluate at K point
        
        A_C = 2.4683456
        e=vf=1
        A_EDGE = A_C/np.sqrt(3)
        A = np.zeros((self.norbs,3))
        t0 = parameters.hoppingIntra([A_EDGE,0,0],kval)
        t = parameters.hoppingIntra(disp[inplane_ind],kval)
        dtk = t-t0
        A = dtk/e/vf
        return A
        
    def get_bands_func(self,k_list):
        orbs_per_atom = 1
        def func_to_return(indices,dev_num=1):
            if type(indices)==int:
                indices = [indices]
            nkp = np.shape(indices)[0]
            eval = np.zeros((self._model.num_eigvals,nkp))
            evec = np.zeros((self.norbs,self._model.num_eigvals,nkp),dtype=complex) 
            for i in range(nkp):
                kval = k_list[indices[i],:]
                Hmatrix,H_row,H_col,Smatrix,S_row,S_col = self.gen_ham(kval)
                
                ham =  csr_matrix((Hmatrix,(H_row,H_col)),shape=(self.norbs,self.norbs))
                overlap =  csr_matrix((Smatrix,(S_row,S_col)),shape=(self.norbs,self.norbs))
                # solve Hamiltonian
                tmpeval, tmpevec = self.sol_ham(ham,overlap,dev_num)
                eval[:,i] = tmpeval
                evec[:,:,i] = tmpevec

                if type(self._model.solve_dict['writeout']) == str:
                    fobj = os.path.join(self._model.solve_dict['writeout'],'kp_'+str(kval)+".hdf5")

                    with h5py.File(fobj, 'w') as f:
                        group = f.create_group(str(kval))
                        dset3 = group.create_dataset('kpoint',data=kval)
                        dset1 = group.create_dataset('eigvals', data=tmpeval)
                        dset2 = group.create_dataset('eigvecs', data=tmpevec)
                    subprocess.call('echo '+str(kval).replace(']','').replace('[','')
                                    +" >> "+os.path.join(self._model.solve_dict['writeout'],'kpoints.calc')
                                    ,shell=True)
                else:
                    return (np.squeeze(eval),np.squeeze(evec))
        return func_to_return
    
    def solve_all(self,k_list):
        orbs_per_atom= 1 
        norb = self._model.atoms.get_global_number_of_atoms() * orbs_per_atom
        self.norbs = norb
        if not self._model.solve_dict['sparse']:
            num_eigvals = norb
        else:
            num_eigvals = self._model.solve_dict["num states"]
        self._model.num_eigvals = num_eigvals
        
        if self._model.solve_dict['writeout']!=None:
            if self._model.solve_dict['restart']:
                if not os.path.exists(self._model.solve_dict['writeout']):
                    os.mkdir(self._model.solve_dict['writeout'])
                elif os.path.exists(os.path.join(self._model.solve_dict['writeout'],'kpoints.calc')):
                    kcalc = np.loadtxt(os.path.join(self._model.solve_dict['writeout'],'kpoints.calc'))
                    use = slices_inarray(k_list,kcalc,axis=0,invert=True)
                    k_list = k_list[use,:]
                    
                    if np.shape(k_list)[0] == 0:
                        self._model.read_data=True
                        return None
                        
            else:
                if os.path.exists(self._model.solve_dict['writeout']):
                    shutil.rmtree(self._model.solve_dict['writeout'])
                os.mkdir(self._model.solve_dict['writeout'])
        
        #solve
        nkp=len(k_list) # number of k points
        band_func = self.get_bands_func(k_list)
        if not self._model.solve_dict['writeout']:
            self._model.read_data = False
            
            ret_eval = np.zeros((num_eigvals,nkp))
            #    indices are [orbital,band,kpoint,spin]
            eigvect = np.zeros((norb,num_eigvals,nkp),dtype=complex)
            #change return to read hdf5 files instead
            if not self._model.solve_dict['cupy']:

                number_of_cpu = joblib.cpu_count()
                output = Parallel(n_jobs=number_of_cpu)(delayed(band_func)(i) for i in range(nkp))
                for i in range(nkp):
                    ret_eval[:,i] = output[i][0]
                    eigvect[:,:,i] = output[i][1]
            else:
                #try to parallelize this loop
                ngpu = self._model.solve_dict['cupy']
                part = nkp//(ngpu)
                kind = np.array(range(nkp))
                use_ind = np.split(kind,ngpu)
                for i in range(ngpu):
                    tmp_evals,tmp_evec = band_func(use_ind[i],dev_num=i)
                    ret_eval[:,use_ind[i]] = tmp_evals 
                    eigvect[:,:,use_ind[i]] = tmp_evec
                    
            # indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
            self._model.eigenvalues = ret_eval
            self._model.eigenvectors = eigvect
            
        else:
            self._model.read_data = True
            if not self._model.solve_dict['cupy']:
                number_of_cpu = joblib.cpu_count()
                Parallel(n_jobs=number_of_cpu)(delayed(band_func)(i) for i in range(nkp))

            else:
                #try to parallelize this loop
                ngpu = self._model.solve_dict['cupy']
                part = nkp//(ngpu)
                kind = range(nkp)
                use_ind = np.split(np.array(kind),ngpu)
                for i in range(ngpu):
                    band_func(use_ind[i],dev_num=i)
                

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

def _nicefy_eig(eval,eig):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=eval.real
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    eig=eig[:,args]
    return (eval,eig)

def ase_arrays(atoms_obj):
    xyz= atoms_obj.positions
    layer_tags=list(atoms_obj.symbols)
    natoms = xyz.shape[0]
    cell=atoms_obj.get_cell()
        
    return np.array(xyz,dtype=np.double),np.array(cell,dtype=np.double),np.array(layer_tags,dtype=np.str_)
