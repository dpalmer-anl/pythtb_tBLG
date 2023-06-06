# -*- coding: utf-8 -*-
"""
Created on Mon May  1 10:21:25 2023

@author: danpa
"""

import numpy as np
import matplotlib.pyplot as plt
import pythtbtBLG.parameters
from tqdm import tqdm
from numba import njit
import numba
from numba.types import float64,int64,int32,complex128
from numba.core import types
from numba.typed import Dict


@njit
def get_qHoppings(pos,cell,sublat,i,q,rcut_interlayer = 5.29):
    natoms = np.shape(pos)[0]
    t_rq = np.zeros((2,2),dtype=complex128)
    l_ind = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.int64[:],
    )
    sl_ind = {'ArAr':np.asarray([0,0], dtype='f8'),
             'ArB':np.asarray([0,1], dtype='f8'),
             'BAr':np.asarray([1,0], dtype='f8'),
             'BB':np.asarray([1,1], dtype='f8')}
    for j in range(natoms):
        delta_disp = parameters.wrap_disp(pos[i,:] , pos[j,:] ,cell)
        dist = np.linalg.norm(delta_disp)
        if i==j:
            continue
        if dist < rcut_interlayer:
            ind = sl_ind[sublat[i]+sublat[j]]
            t_rq[ind] += parameters.hoppingInter(delta_disp,q)
            
    return t_rq/natoms
#get hopping amplitudes at gamma point
#get displacements for each amplitude
#construct tq and sum over all lattice points

def get_recip_vec(cell):
    periodicR1 = cell[0,:]
    periodicR2 = cell[1,:]
    periodicR3 = cell[2,:]
    V = np.dot(periodicR1,np.cross(periodicR2,periodicR3))
    b1 = 2*np.pi*np.cross(periodicR2,periodicR3)/V
    b2 = 2*np.pi*np.cross(periodicR3,periodicR1)/V
    b3 = 2*np.pi*np.cross(periodicR1,periodicR2)/V
    return np.stack((b1,b2,b3),axis=1)
    
def KP_kernel(atoms,p,r,nq=2):
    Ksym = np.array([2/3,1/3,0])
    pos = np.array(atoms.positions,dtype=np.double)
    sublat = np.array(atoms.sublattice,dtype=np.str_)
    natoms = atoms.get_global_number_of_atoms()
    cell = np.array(atoms.get_cell(),dtype=np.double)
    recip_cell = get_recip_vec(cell)
    #figure out what nq is, how to generate appropriate kpoints
    waven=(2*nq+1)**2

    X,Y = np.meshgrid(np.arange(-nq,nq+1,1),np.arange(-nq,nq+1,1)) #reciprocal lattice
    K = np.zeros((waven,3))
    K[:,0] = X.flatten()
    K[:,1] = Y.flatten()
    K = K@recip_cell.T
    
    KP_k = np.zeros((2,2),dtype=complex)
    va = np.array([1/3,1/3,0]) @ cell.T
    vb = np.array([2/3,2/3,0]) @ cell.T
    for j in range(waven):
        M = np.exp(1j*np.dot((K[j,:]-Ksym),(va-vb))) #figure out what va, vb are
        q = p + K[j,:]- Ksym #?
        Tq = get_qHoppings(pos,cell,sublat,r,q)
        KP_k +=  M * Tq
    return KP_k
    
    
def H_tbq(atoms,kpoints,nq = 2):
    nkp = np.shape(kpoints)[0]
    natoms = atoms.get_global_number_of_atoms()
    pos = atoms.positions
    cell = atoms.get_cell()
    recip_cell = get_recip_vec(cell)
    #figure out what nq is, how to generate appropriate kpoints
    waven=(2*nq+1)**2

    X,Y = np.meshgrid(np.arange(-nq,nq+1,1),np.arange(-nq,nq+1,1)) #reciprocal lattice
    K = np.zeros((waven,3))
    K[:,0] = X.flatten()
    K[:,1] = Y.flatten()
    K = K@recip_cell.T
    
    KP_Ham = np.zeros((2*waven,2*waven,nkp),dtype=complex)
    for ksc in tqdm(range(nkp)):
        for i in range(waven):
            for j in range(i,waven):
                H_elem = 0
                for r in range(natoms):
                    H_elem += np.exp(1j*np.dot(K[i,:]-K[j,:],pos[r,:])) *\
                        KP_kernel(atoms,kpoints[ksc,:],r,nq=2)
                KP_Ham[2*i:2*(i+1),2*j:2*(j+1),ksc] = H_elem
        KP_Ham[:,:,ksc] += KP_Ham[:,:,ksc].conj().T\
            - np.diag(np.diag(KP_Ham[:,:,ksc].conj().T))
                
    return KP_Ham


if __name__=="__main__":
    import matplotlib.pyplot as plt
    import pythtb_tblg
    import twist
    Gamma = [0,   0,   0]
    K = [2/3,1/3,0]
    Kprime = [1/3,2/3,0]
    M = [1/2,0,0]
    sym_pts=[K,Gamma,M,K]
    nk=32

    theta=16.43
    lattice_constant = 2.46
    p_found, q_found, theta_comp = twist.find_p_q(theta)
    atoms=twist.make_graphene(cell_type='hex',n_layer=2,
                                      p=p_found,q=q_found,lat_con=lattice_constant,sym=["B","Ti"],
                                            mass=[12.01,12.02],sep=3.444,h_vac=3)
    
    model = pythtb_tblg.tblg_model(atoms,parameters='popov')
    model.set_solver( {'cupy':False,
                        'sparse':False})
    
    Gamma = [0,   0,   0]
    K = [2/3,1/3,0]
    Kprime = [1/3,2/3,0]
    M = [1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=40
    (k_vec,k_dist,k_node) = model.k_path(sym_pts,nk)
    print('building K.P model\n')
    KP_ham = H_tbq(atoms,k_vec)

    eigvals = np.zeros((np.shape(KP_ham)[0],nk))
    print('diagonalizing K.P model')
    for i in tqdm(range(nk)):
        eigvals[:,i] = np.linalg.eigvalsh(KP_ham[:,:,i])
        
    for j in range(np.shape(eigvals)[0]):
        plt.plot(eigvals[j,:])
    plt.show()              
