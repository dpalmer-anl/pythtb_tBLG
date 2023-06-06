# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:11:20 2023

@author: danpa
"""
import pythtb_tblg
import numpy as np
import scipy.optimize
import flatgraphene as fg
import matplotlib.pyplot as plt

def EPH_coupling(model,kgrid,nbands=10):
    model.solve_all(kgrid)
    fermi_ind = model._norb//2
    evals = model.get_eigenvalues()[fermi_ind-nbands//2:fermi_ind+nbands//2,:]
    evec = model.get_eigenvectors()[fermi_ind-nbands//2:fermi_ind+nbands//2,:,:]
    nkp = np.shape(kgrid)[0]
    G_nk = np.zeros((nbands,nbands,nkp,nkp,3))
    
    for k in range(nkp):
        for l in range(nkp):
            for i in range(model._norb):
                
                gradH = model.solver.gradH(i,kgrid[k,:])
                for j in range(model._norb):
                    elemx = evec[j,:,l].T @ gradH[0] @ evec[i,:,k]
                    elemy = evec[j,:,l].T @ gradH[1] @ evec[i,:,k]
                    elemz = evec[j,:,l].T @ gradH[2] @ evec[i,:,k]
                    G_nk[i,j,k,l,:] = [elemx,elemy,elemz]
                    
    return G_nk,evals
                
                
def V_eff(g,evals,omega):
    nkp = np.shape(evals)[1]
    nqp = np.shape(omega)[1]
    nbands = np.shape(evals)[0]
    V = np.zeros((nbands,nkp,nkp,nqp))
    for i in range(nkp):
        for j in range(nkp):
            for k in range(nqp/2):
                V[:,i,j,k] = g[:,i,k]*g[:,j,-k] * omega[k]/(np.power(evals[:,i]-evals[:,j],2)-omega[-k]**2)
    
    return V

def fermi_dirac(e,mu,T):
    kb=1
    if T>1e-4: #numerically unstable if T near 0
        return 1/(1+np.exp((e-mu)/kb/T))
    else:
        occ = np.where(e<=mu)[0]
        n = np.zeros_like(e)
        n[occ] = np.ones(len(occ))
        return n
    
def build_gap_func(V,evals,mu,T):
    def func(gap):
        total = gap + np.sum(V*(1-2*fermi_dirac(evals,mu,T))/2/evals*gap)
        return total
    return func
    
def SC_gap(V,evals,T):
    nkp = np.shape(evals)[1]
    nbands = np.shape(evals)[0]
    gap = np.zeros((nbands,nkp))
    for i in range(nbands):
        gap_func = build_gap_func(V[:,i,:],evals[i,:])
        gap[i,:] = scipy.optimize.newton(gap_func, np.ones(nkp))
    return gap

def get_phonons(kgrid,vm=1):
    #this fxn is just for acoustic phonons, build interface with lammps to get more accurate
    return vm*kgrid

if __name__=="__main__":
    lattice_constant = 2.4545
    twist = 7.34
     
    #create geometry
    p_found, q_found, theta_comp = fg.twist.find_p_q(twist)
    atoms=fg.twist.make_graphene(cell_type='hex',n_layer=2,
                   p=p_found,q=q_found,lat_con=lattice_constant,sym=["B","Ti"],
                         mass=[12.01,12.02],sep=3.444,h_vac=3)
     
    #example usage
    #parameters = [popov,letb,nam koshino], only popov available now
    model = pythtb_tblg.tblg_model(atoms,parameters='popov')
    kgrid = model.k_uniform_mesh((15,15,1))
    model.set_solver( {'cupy':False,
                    'sparse':False,
                    'writeout':'tBLG_grid_t'+str(twist),
                    'restart':True})
    
    #pre-processing, get phonons and effective potential
    omega = get_phonons(kgrid)
    G_nk,evals = EPH_coupling(model,kgrid,nbands=10)
    V = V_eff(G_nk,evals,omega)
    
    #calculate sc gap as a function of temperature
    Temp_ = np.linspace(0,10,20)
    gap_val = np.zeros((20,np.shape(kgrid)[0]))
    for i,T in enumerate(Temp_):
        gap_val[i,:] = SC_gap(V,evals,T)
        
    plt.plot(Temp_,np.max(gap_val,axis=1))
    plt.xlabel('Temp')
    plt.ylabel(r'$\Delta')
    plt.show()
    
        