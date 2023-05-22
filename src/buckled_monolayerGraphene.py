# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:42:01 2023

@author: danpa
"""

import pythtb_tblg
import numpy as np
import ase.io
import matplotlib.pyplot as plt

def plot_bands(colors,labels,all_evals,erange=1.0,title='',figname=None):
    fig, ax = plt.subplots()
    label=(r'$K$', r'$M$', r'$\Gamma $',r'$K$')
    # specify horizontal axis details
    # set range of horizontal axis
    ax.set_xlim(k_node[0],k_node[-1])
    # put tickmarks and labels at node positions
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    # add vertical lines at node positions
    for n in range(len(k_node)):
      ax.axvline(x=k_node[n],linewidth=0.5, color='k')
    # put title
    ax.set_title(title)
    ax.set_xlabel("Path in k-space")
    ax.set_ylabel("Band energy")
    for i in range(np.shape(all_evals)[0]):
        evals = all_evals[i]
        nbands = np.shape(evals)[0]
        efermi = np.mean([evals[nbands//2,0],evals[(nbands-1)//2,0]])
        fermi_ind = nbands//2
        # plot first and second band
        for n in range(np.shape(evals)[0]):
            if i==0:
                ax.plot(k_dist,evals[n,:]-efermi,c=colors[i],label=labels[i])
            else:
                ax.plot(k_dist,evals[n,:]-efermi,c=colors[i])
                
        
    # make an PDF figure of a plot
    fig.tight_layout()
    ax.legend()
    ax.set_ylim(-erange,erange)
    fig.savefig(figname)
    
if __name__=="__main__":
    atoms_HC = ase.io.read('POSCAR_HighCorr.txt',format='vasp')
    atoms_LC = ase.io.read('POSCAR_LowCorr.txt',format='vasp')

    model_HC = pythtb_tblg.tblg_model(atoms_HC,parameters='popov')
    model_LC = pythtb_tblg.tblg_model(atoms_LC,parameters='popov')
    Gamma = [0,   0,   0]
    K = [2/3,1/3,0]
    Kprime = [1/3,2/3,0]
    M = [1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=40
    (k_vec,k_dist,k_node) = model_HC.k_path(sym_pts,nk)
    
    solve_dict = {'cupy':False,
                  'sparse':False}
    model_HC.set_solver(solve_dict)
    model_LC.set_solver(solve_dict)
    
    model_HC.solve_all(k_vec)
    model_LC.solve_all(k_vec)
    
    evals_HC = model_HC.get_eigenvalues()
    evals_LC = model_LC.get_eigenvalues()

    title = "buckled monolayer band structure"
    colors=['black','red']
    labels=['high Corr','low Corr']
    plot_bands(colors,labels,[evals_HC],title=title,erange=0.1,figname='buckled_monolayersHC.png')
    plot_bands(colors,labels,[evals_LC],title=title,erange=1.0,figname='buckled_monolayersLC.png')

