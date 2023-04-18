# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:02:37 2023

@author: danpa
"""
import numpy as np
import flatgraphene as fg
import subprocess
from datetime import datetime
import os
import ase.io
import json
import glob
import matplotlib.pyplot as plt
import argparse
import time
import pythtb_tblg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--directory',type=str,default=False)
    parser.add_argument('-t','--theta',  type=str,default=False)
    parser.add_argument('-o','--output',  type=str,default=False)
    parser.add_argument('-od','--output_dir',type=str,default='band_calc'+str(hash(time.time())))
    parser.add_argument('-i','--index',type=str,default="final")
    parser.add_argument('-m','--model',type=str,default='popov')
    parser.add_argument('-s','--separation',type=str,default="3.35")
    parser.add_argument('-c','--calc_type',type=str,default="bands")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        subprocess.call("mkdir "+args.output_dir,shell=True)
    lattice_constant = 2.4545
    if args.directory:
        calc_dir=args.directory
        # with open(os.path.join(calc_dir,"model.json"),"r") as f:
        #    relax_model = json.load(f)
        if args.index=="final":
            atoms = ase.io.read(calc_dir,format="lammps-dump-text")
            atoms_index = [0]
        elif args.index=="all":
            atoms = ase.io.read(calc_dir,format="lammps-dump-text",index=":")
            atoms_index = np.arange(0,len(atoms),5)
        else:
            try:
                atoms = ase.io.read(calc_dir,format="lammps-dump-text",index=int(args.index))
            except:
                atoms = ase.io.read(calc_dir,index=int(args.index))
            atoms_index=[0]

        for a in atoms:
            new_cell = atoms.get_cell()
            new_cell[1,0]=np.abs(new_cell[1,0])
            atoms.set_cell(new_cell)
            masses=[]
            dump_mass = atoms.get_masses()
            for m in dump_mass:
                if np.isclose(m,dump_mass[0],rtol=1e-4):
                    masses.append(12.01)
                else:
                    masses.append(12.02)
            atoms.set_masses(np.array(masses))
        
    elif args.theta:
        theta=float(args.theta)
        label="flat"
        atoms = fg.get_twist_geom(theta,float(args.separation))

    
    #setup model 
    model = pythtb_tblg.tblg_model(atoms,parameters=args.model)
    model.set_solver( {'cupy':True,
                        'sparse':False,
                        'writeout':args.output_dir,
                        'restart':False,
                        'ngpu':6,
                        #if sparse
                        "fermi energy":-4.51,
                        "num states":14})

    if args.calc_type=='bands':
        Gamma = [0,   0,   0]
        K = [2/3,1/3,0]
        Kprime = [1/3,2/3,0]
        M = [1/2,0,0]
        sym_pts=[K,Gamma,M,Kprime]
        nk=40
        (k_vec,k_dist,k_node) = model.k_path(sym_pts,nk)
        model.solve_all(k_vec)
        
    elif args.calc_type=='berry_flux':
        Gamma = [0,0,0]
        mesh = (15,15,1)
        my_array_1=pythtb_tblg.wf_array(model,mesh)
        my_array_1.solve_on_grid(Gamma)
        fermi_ind = model.num_eigvals//2
        berry_phases = np.zeros((model.num_eigvals,mesh[0]-1,mesh[1]-1))
        for i in range(model.num_eigvals):
            berry_phases[i,:,:] = my_array_1.berry_flux([i],individual_phases=True)
        np.savez(os.path.join(args.output_dir,'berry_flux'),berry_flux = berry_phases)
        
            
    elif args.calc_type=='winding_number':
        circ_step=31
        circ_center=np.array([1.0/3.0,2.0/3.0,0])
        circ_radius=0.05
        # one-dimensional wf_array to store wavefunctions on the path
        w_circ=pythtb_tblg.wf_array(model,[circ_step])
        # now populate array with wavefunctions
        kpoints = np.zeros((circ_step,3))
        for i in range(circ_step):
            # construct k-point coordinate on the path
            ang=2.0*np.pi*float(i)/float(circ_step-1)
            kpt=np.array([np.cos(ang)*circ_radius,np.sin(ang)*circ_radius,0])
            kpt+=circ_center
            kpoints[i,:] = kpt
            # find and store eigenvectors for this k-point
        w_circ.solve_on_path(kpoints)
        # make sure that first and last points are the same
        w_circ[-1]=w_circ[0]
        fermi_ind = model.num_eigvals//2
        # compute Berry phase along circular path
        print("Berry phase along circle with radius: ",circ_radius)
        print("  centered at k-point: ",circ_center)
        print("  for band 0 equals    : ", w_circ.berry_phase([fermi_ind-1.,fermi_ind],0))
        print("  for band 1 equals    : ", w_circ.berry_phase([fermi_ind+1,fermi_ind+2],0))
        print("  for both bands equals: ", w_circ.berry_phase([fermi_ind-1,
                                        fermi_ind,fermi_ind+1,fermi_ind+2],0))
        
        np.savez(os.path.join(args.output_dir,'winding_number',circ_center=circ_center,
                              homo_winding = w_circ.berry_phase([fermi_ind-1.,fermi_ind],0),
                              lumo_winding = w_circ.berry_phase([fermi_ind+1,fermi_ind+2],0),
                              total_winding = w_circ.berry_phase([fermi_ind-1,
                                        fermi_ind,fermi_ind+1,fermi_ind+2],0)))
        
    elif args.calc_type=='sublattice_polarization':
        my_array_1=pythtb_tblg.wf_array(model,[15,15,1])
        my_array_1.solve_on_grid([0,0,0])
        fermi_ind = model.num_eigvals//2
        phi_x = my_array_1.berry_phase([fermi_ind-1,
                                        fermi_ind],0,contin=True)
        phi_y = my_array_1.berry_phase([fermi_ind-1,
                                        fermi_ind],1,contin=True)
        
        