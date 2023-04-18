# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:09:21 2023

@author: danpa
"""
import pythtb_tblg
import numpy as np
import flatgraphene as fg
import matplotlib.pyplot as plt

def get_top_layer(atoms,scaled=True):
    if scaled:
        xyz = atoms.get_scaled_positions()
    else:
        xyz=atoms.positions
    
    tags=np.array(atoms.get_chemical_symbols())
    top_pos_ind=np.where(tags!=tags[0]) #[0]
    top_pos=xyz[top_pos_ind[0],:]
    
    return top_pos,top_pos_ind

def get_bottom_layer(atoms,scaled=True):
    if scaled:
        xyz = atoms.get_scaled_positions()
    else:
        xyz=atoms.positions
    
    tags=np.array(atoms.get_chemical_symbols())
    bot_pos_ind=np.where(tags==tags[0]) #[0]
    bot_pos=xyz[bot_pos_ind[0],:]
    
    return bot_pos,bot_pos_ind

def rot_hamiltonian(atoms):
    #rotate these positions to match lattice vectors
    sublat = np.array(atoms.sublattice)
    top_pos,top_pos_ind = get_top_layer(atoms) 
    top_pos = top_pos.copy()
    top_atoms = atoms[top_pos_ind]
    bot_pos, bot_pos_ind = get_bottom_layer(atoms)
    bot_pos = bot_pos.copy()
    bot_atoms = atoms[bot_pos_ind]
    # need to make sure that A,B are in same cell and are in same position in array
    sub_lat_color = np.zeros(len(sublat))
    sub_lat_color[np.where(sublat=='Ar')] = 1
    sub_lat_color[np.where(sublat=='B')] = -1
    a_top_ind = np.where(sublat[top_pos_ind]=='Ar')[0]
    a_bot_ind = np.where(sublat[bot_pos_ind]=='Ar')[0]
    b_top_ind = np.where(sublat[top_pos_ind]=='B')[0]
    b_bot_ind = np.where(sublat[bot_pos_ind]=='B')[0]
    perm = np.concatenate((a_top_ind,a_bot_ind,b_top_ind,b_bot_ind))
    #print(site_labels)
    #plt.scatter(top_pos[:,0],top_pos[:,1],c=sub_lat_color[top_pos_ind])
    #plt.colorbar()
    #plt.show()
    #plt.clf()
    #labels = {"sublattice":sublat,"layer":}
    return a_top_ind,a_bot_ind,b_top_ind,b_bot_ind

def sublat_polarization(model):
    my_array_1=pythtb_tblg.wf_array(model,[15,15,1])
    my_array_1.solve_on_grid([0,0,0])
    cell = model.atoms.get_cell()
    R1 = cell[:,0]
    R2 = cell[:,1]
    sublat_ind = rot_hamiltonian(model.atoms)
    fermi_ind = model.num_eigvals//2
    P = np.zeros((4,3))
    for i in range(4):
        phi_x = my_array_1.berry_phase([fermi_ind-1,
                                        fermi_ind],0,contin=True,subset=sublat_ind[i])
        phi_y = my_array_1.berry_phase([fermi_ind-1,
                                        fermi_ind],1,contin=True,subset=sublat_ind[i])
        
        P[i,:] = phi_x*R1 + phi_y*R2
    Pa1= P[0,:]
    Pa2= P[1,:]
    Pb1= P[2,:]
    Pb2 = P[3,:]
    return Pa1,Pa2,Pb1,Pb2

if __name__=="__main__":
      
    lattice_constant = 2.4545
    twist = 2.88
     
    #create geometry
    p_found, q_found, theta_comp = fg.twist.find_p_q(twist)
    atoms=fg.twist.make_graphene(cell_type='hex',n_layer=2,
                   p=p_found,q=q_found,lat_con=lattice_constant,sym=["B","Ti"],
                         mass=[12.01,12.02],sep=3.444,h_vac=3)
     
    #example usage
    #parameters = [popov,letb,nam koshino], only popov available now
    model = pythtb_tblg.tblg_model(atoms,parameters='popov')
    Pa1,Pa2,Pb1,Pb2 = sublat_polarization(model)
    print(Pa1,Pa2,Pb1,Pb2)