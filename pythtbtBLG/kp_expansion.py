# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:13:29 2023

@author: danpa
"""

import numpy as np
import flatgraphene as fg
from scipy.sparse import csr_matrix
#import fast_scalar_sparse

def rot_mat_z3d(theta):
    """
    Generates a 3D rotation matrix which results in rotation of
    theta about the z axis
    """
    rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    return rot_mat
def rot_mat_z(theta):
    """
    Generates a 3D rotation matrix which results in rotation of
    theta about the z axis
    """
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    return rot_mat

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
    
def part_hamiltonian(atoms):
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
    return a_top_ind,a_bot_ind,b_top_ind,b_bot_ind

def get_sub_ham(full_ham,sl_ind1,sl_ind2):
    #figure out more efficient way to do this
    na = len(sl_ind1)
    nb = len(sl_ind2)
    sub_ham = np.zeros((na,nb),dtype=complex)
    for i in range(na):
        for j in range(nb):
            sub_ham[i,j] = full_ham[sl_ind1[i],sl_ind2[j]]
    return sub_ham
    
def downfold_ham(tb_model,kpoints,kcutoff=10):
    
    cell = tb_model.atoms.get_cell()
    recip_cell = cell.reciprocal()
    scale_atoms = atoms.get_scaled_positions()
    recip_lattice = np.fft.fft(scale_atoms)
    pos = tb_model.atoms.positions
    kdist = np.linalg.norm(recip_lattice)
    kdist = np.where(kdist<kcutoff)
    sort_ind = np.argsort(kdist)
    #recip_lattice = recip_lattice[sort_ind,:]
    N=2
    waven=(2*N+1)**2
    recip_lattice=np.array(np.zeros((waven, 2)))
    X,Y = np.meshgrid(np.arange(-N,N+1,1),np.arange(-N,N+1,1)) #reciprocal lattice
    recip_lattice[:,0] = X.flatten()
    recip_lattice[:,1] = Y.flatten()
    
    tb_model.solve_all(recip_lattice) #copy/rotate to get all wf in hex. bz
    lp_eigvals = tb_model.get_eigenvalues()
    lp_eigvec = tb_model.get_eigenvectors()
    tb_model.solve_all(kpoints)
    eigvals = tb_model.get_eigenvalues()
    eigvec= tb_model.get_eigenvectors()
    npw = np.shape(recip_lattice)[0]
    KP_Ham = np.zeros((int(4*npw),int(4*npw),np.shape(kpoints)[0]),dtype=complex)
    fermi_ind = np.shape(lp_eigvals)[0]//2
    #a_top_ind,a_bot_ind,b_top_ind,b_bot_ind = part_hamiltonian(tb_model.atoms)
    sublat_layer_ind = np.squeeze([part_hamiltonian(tb_model.atoms)])
    for h in range(np.shape(kpoints)[0]):
        ksc = kpoints[h,:]
        Hmatrix,H_row,H_col,Smatrix,S_row,S_col = tb_model.solver.gen_ham(ksc) #supercell momentum
        full_ham = csr_matrix((Hmatrix,(H_row,H_col)),shape=(tb_model.solver.norbs,tb_model.solver.norbs)).toarray()
        for i in range(npw):
            for j in range(npw):
                for k in range(4): #iterate over sublattices and layers
                    #i think this needs to be integrated, check if doing properly
                    #psi_p = np.mean(np.exp(1j*pos@recip_cell.T)*lp_eigvec[sublat_layer_ind[k],:,i],axis=1 )
                    psi_p = lp_eigvec[sublat_layer_ind[k],fermi_ind,i]
                    tmp_ham=np.zeros((4,4),dtype=complex)
                    for l in range(4):
                        #psi_q = np.mean(np.exp(1j*pos@recip_cell.T)*lp_eigvec[sublat_layer_ind[l],:,j],axis=1)
                        psi_q = lp_eigvec[sublat_layer_ind[l],fermi_ind,j]
                        sub_ham = get_sub_ham(full_ham,sublat_layer_ind[k],sublat_layer_ind[l])
                        #sub_ham = full_ham[sublat_layer_ind[k],sublat_layer_ind[l]]
                        tmp_ham[k,l] = \
                            psi_q.conj().T @ sub_ham @ psi_p
                            
                KP_Ham[int(4*i):int(4*(i+1)),int(4*j):int(4*(j+1)),h] = tmp_ham
            
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
    #kvec,dk_, knode = k_path(sym_pts,nk)
    #kvec = fast_scalar_sparse.k_uniform_mesh((20,20,1))

    theta=16.43
    lattice_constant = 2.46
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta)
    atoms=twist.make_graphene(cell_type='hex',n_layer=2,
                                      p=p_found,q=q_found,lat_con=lattice_constant,sym=["B","Ti"],
                                            mass=[12.01,12.02],sep=3.444,h_vac=3)
    # G = 2*np.pi/3/lattice_constant * np.array([[1,np.sqrt(3)],
    #                                            [1,-np.sqrt(3)]])
    
    # Gred = np.array([[1,0],
    #                  [0,1]])
    # Gtop = rot_mat_z(-theta/2*np.pi/180)@Gred
    # Gbot = rot_mat_z(theta/2*np.pi/180)@Gred
    # line = np.linspace(0,1,10)
    # xyz,ind = get_top_layer(atoms,scaled=True)
    # plt.scatter(xyz[:,0],xyz[:,1])
    # plt.plot((Gtop[0,0])*line,Gtop[0,1]*line,c='blue')
    # plt.plot((Gtop[1,0])*line,Gtop[1,1]*line,c='blue')
    # plt.plot((Gbot[0,0])*line,Gbot[0,1]*line,c='red')
    # plt.plot((Gbot[1,0])*line,Gbot[1,1]*line,c='red')
    # plt.show()
    
    # xy = atoms.positions[:,:2]
    # kxy = np.fft.fft(xy)

    # plt.scatter(kxy[:,0],kxy[:,1],s=5)
    # plt.show()
    
    model = pythtb_tblg.tblg_model(atoms,parameters='popov')
    model.set_solver( {'cupy':False,
                        'sparse':False})
    
    Gamma = [0,   0,   0]
    K = [-2/3,1/3,0]
    Kprime = [-1/3,-1/3,0]
    M = [-1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=40
    (k_vec,k_dist,k_node) = model.k_path(sym_pts,nk)
    KP_ham = downfold_ham(model,k_vec)
    print(np.shape(KP_ham))
    eigvals = np.zeros((np.shape(KP_ham)[0],nk))
    for i in range(nk):
        eigvals[:,i] = np.linalg.eigvalsh(KP_ham[:,:,i])
    plt.imshow(KP_ham[:,:,0].real)
    plt.show()
    for j in range(np.shape(eigvals)[0]):
        plt.plot(eigvals[j,:])
    plt.show()
        
    