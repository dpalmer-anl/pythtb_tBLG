# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:03:29 2023

@author: danpa
"""

import pythtb_tblg
import numpy as np

if __name__ =="__main__":
    import flatgraphene as fg
    import matplotlib.pyplot as plt
    test=False
    hpc_test=True
    berry_phase=False

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
    Gamma = [0,   0,   0]
    K = [2/3,1/3,0]
    Kprime = [1/3,2/3,0]
    M = [1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=40
    (k_vec,k_dist,k_node) = model.k_path(sym_pts,nk)
    if test:
        model.set_solver( {'cupy':False,
                        'sparse':False,
                        'writeout':'test_BandCalc',
                        'restart':False,
                        'ngpu':6,
                        #if sparse
                        "fermi energy":-4.51,
                        "num states":30})
    
        model.solve_all(k_vec)
        fermi_index = model._norb//2
        evals = model.get_eigenvalues()
        evec = model.get_eigenvectors()
        
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
        ax.set_title("tblg band structure")
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy")
    
        nbands = np.shape(evals)[0]
        efermi = np.mean([evals[nbands//2,0],evals[(nbands-1)//2,0]])
        fermi_ind = nbands//2
        # plot first and second band
        for n in range(np.shape(evals)[0]):
            if n == fermi_ind-1 or n==fermi_ind:
                ax.plot(k_dist,evals[n,:]-efermi,c='red')
            else:
                ax.plot(k_dist,evals[n,:]-efermi,c='black')
        
        # make an PDF figure of a plot
        fig.tight_layout()
        ax.set_ylim(-2,2)
        model.save_model("testname")
        # model.load("testname.json")
        model = pythtb_tblg.load_model('testname.npy')
        model.set_solver( {'cupy':False,
                        'sparse':False,
                        #'writeout':'test_BandCalc',
                        'restart':True,
                        #if sparse
                        "fermi energy":-4.51,
                        "num states":30})
        model.solve_all(k_vec)
    
    if hpc_test:
        model.set_solver( {'cupy':True,
                        'sparse':False,
                        'writeout':'test_BandCalc',
                        'restart':False,
                        'ngpu':6,
                        #if sparse
                        "fermi energy":-4.51,
                        "num states":14})
    
        model.solve_all(k_vec)
        fermi_index = 1
        evals = model.get_eigenvalues()
        evec = model.get_eigenvectors()
        
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
        ax.set_title("tblg band structure")
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy")
    
        nbands = np.shape(evals)[0]
        efermi = np.mean([evals[nbands//2,0],evals[(nbands-1)//2,0]])
        fermi_ind = nbands//2
        # plot first and second band
        for n in range(np.shape(evals)[0]):
            if n == fermi_ind-1 or n==fermi_ind:
                ax.scatter(k_dist,evals[n,:]-efermi,c='red')
            else:
                ax.scatter(k_dist,evals[n,:]-efermi,c='black')
        
        # make an PDF figure of a plot
        fig.tight_layout()
        
        # model.set_solver( {'cupy':True,
        #                 'sparse':False,
        #                 #'writeout':'test_BandCalc',
        #                 'restart':False,
        #                 #if sparse
        #                 "fermi energy":-4.51,
        #                 "num states":30})
    
        # model.solve_all(k_vec)
        
        # model.set_solver( {'cupy':True,
        #                 'sparse':True,
        #                 #'writeout':'test_BandCalc',
        #                 'restart':False,
        #                 #if sparse
        #                 "fermi energy":-4.51,
        #                 "num states":30})
    
        # model.solve_all(k_vec)
        
    if berry_phase:
        model.set_solver( {'cupy':False,
                        'sparse':False,
                        #'writeout':'test_BandCalc',
                        'restart':True,
                        #if sparse
                        "fermi energy":-4.51,
                        "num states":14})
        
        circ_step=31
        circ_center=np.array([1.0/3.0,2.0/3.0,0])
        circ_radius=0.05
        # one-dimensional wf_array to store wavefunctions on the path
        w_circ=pythtb_tblg.wf_array(model,[circ_step])
        # now populate array with wavefunctions
        for i in range(circ_step):
            # construct k-point coordinate on the path
            ang=2.0*np.pi*float(i)/float(circ_step-1)
            kpt=np.array([np.cos(ang)*circ_radius,np.sin(ang)*circ_radius,0])
            kpt+=circ_center
            # find and store eigenvectors for this k-point
            w_circ.solve_on_one_point(kpt,i)
        # make sure that first and last points are the same
        w_circ[-1]=w_circ[0]
        fermi_ind = atoms.get_global_number_of_atoms()//2
        # compute Berry phase along circular path
        print("Berry phase along circle with radius: ",circ_radius)
        print("  centered at k-point: ",circ_center)
        print("  for band 0 equals    : ", w_circ.berry_phase([fermi_ind-1.,fermi_ind],0))
        print("  for band 1 equals    : ", w_circ.berry_phase([fermi_ind+1,fermi_ind+2],0))
        print("  for both bands equals: ", w_circ.berry_phase([fermi_ind-1,
                                        fermi_ind,fermi_ind+1,fermi_ind+2],0))
        
        
    
