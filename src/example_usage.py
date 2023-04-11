# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:03:29 2023

@author: danpa
"""

import pythtb_tblg

if __name__ =="__main__":
    import flatgraphene as fg

    Gamma = [0,   0,   0]
    K = [-2/3,1/3,0]
    Kprime = [-1/3,-1/3,0]
    M = [-1/2,0,0]
    sym_pts=[K,Gamma,M,Kprime]
    nk=40
    lattice_constant = 2.4545
    twist = 16.43
    
    #create geometry
    p_found, q_found, theta_comp = fg.twist.find_p_q(twist)
    atoms=fg.twist.make_graphene(cell_type='hex',n_layer=2,
                                  p=p_found,q=q_found,lat_con=lattice_constant,sym=["B","Ti"],
                                        mass=[12.01,12.02],sep=3.444,h_vac=3)
    
    #example usage
    #parameters = [popov,letb,nam koshino], only popov available now
    model = pythtb_tblg.tblg_model(atoms,parameters='popov')
    model.set_solver( {'cupy':True,
                    'sparse':False,
                    'writeout':'test_BandCalc',
                    'restart':True,
                    #if sparse:
                    "fermi energy":-4.51,
                    "num states":30})
    
    (k_vec,k_dist,k_node) = model.k_path(sym_pts,nk)
    model.solve_all(k_vec)
    fermi_index = model._norbs//2
    evals = model.get_eigenvales(band_index=":",kpoints=":")
    evec = model.get_eigenvectors(band_index=":",kpoints=":")
    model.save("testname")
    model.load("testname.json")
    