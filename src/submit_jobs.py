# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:52:14 2023

@author: danpa
"""

import pythtb_tblg
import numpy as np
import flatgraphene as fg
import matplotlib.pyplot as plt
import subprocess
from datetime import datetime

def submit_batch_file(executable,batch_options):

        sbatch_file="job"+str(hash(datetime.now()) )+".qsub"
        batch_copy = batch_options.copy()

        prefix="#BSUB "
        with open(sbatch_file,"w+") as f:
            #f.write("#!/bin/csh -vm \n \n")
            f.write("#!/bin/bash\n")
            f.write("# Begin LSF Directives\n")

            modules=batch_copy["modules"]

            for key in batch_copy:
                if key == "modules":
                    continue
                f.write(prefix+key+str(batch_copy[key])+"\n")

            for m in modules:
                f.write("module load "+m+"\n")
            
            #"jsrun -n1 -c21 -g1 -a1 -EOMP_NUM_THREADS=4 "+
            f.write("\nsource activate /ccs/home/dpalmer3/.conda/envs/my_env\n")
            #f.write("\nsource activate /ccs/proj/mat221/dpalmer3/conda_envs/summit/cupy-summit\n")
            f.write(executable)
        subprocess.call("bsub -L $SHELL "+sbatch_file,shell=True)
        
if __name__ =="__main__":
    berry_phase = True
    
    if berry_phase:
        batch_options = {
                "-P":" MAT221",
                "-W":" 24:00",
                "-nnodes":" 1",
                "-J":" structVary",
                "-o":" log.%J",
                "-e":" error.%J",
                "-q":" batch-hm",
                "-N":" dpalmer3@illinois.edu",
                "modules": ["gcc", "netlib-lapack","python","cuda"]}
        
        #outatoms_dir = 'outplane_StructVary_0_99.traj' out of plane
        #inatoms_dir = 'inplane_StructVary_0_99.traj' inplane
        atoms_dir = 'interpUR_StructVary_0_99.traj' #full
        tbmodel = 'popov'
        theta = 0.99
        num_ind = 10
        label = "bandsStructVary_t_"+str(theta)

        theta_str = str(theta).replace(".","_")
        outdir = '../../tblg_db/band_data_interpUR_'+theta_str+'_'+tbmodel
        for i in range(num_ind):
            executable = "jsrun -n1 -c42 -a1 -r1 -b rs python run_python_bands.py -d "\
            +atoms_dir+" -o "+label+"_"+str(i)+' -i '+str(i)+' -m '+tbmodel
            +" -od "+outdir+'-c berry'
            submit_batch_file(executable,batch_options)
