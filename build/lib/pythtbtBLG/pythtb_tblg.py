from __future__ import print_function

# PythTB python tight binding module.
# September 20th, 2022
__version__='1.8.0'

# Copyright 2010, 2012, 2016, 2017, 2022 by Sinisa Coh and David Vanderbilt
#
# This file is part of PythTB.  PythTB is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# PythTB is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# A copy of the GNU General Public License should be available
# alongside this source in a file named gpl-3.0.txt.  If not,
# see <http://www.gnu.org/licenses/>.
#
# PythTB is availabe at http://www.physics.rutgers.edu/pythtb/

import numpy as np # numerics for matrices
import sys # for exiting
import copy # for deepcopying
import solverUtils
import os
from scipy.optimize import linear_sum_assignment
import h5py
import glob
import re
import ase

class tblg_model(object):
    r"""
    This is the main class of the PythTB package which contains all
    information for the tight-binding model.

    :param dim_k: Dimensionality of reciprocal space, i.e., specifies how
      many directions are considered to be periodic.

    :param dim_r: Dimensionality of real space, i.e., specifies how many
      real space lattice vectors there are and how many coordinates are
      needed to specify the orbital coordinates.

    .. note::

      Parameter *dim_r* can be larger than *dim_k*! For example,
      a polymer is a three-dimensional molecule (one needs three
      coordinates to specify orbital positions), but it is periodic
      along only one direction. For a polymer, therefore, we should
      have *dim_k* equal to 1 and *dim_r* equal to 3. See similar example
      here: :ref:`trestle-example`.

    :param lat: Array containing lattice vectors in Cartesian
      coordinates (in arbitrary units). In example the below, the first
      lattice vector has coordinates [1.0,0.5] while the second
      one has coordinates [0.0,2.0].  By default, lattice vectors
      are an identity matrix.

    :param orb: Array containing reduced coordinates of all
      tight-binding orbitals. In the example below, the first
      orbital is defined with reduced coordinates [0.2,0.3]. Its
      Cartesian coordinates are therefore 0.2 times the first
      lattice vector plus 0.3 times the second lattice vector.
      If *orb* is an integer code will assume that there are these many
      orbitals all at the origin of the unit cell.  By default
      the code will assume a single orbital at the origin.

    :param per: This is an optional parameter giving a list of lattice
      vectors which are considered to be periodic. In the example below,
      only the vector [0.0,2.0] is considered to be periodic (since
      per=[1]). By default, all lattice vectors are assumed to be
      periodic. If dim_k is smaller than dim_r, then by default the first
      dim_k vectors are considered to be periodic.

    :param nspin: Number of explicit spin components assumed for each
      orbital in *orb*. Allowed values of *nspin* are *1* and *2*. If
      *nspin* is 1 then the model is spinless, if *nspin* is 2 then it
      is explicitly a spinfull model and each orbital is assumed to
      have two spin components. Default value of this parameter is
      *1*.  Of course one can make spinfull calculation even with
      *nspin* set to 1, but then the user must keep track of which
      orbital corresponds to which spin component.

    Example usage::

       # Creates model that is two-dimensional in real space but only
       # one-dimensional in reciprocal space. Second lattice vector is
       # chosen to be periodic (since per=[1]). Three orbital
       # coordinates are specified.       
       tb = tb_model(1, 2,
                   lat=[[1.0, 0.5], [0.0, 2.0]],
                   orb=[[0.2, 0.3], [0.1, 0.1], [0.2, 0.2]],
                   per=[1])

    """

    def __init__(self,atoms,parameters='popov',nspin=1):
        
        self._dim_k = 3
        self._dim_r = 3
        self._per=list(range(self._dim_k))
        self.atoms = atoms
        self.parameters=parameters
        self._orb = self.atoms.positions
        self._lat = self.atoms.get_cell()
        self._norb = atoms.get_global_number_of_atoms()
        # remember number of spin components
        if nspin not in [1,2]:
            raise Exception("\n\nWrong value of nspin, must be 1 or 2!")
        self._nspin=nspin
        self.eigenvalues = None
        self.eigenvectors = None
        self.read_data = False
        # by default, assume model did not come from w90 object and that
        # position operator is diagonal
        self._assume_position_operator_diagonal=True
        self.solve_dict = {'cupy':False,
                    'sparse':False,
                    'writeout':None,
                    'restart':False,
                    #if sparse:
                    "fermi energy":-4.51,
                    "num states":30}

    def _val_to_block(self,val):
        """If nspin=2 then returns a 2 by 2 matrix from the input
        parameters. If only one real number is given in the input then
        assume that this is the diagonal term. If array with four
        elements is given then first one is the diagonal term, and
        other three are Zeeman field direction. If given a 2 by 2
        matrix, just return it.  If nspin=1 then just returns val."""
        # spinless case
        if self._nspin==1:
            return val
        # spinfull case
        elif self._nspin==2:
            # matrix to return
            ret=np.zeros((2,2),dtype=complex)
            # 
            use_val=np.array(val)
            # only one number is given
            if use_val.shape==():
                ret[0,0]+=use_val
                ret[1,1]+=use_val
            # if four numbers are given
            elif use_val.shape==(4,):
                # diagonal
                ret[0,0]+=use_val[0]
                ret[1,1]+=use_val[0]
                # sigma_x
                ret[0,1]+=use_val[1]
                ret[1,0]+=use_val[1]
                # sigma_y
                ret[0,1]+=use_val[2]*(-1.0j)
                ret[1,0]+=use_val[2]*( 1.0j)
                # sigma_z
                ret[0,0]+=use_val[3]
                ret[1,1]+=use_val[3]*(-1.0)        
            # if 2 by 2 matrix is given
            elif use_val.shape==(2,2):
                return use_val
            else:
                raise Exception(\
"""\n
Wrong format of the on-site or hopping term. Must be single number, or
in the case of a spinfull model can be array of four numbers or 2x2
matrix.""")            
            return ret        
    def save_model(self,dir_name):
        data= {'positions':self.atoms.positions,
                 'cell':self.atoms.get_cell(),
                 'layer_tags':list(self.atoms.symbols),
                 'parameters':self.parameters,
                 'solve_dict':self.solve_dict}
        np.save(dir_name,data)
        
        
    def set_solver(self,solve_dict):
        for k in solve_dict.keys():    
            self.solve_dict[k] = solve_dict[k]
        self.solver = solverUtils.solver(self)
        return None
        
    def get_num_orbitals(self):
        "Returns number of orbitals in the model."
        return self.atoms.get_global_number_of_atoms()
    
    def get_eigenvalues(self,k_list='all'):
        if type(self.eigenvalues)==np.ndarray:
            return self.eigenvalues
        if self.read_data:
            if type(k_list)==str:
                if k_list=='all':
                    k_list = self.kpoints.copy()
            evals,evec,kpoints = load_dataHDF5(self.solve_dict['writeout'],k_list)
            self.eigenvalues = evals
            self.eigenvectors = evec
            self.read_data = False
            return self.eigenvalues
        
    def get_eigenvectors(self,k_list='all'):
        if type(self.eigenvectors)==np.ndarray:
            return self.eigenvectors
        
        if self.read_data:
            if k_list!='all':
                evals,evec,kpoints = load_dataHDF5(self.solve_dict['writeout'],k_list=k_list)
            else:
                evals,evec,kpoints = load_dataHDF5(self.solve_dict['writeout'])
            self.eigenvalues = evals
            self.eigenvectors = evec
            self.kpoints = kpoints
            self.read_data = False
            return self.eigenvectors
                

    def solve_all(self,k_list=None):
        self.kpoints = k_list.copy()
        self.solver.solve_all(k_list)
        # indices of eval are [band,kpoint] for evec are [orbital,band,kpoint,(spin)]

    def solve_one(self,k_point=None):
        r"""

        Similar to :func:`pythtb.tb_model.solve_all` but solves tight-binding
        model for only one k-vector.

        """

        self.solve_all(k_list=np.array([k_point]))

    def k_uniform_mesh(self,mesh_size):
        r""" 
        Returns a uniform grid of k-points that can be passed to
        passed to function :func:`pythtb.tb_model.solve_all`.  This
        function is useful for plotting density of states histogram
        and similar.

        Returned uniform grid of k-points always contains the origin.

        :param mesh_size: Number of k-points in the mesh in each
          periodic direction of the model.
          
        :returns:

          * **k_vec** -- Array of k-vectors on the mesh that can be
            directly passed to function  :func:`pythtb.tb_model.solve_all`.

        Example usage::
          
          # returns a 10x20x30 mesh of a tight binding model
          # with three periodic directions
          k_vec = my_model.k_uniform_mesh([10,20,30])
          # solve model on the uniform mesh
          my_model.solve_all(k_vec)
        
        """
        
        # get the mesh size and checks for consistency
        use_mesh=np.array(list(map(round,mesh_size)),dtype=int)
        if use_mesh.shape!=(self._dim_k,):
            print(use_mesh.shape)
            raise Exception("\n\nIncorrect size of the specified k-mesh!")
        if np.min(use_mesh)<=0:
            raise Exception("\n\nMesh must have positive non-zero number of elements.")

        # construct the mesh
        if self._dim_k==1:
            # get a mesh
            k_vec=np.mgrid[0:use_mesh[0]]
            # normalize the mesh
            norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
            norm=norm.reshape(use_mesh.tolist()+[1])
            norm=norm.transpose([1,0])
            k_vec=k_vec/norm
            # final reshape
            k_vec=k_vec.transpose([1,0]).reshape([use_mesh[0],1])
        elif self._dim_k==2:
            # get a mesh
            k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1]]
            # normalize the mesh
            norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
            norm=norm.reshape(use_mesh.tolist()+[2])
            norm=norm.transpose([2,0,1])
            k_vec=k_vec/norm
            # final reshape
            k_vec=k_vec.transpose([1,2,0]).reshape([use_mesh[0]*use_mesh[1],2])
        elif self._dim_k==3:
            # get a mesh
            k_vec=np.mgrid[0:use_mesh[0],0:use_mesh[1],0:use_mesh[2]]
            # normalize the mesh
            norm=np.tile(np.array(use_mesh,dtype=float),use_mesh)
            norm=norm.reshape(use_mesh.tolist()+[3])
            norm=norm.transpose([3,0,1,2])
            k_vec=k_vec/norm
            # final reshape
            k_vec=k_vec.transpose([1,2,3,0]).reshape([use_mesh[0]*use_mesh[1]*use_mesh[2],3])
        else:
            raise Exception("\n\nUnsupported dim_k!")

        return k_vec

    def k_path(self,sym_pts,nk,report=False):
        r"""
    
        Interpolates a path in reciprocal space between specified
        k-points.  In 2D or 3D the k-path can consist of several
        straight segments connecting high-symmetry points ("nodes"),
        and the results can be used to plot the bands along this path.
        
        The interpolated path that is returned contains as
        equidistant k-points as possible.
    
        :param kpts: Array of k-vectors in reciprocal space between
          which interpolated path should be constructed. These
          k-vectors must be given in reduced coordinates.  As a
          special case, in 1D k-space kpts may be a string:
    
          * *"full"*  -- Implies  *[ 0.0, 0.5, 1.0]*  (full BZ)
          * *"fullc"* -- Implies  *[-0.5, 0.0, 0.5]*  (full BZ, centered)
          * *"half"*  -- Implies  *[ 0.0, 0.5]*  (half BZ)
    
        :param nk: Total number of k-points to be used in making the plot.
        
        :param report: Optional parameter specifying whether printout
          is desired (default is True).

        :returns:

          * **k_vec** -- Array of (nearly) equidistant interpolated
            k-points. The distance between the points is calculated in
            the Cartesian frame, however coordinates themselves are
            given in dimensionless reduced coordinates!  This is done
            so that this array can be directly passed to function
            :func:`pythtb.tb_model.solve_all`.

          * **k_dist** -- Array giving accumulated k-distance to each
            k-point in the path.  Unlike array *k_vec* this one has
            dimensions! (Units are defined here so that for an
            one-dimensional crystal with lattice constant equal to for
            example *10* the length of the Brillouin zone would equal
            *1/10=0.1*.  In other words factors of :math:`2\pi` are
            absorbed into *k*.) This array can be used to plot path in
            the k-space so that the distances between the k-points in
            the plot are exact.

          * **k_node** -- Array giving accumulated k-distance to each
            node on the path in Cartesian coordinates.  This array is
            typically used to plot nodes (typically special points) on
            the path in k-space.
    
        Example usage::
    
          # Construct a path connecting four nodal points in k-space
          # Path will contain 401 k-points, roughly equally spaced
          path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
          (k_vec,k_dist,k_node) = my_model.k_path(path,401)
          # solve for eigenvalues on that path
          evals = tb.solve_all(k_vec)
          # then use evals, k_dist, and k_node to plot bandstructure
          # (see examples)
        
        """
    
        k_list=np.array(sym_pts)
    
        # number of nodes
        n_nodes=k_list.shape[0]
    
        mesh_step = nk//(n_nodes-1)
        mesh = np.linspace(0,1,mesh_step)
        step = (np.arange(0,mesh_step,1)/mesh_step)
    
        kvec = np.zeros((0,3))
        knode = np.zeros(n_nodes)
        for i in range(n_nodes-1):
           n1 = k_list[i,:]
           n2 = k_list[i+1,:]
           diffq = np.outer((n2 - n1),  step).T + n1
    
           dn = np.linalg.norm(n2-n1)
           knode[i+1] = dn + knode[i]
           if i==0:
              kvec = np.vstack((kvec,diffq))
           else:
              kvec = np.vstack((kvec,diffq))
        kvec = np.vstack((kvec,k_list[-1,:]))
    
        dk_ = np.zeros(np.shape(kvec)[0])
        for i in range(1,np.shape(kvec)[0]):
           dk_[i] = np.linalg.norm(kvec[i,:]-kvec[i-1,:]) + dk_[i-1]
    
        return (kvec,dk_, knode)

    def ignore_position_operator_offdiagonal(self):
        """Call to this function enables one to approximately compute
        Berry-like objects from tight-binding models that were
        obtained from Wannier90."""  
        self._assume_position_operator_diagonal=True

    def position_matrix(self, evec, dir):
        r"""

        Returns matrix elements of the position operator along
        direction *dir* for eigenvectors *evec* at a single k-point.
        Position operator is defined in reduced coordinates.

        The returned object :math:`X` is

        .. math::

          X_{m n {\bf k}}^{\alpha} = \langle u_{m {\bf k}} \vert
          r^{\alpha} \vert u_{n {\bf k}} \rangle

        Here :math:`r^{\alpha}` is the position operator along direction
        :math:`\alpha` that is selected by *dir*.

        :param evec: Eigenvectors for which we are computing matrix
          elements of the position operator.  The shape of this array
          is evec[band,orbital] if *nspin* equals 1 and
          evec[band,orbital,spin] if *nspin* equals 2.

        :param dir: Direction along which we are computing the center.
          This integer must not be one of the periodic directions
          since position operator matrix element in that case is not
          well defined.

        :returns:
          * **pos_mat** -- Position operator matrix :math:`X_{m n}` as defined 
            above. This is a square matrix with size determined by number of bands
            given in *evec* input array.  First index of *pos_mat* corresponds to
            bra vector (*m*) and second index to ket (*n*).

        Example usage::

          # diagonalizes Hamiltonian at some k-points
          (evals, evecs) = my_model.solve_all(k_vec,eig_vectors=True)
          # computes position operator matrix elements for 3-rd kpoint 
          # and bottom five bands along first coordinate
          pos_mat = my_model.position_matrix(evecs[:5,2], 0)

        See also this example: :ref:`haldane_hwf-example`,

        """

        # make sure specified direction is not periodic!
        if dir in self._per:
            raise Exception("Can not compute position matrix elements along periodic direction!")
        # make sure direction is not out of range
        if dir<0 or dir>=self._dim_r:
            raise Exception("Direction out of range!")
        
        # check if model came from w90
        if self._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        # get coordinates of orbitals along the specified direction
        pos_tmp=self._orb[:,dir]
        # reshape arrays in the case of spinfull calculation
        if self._nspin==2:
            # tile along spin direction if needed
            pos_use=np.tile(pos_tmp,(2,1)).transpose().flatten()
            # also flatten the state along the spin index
            evec_use=evec.reshape((evec.shape[0],evec.shape[1]*evec.shape[2]))                
        else:
            pos_use=pos_tmp
            evec_use=evec

        # position matrix elements
        pos_mat=np.zeros((evec_use.shape[0],evec_use.shape[0]),dtype=complex)
        # go over all bands
        for i in range(evec_use.shape[0]):
            for j in range(evec_use.shape[0]):
                pos_mat[i,j]=np.dot(evec_use[i].conj(),pos_use*evec_use[j])

        # make sure matrix is hermitian
        if np.max(pos_mat-pos_mat.T.conj())>1.0E-9:
            raise Exception("\n\n Position matrix is not hermitian?!")

        return pos_mat

    def position_expectation(self,evec,dir):
        r""" 

        Returns diagonal matrix elements of the position operator.
        These elements :math:`X_{n n}` can be interpreted as an
        average position of n-th Bloch state *evec[n]* along
        direction *dir*.  Generally speaking these centers are *not*
        hybrid Wannier function centers (which are instead
        returned by :func:`pythtb.tb_model.position_hwf`).
        
        See function :func:`pythtb.tb_model.position_matrix` for
        definition of matrix :math:`X`.

        :param evec: Eigenvectors for which we are computing matrix
          elements of the position operator.  The shape of this array
          is evec[band,orbital] if *nspin* equals 1 and
          evec[band,orbital,spin] if *nspin* equals 2.

        :param dir: Direction along which we are computing matrix
          elements.  This integer must not be one of the periodic
          directions since position operator matrix element in that
          case is not well defined.

        :returns:
          * **pos_exp** -- Diagonal elements of the position operator matrix :math:`X`.
            Length of this vector is determined by number of bands given in *evec* input 
            array.

        Example usage::

          # diagonalizes Hamiltonian at some k-points
          (evals, evecs) = my_model.solve_all(k_vec,eig_vectors=True)
          # computes average position for 3-rd kpoint 
          # and bottom five bands along first coordinate
          pos_exp = my_model.position_expectation(evecs[:5,2], 0)

        See also this example: :ref:`haldane_hwf-example`.

        """

        # check if model came from w90
        if self._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        pos_exp=self.position_matrix(evec,dir).diagonal()
        return np.array(np.real(pos_exp),dtype=float)

    def position_hwf(self,evec,dir,hwf_evec=False,basis="orbital"):
        r""" 

        Returns eigenvalues and optionally eigenvectors of the
        position operator matrix :math:`X` in basis of the orbitals
        or, optionally, of the input wave functions (typically Bloch
        functions).  The returned eigenvectors can be interpreted as
        linear combinations of the input states *evec* that have
        minimal extent (or spread :math:`\Omega` in the sense of
        maximally localized Wannier functions) along direction
        *dir*. The eigenvalues are average positions of these
        localized states.

        Note that these eigenvectors are not maximally localized
        Wannier functions in the usual sense because they are
        localized only along one direction.  They are also not the
        average positions of the Bloch states *evec*, which are
        instead computed by :func:`pythtb.tb_model.position_expectation`.

        See function :func:`pythtb.tb_model.position_matrix` for
        the definition of the matrix :math:`X`.

        See also Fig. 3 in Phys. Rev. Lett. 102, 107603 (2009) for a
        discussion of the hybrid Wannier function centers in the
        context of a Chern insulator.

        :param evec: Eigenvectors for which we are computing matrix
          elements of the position operator.  The shape of this array
          is evec[band,orbital] if *nspin* equals 1 and
          evec[band,orbital,spin] if *nspin* equals 2.

        :param dir: Direction along which we are computing matrix
          elements.  This integer must not be one of the periodic
          directions since position operator matrix element in that
          case is not well defined.

        :param hwf_evec: Optional boolean variable.  If set to *True* 
          this function will return not only eigenvalues but also 
          eigenvectors of :math:`X`. Default value is *False*.

        :param basis: Optional parameter. If basis="wavefunction", the hybrid
          Wannier function *hwf_evec* is returned in the basis of the input
          wave functions.  That is, the elements of hwf[i,j] give the amplitudes
          of the i-th hybrid Wannier function on the j-th input state.
          Note that option basis="bloch" is a synonym for basis="wavefunction".
          If basis="orbital", the elements of hwf[i,orb] (or hwf[i,orb,spin]
          if nspin=2) give the amplitudes of the i-th hybrid Wannier function on
          the specified basis function.  Default is basis="orbital".

        :returns:
          * **hwfc** -- Eigenvalues of the position operator matrix :math:`X`
            (also called hybrid Wannier function centers).
            Length of this vector equals number of bands given in *evec* input 
            array.  Hybrid Wannier function centers are ordered in ascending order.
            Note that in general *n*-th hwfc does not correspond to *n*-th electronic
            state *evec*.

          * **hwf** -- Eigenvectors of the position operator matrix :math:`X`.
            (also called hybrid Wannier functions).  These are returned only if
            parameter *hwf_evec* is set to *True*.
            The shape of this array is [h,x] or [h,x,s] depending on value of *basis*
            and *nspin*.  If *basis* is "bloch" then x refers to indices of 
            Bloch states *evec*.  If *basis* is "orbital" then *x* (or *x* and *s*)
            correspond to orbital index (or orbital and spin index if *nspin* is 2).

        Example usage::

          # diagonalizes Hamiltonian at some k-points
          (evals, evecs) = my_model.solve_all(k_vec,eig_vectors=True)
          # computes hybrid Wannier centers (and functions) for 3-rd kpoint 
          # and bottom five bands along first coordinate
          (hwfc, hwf) = my_model.position_hwf(evecs[:5,2], 0, hwf_evec=True, basis="orbital")

        See also this example: :ref:`haldane_hwf-example`,

        """
        # check if model came from w90
        if self._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        # get position matrix
        pos_mat=self.position_matrix(evec,dir)

        # diagonalize
        if hwf_evec==False:
            hwfc=np.linalg.eigvalsh(pos_mat)
            # sort eigenvalues and convert to real numbers
            hwfc=_nicefy_eig(hwfc)
            return np.array(hwfc,dtype=float)
        else: # find eigenvalues and eigenvectors
            (hwfc,hwf)=np.linalg.eigh(pos_mat)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            hwf=hwf.T
            # sort evectors, eigenvalues and convert to real numbers
            (hwfc,hwf)=_nicefy_eig(hwfc,hwf)
            # convert to right basis
            if basis.lower().strip() in ["wavefunction","bloch"]:
                return (hwfc,hwf)
            elif basis.lower().strip()=="orbital":
                if self._nspin==1:
                    ret_hwf=np.zeros((hwf.shape[0],self._norb),dtype=complex)
                    # sum over bloch states to get hwf in orbital basis
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i]=np.dot(hwf[i],evec)
                    hwf=ret_hwf
                else:
                    ret_hwf=np.zeros((hwf.shape[0],self._norb*2),dtype=complex)
                    # get rid of spin indices
                    evec_use=evec.reshape([hwf.shape[0],self._norb*2])
                    # sum over states
                    for i in range(ret_hwf.shape[0]):
                        ret_hwf[i]=np.dot(hwf[i],evec_use)
                    # restore spin indices
                    hwf=ret_hwf.reshape([hwf.shape[0],self._norb,2])
                return (hwfc,hwf)
            else:
                raise Exception("\n\nBasis must be either 'wavefunction', 'bloch', or 'orbital'")


#=======================================================================
class wf_array(object):
#=======================================================================
    r"""

    This class is used to store and manipulate an array of
    wavefunctions of a tight-binding model
    :class:`pythtb.tb_model` on a regular or non-regular grid
    These are typically the Bloch energy eigenstates of the
    model, but this class can also be used to store a subset
    of Bloch bands, a set of hybrid Wannier functions for a
    ribbon or slab, or any other set of wavefunctions that
    are expressed in terms of the underlying basis orbitals.
    It provides methods that can be used to calculate Berry 
    phases, Berry curvatures, 1st Chern numbers, etc.

    *Regular k-space grid*:
    If the grid is a regular k-mesh (no parametric dimensions),
    a single call to the function
    :func:`pythtb.wf_array.solve_on_grid` will both construct a
    k-mesh that uniformly covers the Brillouin zone, and populate
    it with wavefunctions (eigenvectors) computed on this grid.
    The last point in each k-dimension is set so that it represents
    the same Bloch function as the first one (this involves the
    insertion of some orbital-position-dependent phase factors).

    Example :ref:`haldane_bp-example` shows how to use wf_array on
    a regular grid of points in k-space. Examples :ref:`cone-example`
    and :ref:`3site_cycle-example` show how to use non-regular grid of
    points.

    *Parametric or irregular k-space grid grid*:
    An irregular grid of points, or a grid that includes also
    one or more parametric dimensions, can be populated manually
    with the help of the *[]* operator.  For example, to copy
    eigenvectors *evec* into coordinate (2,3) in the *wf_array*
    object *wf* one can simply do::

      wf[2,3]=evec

    The wavefunctions (here the eigenvectors) *evec* above
    are expected to be in the format *evec[state,orbital]*
    (or *evec[state,orbital,spin]* for the spinfull calculation),
    where *state* typically runs over all bands.
    This is the same format as returned by
    :func:`pythtb.tb_model.solve_one` or
    :func:`pythtb.tb_model.solve_all` (in the latter case one
    needs to restrict it to a single k-point as *evec[:,kpt,:]*
    if the model has *dim_k>=1*).

    If wf_array is used for closed paths, either in a
    reciprocal-space or parametric direction, then one needs to
    include both the starting and ending eigenfunctions even though
    they are physically equivalent.  If the array dimension in
    question is a k-vector direction and the path traverses the
    Brillouin zone in a primitive reciprocal-lattice direction,
    :func:`pythtb.wf_array.impose_pbc` can be used to associate
    the starting and ending points with each other; if it is a
    non-winding loop in k-space or a loop in parameter space,
    then :func:`pythtb.wf_array.impose_loop` can be used instead.
    (These may not be necessary if only Berry fluxes are needed.)

    Example :ref:`3site_cycle-example` shows how one
    of the directions of *wf_array* object need not be a k-vector
    direction, but can instead be a Hamiltonian parameter :math:`\lambda`
    (see also discussion after equation 4.1 in :download:`notes on
    tight-binding formalism <misc/pythtb-formalism.pdf>`).

    The wavevectors stored in *wf_array* are typically Hamiltonian
    eigenstates (e.g., Bloch functions for k-space arrays),
    with the *state* index running over all bands.  However, a
    *wf_array* object can also be used for other purposes, such
    as to store only a restricted set of Bloch states (e.g.,
    just the occupied ones); a set of modified Bloch states
    (e.g., premultiplied by a position, velocity, or Hamiltonian
    operator); or for hybrid Wannier functions (i.e., eigenstates
    of a position operator in a nonperiodic direction).  For an
    example of this kind, see :ref:`cubic_slab_hwf`.

    :param model: Object of type :class:`pythtb.tb_model` representing
      tight-binding model associated with this array of eigenvectors.

    :param mesh_arr: List of dimensions of the mesh of the *wf_array*,
      in order of reciprocal-space and/or parametric directions.

    :param nsta_arr: Optional parameter specifying the number of states
      packed into the *wf_array* at each point on the mesh.  Defaults
      to all states (i.e., norb*nspin).

    Example usage::

      # Construct wf_array capable of storing an 11x21 array of
      # wavefunctions      
      wf = wf_array(tb, [11, 21])
      # populate this wf_array with regular grid of points in
      # Brillouin zone
      wf.solve_on_grid([0.0, 0.0])
      
      # Compute set of eigenvectors at one k-point
      (eval, evec) = tb.solve_one([kx, ky], eig_vectors = True)
      # Store it manually into a specified location in the array
      wf[3,4] = evec
      # To access the eigenvectors from the same position
      print(wf[3,4])

    """
    def __init__(self,model,mesh_arr,nsta_arr=None):
        # number of electronic states for each k-point
        norb = model.atoms.get_global_number_of_atoms() 
        self.norbs = norb
        if not model.solve_dict['sparse']:
            num_eigvals = norb
        else:
            num_eigvals = model.solve_dict["num states"]
        if nsta_arr is None:
            self._nsta_arr=num_eigvals  # this = norb*nspin = no. of bands
            model._nsta = num_eigvals
            # note: 'None' means to use the default, which is all bands!
        else:
            if not _is_int(nsta_arr):
                raise Exception("\n\nArgument nsta_arr not an integer")
            self._nsta_arr=nsta_arr         # set by optional argument
        # number of spin components
        self._nspin=model._nspin
        # number of orbitals
        self._norb=model._norb
        # store orbitals from the model
        self._orb=np.copy(model._orb)
        # store entire model as well
        self._model=model #copy.deepcopy(model)
        # store dimension of array of points on which to keep wavefunctions
        self._mesh_arr=np.array(mesh_arr)        
        self._dim_arr=len(self._mesh_arr)
        # all dimensions should be 2 or larger, because pbc can be used
        # generate temporary array used later to generate object ._wfs
        wfs_dim=np.copy(self._mesh_arr)
        wfs_dim=np.append(wfs_dim,self._nsta_arr)
        self._evals = np.zeros(wfs_dim)
        wfs_dim=np.append(wfs_dim,self._norb)
        if self._nspin==2:
            wfs_dim=np.append(wfs_dim,self._nspin)            
        # store wavefunctions in the form
        #   _wfs[kx_index,ky_index, ... ,state,orb,spin]
        self._wfs=np.zeros(wfs_dim,dtype=complex)


        
    def solve_on_grid(self,start_k,scale=1.0):
        r"""

        Solve a tight-binding model on a regular mesh of k-points covering
        the entire reciprocal-space unit cell. Both points at the opposite
        sides of reciprocal-space unit cell are included in the array.

        This function also automatically imposes periodic boundary
        conditions on the eigenfunctions. See also the discussion in
        :func:`pythtb.wf_array.impose_pbc`.

        :param start_k: Origin of a regular grid of points in the reciprocal space.

        :returns:
          * **gaps** -- returns minimal direct bandgap between n-th and n+1-th 
              band on all the k-points in the mesh.  Note that in the case of band
              crossings one may have to use very dense k-meshes to resolve
              the crossing.

        Example usage::

          # Solve eigenvectors on a regular grid anchored
          # at a given point
          wf.solve_on_grid([-0.5, -0.5])

        """

        # check dimensionality

        if self._dim_arr!=self._model._dim_k:
            raise Exception(\
                "\n\nIf using solve_on_grid method, dimension of wf_array must equal"\
                "\ndim_k of the tight-binding model!")

        # check number of states
        if self._nsta_arr!=self._model._nsta:
            raise Exception(\
                "\n\nWhen initializing this object, you specified nsta_arr to be "+str(self._nsta_arr)+", but"\
                "\nthis does not match the total number of bands specified in the model,"\
                "\nwhich was "+str(self._model._nsta)+".  If you wish to use the solve_on_grid method, do"\
                "\nnot specify the nsta_arr parameter when initializing this object.\n\n")
        
        # store start_k
        self._start_k=start_k

        # to return gaps at all k-points
        if self._nsta_arr<=1:
            all_gaps=None # trivial case since there is only one band
        else:
            gap_dim=np.copy(self._mesh_arr)-1
            gap_dim=np.append(gap_dim,self._nsta_arr-1)
            all_gaps=np.zeros(gap_dim,dtype=float)

        k_list = np.zeros((np.prod(self._mesh_arr),3))
        dim = np.append(self._mesh_arr,3)
        kpoint_berry = np.zeros(dim)
        n=0
        for i in range(self._mesh_arr[0]):
            for j in range(self._mesh_arr[1]):
                for k in range(self._mesh_arr[2]):
                    kpt=scale*(np.array([start_k[0]+float(i)/float(self._mesh_arr[0]),\
                         start_k[1]+float(j)/float(self._mesh_arr[1]),\
                         start_k[2]+float(k)/float(self._mesh_arr[2])]))
                    kpoint_berry[i,j,k,:] = kpt
                    k_list[n,:] = kpt
                    n+=1
        #parallelizes over kpoints  
        self._model.solve_all(k_list)
        eval = self._model.get_eigenvalues(k_list)
        evec = self._model.get_eigenvectors(k_list).T
        for i in range(np.shape(kpoint_berry)[0]):
            for j in range(np.shape(kpoint_berry)[1]):
                for k in range(np.shape(kpoint_berry)[2]):
                    tmpk = kpoint_berry[i,j,k,:]
                    k_ind = closest_index(k_list,tmpk)
                    self[i,j,k]=evec[k_ind,:,:]
                    self._evals[i,j,k] = eval[:,k_ind]        
        #need to make sure this work
        for dir in range(3):
            self.impose_pbc(dir,self._model._per[dir])

    def solve_on_path(self,k_list):
        self._model.solve_all(k_list)
        evec = self._model.get_eigenvectors
        for i in range(np.shape(k_list)[0]):
            k_ind = closest_index(k_list,k_list[i,:])
            self[i]=evec[k_ind,:,:]
        
    def solve_on_one_point(self,kpt,mesh_indices,use_cupy=False,sparse=True,
                  output_name=None,eig_vectors=True):
        r"""

        Solve a tight-binding model on a single k-point and store the eigenvectors
        in the *wf_array* object in the location specified by *mesh_indices*.

        :param kpt: List specifying desired k-point

        :param mesh_indices: List specifying associated set of mesh indices

        :returns:
          None

        Example usage::

          # Solve eigenvectors on a sphere of radius kappa surrounding
          # point k_0 in 3d k-space and pack into a predefined 2d wf_array
          for i in range[n+1]:
            for j in range[m+1]:
              theta=np.pi*i/n
              phi=2*np.pi*j/m
              kx=k_0[0]+kappa*np.sin(theta)*np.cos(phi)
              ky=k_0[1]+kappa*np.sin(theta)*np.sin(phi)
              kz=k_0[2]+kappa*np.cos(theta)
              wf.solve_on_one_point([kx,ky,kz],[i,j])

        """
        self._model.solve_one(kpt) 
        eval = self._model.get_eigenvalues()
        evec = self._model.get_eigenvectors().T
        if _is_int(mesh_indices):
          self._wfs[(mesh_indices,)]=np.squeeze(evec)
        else:
          self._wfs[tuple(mesh_indices)]=np.squeeze(evec)

    def choose_states(self,subset):
        r"""

        Create a new *wf_array* object containing a subset of the
        states in the original one.

        :param subset: List of integers specifying states to keep

        :returns:
          * **wf_new** -- returns a *wf_array* that is identical in all
              respects except that a subset of states have been kept.

        Example usage::

          # Make new *wf_array* object containing only two states
          wf_new=wf.choose_states([3,5])

        """

        # make a full copy of the wf_array
        wf_new=copy.deepcopy(self)

        subset=np.array(subset,dtype=int)
        if subset.ndim!=1:
            raise Exception("\n\nParameter subset must be a one-dimensional array.")
        
        wf_new._nsta_arr=subset.shape[0]

        if self._dim_arr==1:
            wf_new._wfs=wf_new._wfs[:,subset]
        elif self._dim_arr==2:
            wf_new._wfs=wf_new._wfs[:,:,subset]
        elif self._dim_arr==3:
            wf_new._wfs=wf_new._wfs[:,:,:,subset]
        elif self._dim_arr==4:
            wf_new._wfs=wf_new._wfs[:,:,:,:,subset]
        else:
            raise Exception("\n\n_dim_array too large.")

        return(wf_new)

    def empty_like(self,nsta_arr=None):
        r"""

        Create a new empty *wf_array* object based on the original,
        optionally modifying the number of states carried in the array.

        :param nsta_arr: Optional parameter specifying the number
              of states (or bands) to be carried in the array.
              Defaults to the same as the original *wf_array* object.

        :returns:
          * **wf_new** -- returns a similar wf_array except that array
              elements are unitialized and the number of states may have
              changed.

        Example usage::

          # Make new empty wf_array object containing 6 bands per k-point
          wf_new=wf.empty_like(nsta_arr=6)

        """

        # make a full copy of the wf_array
        wf_new=copy.deepcopy(self)

        if nsta_arr is None:
            wf_new._wfs=np.empty_like(wf_new._wfs)
        else:
            wf_shape=list(wf_new._wfs.shape)
            # modify numer of states (after k indices & before orb and spin)
            wf_shape[self._dim_arr]=nsta_arr
            wf_new._wfs=np.empty_like(wf_new._wfs,shape=wf_shape)

        return(wf_new)

    def __check_key(self,key):
        # key is an index list specifying the grid point of interest
        # exception: in 1D, key should simply be an integer
        if self._dim_arr==1:
            if not _is_int(key):
                raise TypeError("Key should be an integer!")
            if key<(-1)*self._mesh_arr[0] or key>=self._mesh_arr[0]:
                raise IndexError("Key outside the range!")
        # do checks for higher dimension
        else:
            if len(key)!=self._dim_arr:
                raise TypeError("Wrong dimensionality of key!")
            for i,k in enumerate(key):
                if not _is_int(k):
                    raise TypeError("Key should be set of integers!")
                if k<(-1)*self._mesh_arr[i] or k>=self._mesh_arr[i]:
                    raise IndexError("Key outside the range!")

    def __getitem__(self,key):
        # check that index array 'key' is valid
        self.__check_key(key)
        # return wavefunction
        return self._wfs[key]
    
    def __setitem__(self,key,value):
        # check that index array 'key' is valid
        self.__check_key(key)
        # store wavefunction
        self._wfs[key]=np.array(value,dtype=complex)

    def impose_pbc(self,mesh_dir,k_dir):
        r"""

        If the *wf_array* object was populated using the
        :func:`pythtb.wf_array.solve_on_grid` method, this function
        should not be used since it will be called automatically by
        the code.

        The eigenfunctions :math:`\Psi_{n {\bf k}}` are by convention
        chosen to obey a periodic gauge, i.e.,
        :math:`\Psi_{n,{\bf k+G}}=\Psi_{n {\bf k}}` not only up to a
        phase, but they are also equal in phase.  It follows that
        the cell-periodic Bloch functions are related by
        :math:`u_{n,{\bf k+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k}}`.
        See :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>` section 4.4 and equation 4.18 for
        more detail.  This routine sets the cell-periodic Bloch function
        at the end of the string in direction :math:`{\bf G}` according
        to this formula, overwriting the previous value.

        This function will impose these periodic boundary conditions along
        one direction of the array. We are assuming that the k-point
        mesh increases by exactly one reciprocal lattice vector along
        this direction. This is currently **not** checked by the code;
        it is the responsibility of the user. Currently *wf_array*
        does not store the k-vectors on which the model was solved;
        it only stores the eigenvectors (wavefunctions).
        
        :param mesh_dir: Direction of wf_array along which you wish to
          impose periodic boundary conditions.

        :param k_dir: Corresponding to the periodic k-vector direction
          in the Brillouin zone of the underlying *tb_model*.  Since
          version 1.7.0 this parameter is defined so that it is
          specified between 0 and *dim_r-1*.

        See example :ref:`3site_cycle-example`, where the periodic boundary
        condition is applied only along one direction of *wf_array*.

        Example usage::

          # Imposes periodic boundary conditions along the mesh_dir=0
          # direction of the wf_array object, assuming that along that
          # direction the k_dir=1 component of the k-vector is increased
          # by one reciprocal lattice vector.  This could happen, for
          # example, if the underlying tb_model is two dimensional but
          # wf_array is a one-dimensional path along k_y direction.          
          wf.impose_pbc(mesh_dir=0,k_dir=1)

        """

        if k_dir not in self._model._per:
            raise Exception("Periodic boundary condition can be specified only along periodic directions!")

        # Compute phase factors
        ffac=np.exp(-2.j*np.pi*self._orb[:,k_dir])
        if self._nspin==1:
            phase=ffac
        else:
            # for spinors, same phase multiplies both components
            phase=np.zeros((self._norb,2),dtype=complex)
            phase[:,0]=ffac
            phase[:,1]=ffac
        
        # Copy first eigenvector onto last one, multiplying by phase factors
        # We can use numpy broadcasting since the orbital index is last
        if mesh_dir==0:
            self._wfs[-1,...]=self._wfs[0,...]*phase
        elif mesh_dir==1:
            self._wfs[:,-1,...]=self._wfs[:,0,...]*phase
        elif mesh_dir==2:
            self._wfs[:,:,-1,...]=self._wfs[:,:,0,...]*phase
        elif mesh_dir==3:
            self._wfs[:,:,:,-1,...]=self._wfs[:,:,:,0,...]*phase
        else:
            raise Exception("\n\nWrong value of mesh_dir.")

    def impose_loop(self,mesh_dir):
        r"""

        If the user knows that the first and last points along the
        *mesh_dir* direction correspond to the same Hamiltonian (this
        is **not** checked), then this routine can be used to set the
        eigenvectors equal (with equal phase), by replacing the last
        eigenvector with the first one (for each band, and for each
        other mesh direction, if any).

        This routine should not be used if the first and last points
        are related by a reciprocal lattice vector; in that case,
        :func:`pythtb.wf_array.impose_pbc` should be used instead.

        :param mesh_dir: Direction of wf_array along which you wish to
          impose periodic boundary conditions.

        Example usage::

          # Suppose the wf_array object is three-dimensional
          # corresponding to (kx,ky,lambda) where (kx,ky) are
          # wavevectors of a 2D insulator and lambda is an
          # adiabatic parameter that goes around a closed loop.
          # Then to insure that the states at the ends of the lambda
          # path are equal (with equal phase) in preparation for
          # computing Berry phases in lambda for given (kx,ky),
          # do wf.impose_loop(mesh_dir=2)

        """

        # Copy first eigenvector onto last one
        if mesh_dir==0:
            self._wfs[-1,...]=self._wfs[0,...]
        elif mesh_dir==1:
            self._wfs[:,-1,...]=self._wfs[:,0,...]
        elif mesh_dir==2:
            self._wfs[:,:,-1,...]=self._wfs[:,:,0,...]
        elif mesh_dir==3:
            self._wfs[:,:,:,-1,...]=self._wfs[:,:,:,0,...]
        else:
            raise Exception("\n\nWrong value of mesh_dir.")

    def position_matrix(self, key, occ, dir):
        """Similar to :func:`pythtb.tb_model.position_matrix`.  Only
        difference is that, in addition to specifying *dir*, one also
        has to specify *key* (k-point of interest) and *occ* (list of
        states to be included, which can optionally be 'All')."""

        # Check for special case of parameter occ
        if type(occ) is str and occ == 'All':
            occ=np.arange(self._nsta_arr,dtype=int)
        else:
            occ=np.array(occ,dtype=int)

        if occ.ndim!=1:
            raise Exception("""\n\nParameter occ must be a one-dimensional array or string "All".""")
            
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        #
        evec=self._wfs[tuple(key)][occ]
        return self._model.position_matrix(evec,dir)

    def position_expectation(self, key, occ, dir):
        """Similar to :func:`pythtb.tb_model.position_expectation`.  Only
        difference is that, in addition to specifying *dir*, one also
        has to specify *key* (k-point of interest) and *occ* (list of
        states to be included, which can optionally be 'All')."""

        # Check for special case of parameter occ
        if type(occ) is str and occ == 'All':
            occ=np.arange(self._nsta_arr,dtype=int)
        else:
            occ=np.array(occ,dtype=int)
            
        if occ.ndim!=1:
            raise Exception("""\n\nParameter occ must be a one-dimensional array or string "All".""")
            
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        #
        evec=self._wfs[tuple(key)][occ]
        return self._model.position_expectation(evec,dir)

    def position_hwf(self, key, occ, dir, hwf_evec=False, basis="wavefunction"):
        """Similar to :func:`pythtb.tb_model.position_hwf`, except that
        in addition to specifying *dir*, one also has to specify
        *key*, the k-point of interest, and *occ*, a list of states to
        be included (typically the occupied states).

        For backwards compatibility the default value of *basis* here is different
        from that in :func:`pythtb.tb_model.position_hwf`.
        """

        # Check for special case of parameter occ
        if type(occ) is str and occ == 'All':
            occ=np.arange(self._nsta_arr,dtype=int)
        else:
            occ=np.array(occ,dtype=int)

        if occ.ndim!=1:
            raise Exception("""\n\nParameter occ must be a one-dimensional array or string "All".""")
            
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        evec=self._wfs[tuple(key)][occ]
        return self._model.position_hwf(evec,dir,hwf_evec,basis)

    def berry_phase(self,occ="All",dir=None,contin=True,berry_evals=False,subset=None):
        r"""

        Computes the Berry phase along a given array direction
        and for a given set of states.  These are typically the
        occupied Bloch states, in which case *occ* should range
        over all occupied bands.  In this context, the occupied
        and unoccupied bands should be well separated in energy;
        it is the responsibility of the user to check that this
        is satisfied.  If *occ* is not specified or is specified
        as 'All', all states are selected. By default, the
        function returns the Berry phase traced over the
        specified set of bands, but optionally the individual
        phases of the eigenvalues of the global unitary rotation
        matrix (corresponding to "maximally localized Wannier
        centers" or "Wilson loop eigenvalues") can be requested
        (see parameter *berry_evals* for more details).

        For an array of size *N* in direction $dir$, the Berry phase
        is computed from the *N-1* inner products of neighboring
        eigenfunctions.  This corresponds to an "open-path Berry
        phase" if the first and last points have no special
        relation.  If they correspond to the same physical
        Hamiltonian, and have been properly aligned in phase using
        :func:`pythtb.wf_array.impose_pbc` or
        :func:`pythtb.wf_array.impose_loop`, then a closed-path
        Berry phase will be computed.

        For a one-dimensional wf_array (i.e., a single string), the
        computed Berry phases are always chosen to be between -pi and pi.
        For a higher dimensional wf_array, the Berry phase is computed
        for each one-dimensional string of points, and an array of
        Berry phases is returned. The Berry phase for the first string
        (with lowest index) is always constrained to be between -pi and
        pi. The range of the remaining phases depends on the value of
        the input parameter *contin*.

        The discretized formula used to compute Berry phase is described
        in Sec. 4.5 of :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>`.

        :param occ: Optional array of indices of states to be included
          in the subsequent calculations, typically the indices of
          bands considered occupied.  Default is all bands.

        :param dir: Index of wf_array direction along which Berry phase is
          computed. This parameters needs not be specified for
          a one-dimensional wf_array.

        :param contin: Optional boolean parameter. If True then the
          branch choice of the Berry phase (which is indeterminate
          modulo 2*pi) is made so that neighboring strings (in the
          direction of increasing index value) have as close as
          possible phases. The phase of the first string (with lowest
          index) is always constrained to be between -pi and pi. If
          False, the Berry phase for every string is constrained to be
          between -pi and pi. The default value is True.

        :param berry_evals: Optional boolean parameter. If True then
          will compute and return the phases of the eigenvalues of the
          product of overlap matrices. (These numbers correspond also
          to hybrid Wannier function centers.) These phases are either
          forced to be between -pi and pi (if *contin* is *False*) or
          they are made to be continuous (if *contin* is True).

        :returns:
          * **pha** -- If *berry_evals* is False (default value) then
            returns the Berry phase for each string. For a
            one-dimensional wf_array this is just one number. For a
            higher-dimensional wf_array *pha* contains one phase for
            each one-dimensional string in the following format. For
            example, if *wf_array* contains k-points on mesh with
            indices [i,j,k] and if direction along which Berry phase
            is computed is *dir=1* then *pha* will be two dimensional
            array with indices [i,k], since Berry phase is computed
            along second direction. If *berry_evals* is True then for
            each string returns phases of all eigenvalues of the
            product of overlap matrices. In the convention used for
            previous example, *pha* in this case would have indices
            [i,k,n] where *n* refers to index of individual phase of
            the product matrix eigenvalue.

        Example usage::

          # Computes Berry phases along second direction for three lowest
          # occupied states. For example, if wf is threedimensional, then
          # pha[2,3] would correspond to Berry phase of string of states
          # along wf[2,:,3]
          pha = wf.berry_phase([0, 1, 2], 1)

        See also these examples: :ref:`haldane_bp-example`,
        :ref:`cone-example`, :ref:`3site_cycle-example`,

        """
        if type(subset)!=np.ndarray:
            subset = np.array(range(self._model.norbs)) #use all orbitals
        # special case requesting all states in the array
        if (type(occ) is str and occ == 'All') or occ is None:
            # note that 'None' means 'not specified', not 'no states'
            occ=np.arange(self._nsta_arr,dtype=int)
        else:
            occ=np.array(occ,dtype=int)

        if occ.ndim!=1:
            raise Exception("""\n\nParameter occ must be a one-dimensional array or string "All" or None.""")

                    
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        #if dir<0 or dir>self._dim_arr-1:
        #  raise Exception("\n\nDirection key out of range")
        #
        # This could be coded more efficiently, but it is hard-coded for now.
        #
        # 1D case
        if self._dim_arr==1:
            # pick which wavefunctions to use
            wf_use=self._wfs[:,occ,subset]
            # calculate berry phase
            ret=_one_berry_loop(wf_use,berry_evals)
        # 2D case
        elif self._dim_arr==2:
            # choice along which direction you wish to calculate berry phase
            if dir==0:
                ret=[]
                for i in range(self._mesh_arr[1]):
                    wf_use=self._wfs[:,i,:,:][:,occ,subset]
                    ret.append(_one_berry_loop(wf_use,berry_evals))
            elif dir==1:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    wf_use=self._wfs[i,:,:,:][:,occ,subset]
                    ret.append(_one_berry_loop(wf_use,berry_evals))
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        # 3D case
        elif self._dim_arr==3:
            # choice along which direction you wish to calculate berry phase
            if dir==0:
                ret=[]
                for i in range(self._mesh_arr[1]):
                    ret_t=[]
                    for j in range(self._mesh_arr[2]):
                        wf_use=self._wfs[:,i,j,:,:][:,occ,subset]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            elif dir==1:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    ret_t=[]
                    for j in range(self._mesh_arr[2]):
                        wf_use=self._wfs[i,:,j,:,:][:,occ,subset]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            elif dir==2:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    ret_t=[]
                    for j in range(self._mesh_arr[1]):
                        wf_use=self._wfs[i,j,:,:,:][:,occ,subset]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        else:
            raise Exception("\n\nWrong dimensionality!")

        # convert phases to numpy array
        if self._dim_arr>1 or berry_evals==True:
            ret=np.array(ret,dtype=float)

        # make phases of eigenvalues continuous
        if contin==True:
            # iron out 2pi jumps, make the gauge choice such that first phase in the
            # list is fixed, others are then made continuous.
            if berry_evals==False:
                # 2D case
                if self._dim_arr==2:
                    ret=_one_phase_cont(ret,ret[0])
                # 3D case
                elif self._dim_arr==3:
                    for i in range(ret.shape[1]):
                        if i==0: clos=ret[0,0]
                        else: clos=ret[0,i-1]
                        ret[:,i]=_one_phase_cont(ret[:,i],clos)
                elif self._dim_arr!=1:
                    raise Exception("\n\nWrong dimensionality!")
            # make eigenvalues continuous. This does not take care of band-character
            # at band crossing for example it will just connect pairs that are closest
            # at neighboring points.
            else:
                # 2D case
                if self._dim_arr==2:
                    ret=_array_phases_cont(ret,ret[0,:])
                # 3D case
                elif self._dim_arr==3:
                    for i in range(ret.shape[1]):
                        if i==0: clos=ret[0,0,:]
                        else: clos=ret[0,i-1,:]
                        ret[:,i]=_array_phases_cont(ret[:,i],clos)
                elif self._dim_arr!=1:
                    raise Exception("\n\nWrong dimensionality!")
        return ret
    
    def berry_flux(self,occ="All",dirs=None,individual_phases=False):
        r"""

        In the case of a 2-dimensional *wf_array* array calculates the
        integral of Berry curvature over the entire plane.  In higher
        dimensional case (3 or 4) it will compute integrated curvature
        over all 2-dimensional slices of a higher-dimensional
        *wf_array*.

        :param occ: Optional array of indices of states to be included
          in the subsequent calculations, typically the indices of
          bands considered occupied.  If not specified or specified as
          'All', all bands are included.

        :param dirs: Array of indices of two wf_array directions on which
          the Berry flux is computed. This parameter needs not be
          specified for a two-dimensional wf_array.  By default *dirs* takes
          first two directions in the array.

        :param individual_phases: If *True* then returns Berry phase
          for each plaquette (small square) in the array. Default
          value is *False*.

        :returns:

          * **flux** -- In a 2-dimensional case returns and integral
            of Berry curvature (if *individual_phases* is *True* then
            returns integral of Berry phase around each plaquette).
            In higher dimensional case returns integral of Berry
            curvature over all slices defined with directions *dirs*.
            Returned value is an array over the remaining indices of
            *wf_array*.  (If *individual_phases* is *True* then it
            returns again phases around each plaquette for each
            slice. First indices define the slice, last two indices
            index the plaquette.)

        Example usage::

          # Computes integral of Berry curvature of first three bands
          flux = wf.berry_flux([0, 1, 2])

        """

        # special case requesting all states in the array
        if (type(occ) is str and occ == 'All') or occ is None:
            # note that 'None' means 'not specified', not 'no states'
            occ=np.arange(self._nsta_arr,dtype=int)
        else:
            occ=np.array(occ,dtype=int)
            
        # check if model came from w90
        if self._model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()

        # default case is to take first two directions for flux calculation
        if dirs is None:
            dirs=[0,1]

        # consistency checks
        if dirs[0]==dirs[1]:
            raise Exception("Need to specify two different directions for Berry flux calculation.")
        if dirs[0]>=self._dim_arr or dirs[1]>=self._dim_arr or dirs[0]<0 or dirs[1]<0:
            raise Exception("Direction for Berry flux calculation out of bounds.")

        # 2D case
        if self._dim_arr==2:
            # compute the fluxes through all plaquettes on the entire plane 
            ord=list(range(len(self._wfs.shape)))
            # select two directions from dirs
            ord[0]=dirs[0]
            ord[1]=dirs[1]
            plane_wfs=self._wfs.transpose(ord)
            # take bands of choice
            plane_wfs=plane_wfs[:,:,occ]

            # compute fluxes
            all_phases=_one_flux_plane(plane_wfs)

            # return either total flux or individual phase for each plaquete
            if individual_phases==False:
                return all_phases.sum()
            else:
                return all_phases

        # 3D or 4D case
        elif self._dim_arr in [3,4]:
            # compute the fluxes through all plaquettes on the entire plane 
            ord=list(range(len(self._wfs.shape)))
            # select two directions from dirs
            ord[0]=dirs[0]
            ord[1]=dirs[1]

            # find directions over which we wish to loop
            ld=list(range(self._dim_arr))
            ld.remove(dirs[0])
            ld.remove(dirs[1])
            if len(ld)!=self._dim_arr-2:
                raise Exception("Hm, this should not happen? Inconsistency with the mesh size.")
            
            # add remaining indices
            if self._dim_arr==3:
                ord[2]=ld[0]
            if self._dim_arr==4:
                ord[2]=ld[0]
                ord[3]=ld[1]

            # reorder wavefunctions
            use_wfs=self._wfs.transpose(ord)

            # loop over the the remaining direction
            if self._dim_arr==3:
                slice_phases=np.zeros((self._mesh_arr[ord[2]],self._mesh_arr[dirs[0]]-1,self._mesh_arr[dirs[1]]-1),dtype=float)
                for i in range(self._mesh_arr[ord[2]]):
                    # take a 2d slice
                    plane_wfs=use_wfs[:,:,i]
                    # take bands of choice
                    plane_wfs=plane_wfs[:,:,occ]
                    # compute fluxes on the slice
                    slice_phases[i,:,:]=_one_flux_plane(plane_wfs)
            elif self._dim_arr==4:
                slice_phases=np.zeros((self._mesh_arr[ord[2]],self._mesh_arr[ord[3]],self._mesh_arr[dirs[0]]-1,self._mesh_arr[dirs[1]]-1),dtype=float)
                for i in range(self._mesh_arr[ord[2]]):
                    for j in range(self._mesh_arr[ord[3]]):
                        # take a 2d slice
                        plane_wfs=use_wfs[:,:,i,j]
                        # take bands of choice
                        plane_wfs=plane_wfs[:,:,occ]
                        # compute fluxes on the slice
                        slice_phases[i,j,:,:]=_one_flux_plane(plane_wfs)

            # return either total flux or individual phase for each plaquete
            if individual_phases==False:
                return slice_phases.sum(axis=(-2,-1))
            else:
                return slice_phases

        else:
            raise Exception("\n\nWrong dimensionality!")

#=======================================================================
# Begin internal definitions
#=======================================================================
def best_match(psi1, psi2,threshold=0.1):

    if threshold is None:
        threshold = (2 * psi1.shape[0])**-0.25
    Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
    
    orig, perm = linear_sum_assignment(-Q)
    #print(Q[orig, perm] )
    return perm, Q[orig, perm] < threshold
    
def _nicefy_eig(eval,eig=None):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=np.array(eval.real,dtype=float)
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    if not (eig is None):
        eig=eig[args]
        return (eval,eig)
    return eval

# for nice justified printout
def _nice_float(x,just,rnd):
    return str(round(x,rnd)).rjust(just)
def _nice_int(x,just):
    return str(x).rjust(just)
def _nice_complex(x,just,rnd):
    ret=""
    ret+=_nice_float(complex(x).real,just,rnd)
    if complex(x).imag<0.0:
        ret+=" - "
    else:
        ret+=" + "
    ret+=_nice_float(abs(complex(x).imag),just,rnd)
    ret+=" i"
    return ret
    
def _wf_dpr(wf1,wf2):
    """calculate dot product between two wavefunctions.
    wf1 and wf2 are of the form [orbital,spin]"""
    return np.dot(wf1.flatten().conjugate(),wf2.flatten())

def _one_berry_loop(wf,berry_evals=False):
    """Do one Berry phase calculation (also returns a product of M
    matrices).  Always returns numbers between -pi and pi.  wf has
    format [kpnt,band,orbital,spin] and kpnt has to be one dimensional.
    Assumes that first and last k-point are the same. Therefore if
    there are n wavefunctions in total, will calculate phase along n-1
    links only!  If berry_evals is True then will compute phases for
    individual states, these corresponds to 1d hybrid Wannier
    function centers. Otherwise just return one number, Berry phase."""
    # number of occupied states
    nocc=wf.shape[1]
    # temporary matrices
    prd=np.identity(nocc,dtype=complex)
    ovr=np.zeros([nocc,nocc],dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    for i in range(wf.shape[0]-1):
        # generate overlap matrix, go over all bands
        for j in range(nocc):
            for k in range(nocc):
                ovr[j,k]=_wf_dpr(wf[i,j,:],wf[i+1,k,:])
        # only find Berry phase
        if berry_evals==False:
            # multiply overlap matrices
            prd=np.dot(prd,ovr)
        # also find phases of individual eigenvalues
        else:
            # cleanup matrices with SVD then take product
            matU,sing,matV=np.linalg.svd(ovr)
            prd=np.dot(prd,np.dot(matU,matV))
    # calculate Berry phase
    if berry_evals==False:
        det=np.linalg.det(prd)
        pha=(-1.0)*np.angle(det)
        return pha
    # calculate phases of all eigenvalues
    else:
        evals=np.linalg.eigvals(prd)
        eval_pha=(-1.0)*np.angle(evals)
        # sort these numbers as well
        eval_pha=np.sort(eval_pha)
        return eval_pha

def _one_flux_plane(wfs2d):
    "Compute fluxes on a two-dimensional plane of states."
    # size of the mesh
    nk0=wfs2d.shape[0]
    nk1=wfs2d.shape[1]
    # number of bands (will compute flux of all bands taken together)
    nbnd=wfs2d.shape[2]

    # here store flux through each plaquette of the mesh
    all_phases=np.zeros((nk0-1,nk1-1),dtype=float)

    # go over all plaquettes
    for i in range(nk0-1):
        for j in range(nk1-1):
            # generate a small loop made out of four pieces
            wf_use=[]
            wf_use.append(wfs2d[i,j])
            wf_use.append(wfs2d[i+1,j])
            wf_use.append(wfs2d[i+1,j+1])
            wf_use.append(wfs2d[i,j+1])
            wf_use.append(wfs2d[i,j])
            wf_use=np.array(wf_use,dtype=complex)
            # calculate phase around one plaquette
            all_phases[i,j]=_one_berry_loop(wf_use)

    return all_phases

def no_2pi(x,clos):
    "Make x as close to clos by adding or removing 2pi"
    while abs(clos-x)>np.pi:
        if clos-x>np.pi:
            x+=2.0*np.pi
        elif clos-x<-1.0*np.pi:
            x-=2.0*np.pi
    return x

def _one_phase_cont(pha,clos):
    """Reads in 1d array of numbers *pha* and makes sure that they are
    continuous, i.e., that there are no jumps of 2pi. First number is
    made as close to *clos* as possible."""
    ret=np.copy(pha)
    # go through entire list and "iron out" 2pi jumps
    for i in range(len(ret)):
        # which number to compare to
        if i==0: cmpr=clos
        else: cmpr=ret[i-1]
        # make sure there are no 2pi jumps
        ret[i]=no_2pi(ret[i],cmpr)
    return ret

def _array_phases_cont(arr_pha,clos):
    """Reads in 2d array of phases *arr_pha* and makes sure that they
    are continuous along first index, i.e., that there are no jumps of
    2pi. First array of phasese is made as close to *clos* as
    possible."""
    ret=np.zeros_like(arr_pha)
    # go over all points
    for i in range(arr_pha.shape[0]):
        # which phases to compare to
        if i==0: cmpr=clos
        else: cmpr=ret[i-1,:]
        # remember which indices are still available to be matched
        avail=list(range(arr_pha.shape[1]))
        # go over all phases in cmpr[:]
        for j in range(cmpr.shape[0]):
            # minimal distance between pairs
            min_dist=1.0E10
            # closest index
            best_k=None
            # go over each phase in arr_pha[i,:]
            for k in avail:
                cur_dist=np.abs(np.exp(1.0j*cmpr[j])-np.exp(1.0j*arr_pha[i,k]))
                if cur_dist<=min_dist:
                    min_dist=cur_dist
                    best_k=k
            # remove this index from being possible pair later
            avail.pop(avail.index(best_k))
            # store phase in correct place
            ret[i,j]=arr_pha[i,best_k]
            # make sure there are no 2pi jumps
            ret[i,j]=no_2pi(ret[i,j],cmpr[j])
    return ret



def _cart_to_red(tmp,cart):
    "Convert cartesian vectors cart to reduced coordinates of a1,a2,a3 vectors"
    (a1,a2,a3)=tmp
    # matrix with lattice vectors
    cnv=np.array([a1,a2,a3])
    # transpose a matrix
    cnv=cnv.T
    # invert a matrix
    cnv=np.linalg.inv(cnv)
    # reduced coordinates
    red=np.zeros_like(cart,dtype=float)
    for i in range(0,len(cart)):
        red[i]=np.dot(cnv,cart[i])
    return red

def _red_to_cart(tmp,red):
    "Convert reduced to cartesian vectors."
    (a1,a2,a3)=tmp
    # cartesian coordinates
    cart=np.zeros_like(red,dtype=float)
    for i in range(0,len(cart)):
        cart[i,:]=a1*red[i][0]+a2*red[i][1]+a3*red[i][2]
    return cart

def _is_int(a):
    return np.issubdtype(type(a), np.integer)

def check_arr(slice,array):
    tol = 1e-4
    v = np.linalg.norm(np.tile(slice,(np.shape(array)[0],1))-array,axis=1)
    if np.min(v) < tol:
        return True
    else:
        return False
    
def closest_index(space1,slice_val):
    k_ind=0
    min_dist = 1e6
    for i in range(np.shape(space1)[0]):
        dist = np.linalg.norm(space1[i,:]-slice_val)
        if dist < min_dist:
            k_ind=i
            min_dist=dist

    return k_ind

def load_dataHDF5(dir_name,k_list):
    files = glob.glob(os.path.join(dir_name,"*.hdf5"))
    evals=[]
    evec=[]
    kpoints=np.empty_like(k_list)
    indices=[]
    i=0
    for fname in files:
        f = h5py.File(fname, 'r')
        keys = f.keys()
        for k in keys:
            # kp = fname.split("/")[-1].split("_")[1].split(".hdf5")[0]
            # kp = re.findall(r"[-+]?\d*\.\d+|\d+", kp)
            # kp = [float(s) for s in kp] 
            g = f[k]
            kp = g['kpoint'][:]
            # if type(k_list)==np.ndarray:
            #     if check_arr(kp,k_list):
            #         continue
            index = closest_index(k_list,kp)
            kpoints[index,:]=kp
            tmp_evals = g['eigvals'][:]
            tmp_evec = g['eigvecs'][:]
            if i ==0:
                evals = np.empty((np.shape(tmp_evals)[0],np.shape(k_list)[0]))
                evec = np.empty((np.shape(tmp_evec)[0],np.shape(tmp_evec)[1],np.shape(k_list)[0]),dtype=complex)
            evals[:,index] = tmp_evals
            evec[:,:,index] = tmp_evec
            
            i+=1

    return evals,evec,kpoints

def _offdiag_approximation_warning_and_stop():
    raise Exception("""

                    
----------------------------------------------------------------------

  It looks like you are trying to calculate Berry-like object that
  involves position operator.  However, you are using a tight-binding
  model that was generated from Wannier90.  This procedure introduces
  approximation as it ignores off-diagonal elements of the position
  operator in the Wannier basis.  This is discussed here in more
  detail:

    http://www.physics.rutgers.edu/pythtb/usage.html#pythtb.w90

  If you know what you are doing and wish to continue with the
  calculation despite this approximation, please call the following
  function on your tb_model object

    my_model.ignore_position_operator_offdiagonal()

----------------------------------------------------------------------

""")

def load_model(dir_name):
    data = np.load(dir_name,allow_pickle=True) 
    positions=data.item().get('positions')
    cell=data.item().get('cell')
    symbols=data.item().get('layer_tags')
    atoms = ase.Atoms(positions=positions,cell=cell,symbols=symbols)
    parameters=data.item().get('parameters')
    solve_dict=data.item().get('solve_dict')
    model = tblg_model(atoms,parameters=parameters)
    model.read_data = True
    model.set_solver(solve_dict)
    return model
