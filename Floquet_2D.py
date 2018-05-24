# file Floquet_2D package
# contains functions needed for floquet lattice calculation.

# Author: Amin Ahmadi
# Date: 30 Jan, 2018

import numpy as np
############################################################
def make_sq(mlat, dAB, *J):
    """Constructs the Hamiltonian and the connection 
    matrix of a bipartite square lattice. 
    0--o  0--o    
    |  |  |  |
    o--0  o--0
    |  |  |  |
    0--o  0--o
    
    One period of the driven fieldconsists of 5 time
    interval which are defined through hoping amplitude Ji
    \in [1,2,..,5], where in the last interval all hopping
    amplitude are off.
    
    input:
    ------
    mlat: integer, width of slab, number of site in 
    one super unitcell would br 2xm
    
    J: is a tuple of float. Provide the hopping amplitude
    for different time intrval

    returns: 
    --------
    h: 2mlatx2mlat complex matrix, Hamiltonian of the slab
    
    tau: 2mlatx2mlat complex matrix, the connection matrix 
    between two neighbor super unit cell

    """
    if (len(J)!=4):
        print("Number of paramaters are exceeded 5!")
    NN = 2*mlat
    
    tau = np.zeros((NN,NN), dtype=complex)
    h = np.zeros((NN,NN), dtype=complex)
    
    for i in range(mlat-1):
        if (i%2==0):
            h[i,i] = dAB/2.                  # on-site energy
            h[mlat+i,mlat+i] = -dAB/2.       # on-site energy            
            h[i, mlat+i] = J[0]
            h[i, i+1] = J[1]
            h[mlat+i, mlat+i+1] = J[3]
            #
            tau[mlat+i, i] = J[2]
        elif (i%2==1):
            h[i,i] = -dAB/2.                # on-site energy
            h[mlat+i,mlat+i] = dAB/2.       # on-site energy            
            h[i, mlat+i] = J[2]
            h[i, i+1] = J[3]
            h[mlat+i, mlat+i+1] = J[1]
            #
            tau[mlat+i, i] = J[0]

    # End of loop over lattice sites

    # The upper edge site
    if (mlat-1 % 2==0):
        h[mlat-1, mlat-1] = dAB/2.                  # on-site energy
        h[NN-1,NN-1] = -dAB/2.                      # on-site energy            
        h[mlat-1, NN-1] = J[0]
        #
        tau[NN-1, mlat-1] = J[2]
    elif (mlat-1 % 2==1):
        h[mlat-1, mlat-1] = -dAB/2.                 # on-site energy
        h[NN-1,NN-1] = dAB/2.                       # on-site energy            
        h[mlat-1, NN-1] = J[2]
        #
        tau[NN-1, mlat-1] = J[0]        
    
    h = h + h.conj().T          # make it hermitian
    return h, tau

############################################################
def make_Gr(mlat, *J):
    """ Constructs the Hamiltonian and the connection 
    matrix of an armchair graphene strip.
    0--o  0--o
    |  |  |  |
    o  0--o  0
    |  |  |  |
    0--o  0--o
    |  |  |  |
    o  0--o  0
    |  |  |  |
    0--o  0--o
    
    One period of the driven fieldconsists of 3 time
    interval which are defined through hoping amplitude Ji
    \in [1,2,3].
    
    
    input:
    ------
    mlat: integer, width of slab, number of site in 
    one super unitcell would br 2xm

    returns: 
    --------
    h: 2mlatx2mlat complex matrix, Hamiltonian of the slab
    
    tau: 2mlatx2mlat complex matrix, the connection matrix 
    between two neighbor super unit cell

    """

    if (len(J)!=3):
        print("Number of paramaters are not right, must be  5!")
    
    NN = 2*mlat                 # # of sites in one super unitcell
    tau = -np.zeros((NN, NN),dtype=complex)
    h = np.zeros((NN,NN), dtype=complex)

    # translational cell's Hamiltonian
    for i in range(mlat-1):
        if (i%2==0):
            h[i,i+1] = J[0]
            h[mlat+i,mlat+i+1] = J[1]
            h[i,mlat+i] = J[2]    # horizoltal connection
        elif (i%2==1):
            h[i,i+1] = J[1]
            h[mlat+i,mlat+i+1] = J[0]
    # longitudinal connection of the last sites
    if (mlat-1)%2 == 0:
        h[mlat-1,2*mlat-1] = J[2]
            
    h = h + h.conj().T          # make it hermitian

    # Hopping matrix
    for i in range(1,mlat,2):
        tau[i+mlat,i] = J[2]

    return h, tau
