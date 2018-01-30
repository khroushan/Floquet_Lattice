# file Floquet_2D package
# contains functions needed for floquet lattice calculation.

# Author: Amin Ahmadi
# Date: 30 Jan, 2018

import numpy as np
############################################################
def make_sq(mlat):
    """ Constructs the Hamiltonian and the connection 
    matrix of a square lattice
    0--0   h:= |
    |  |   tau:= --
    0--0
    |  |
    0--0
    returns: unitcell hamiltonian h
             hopping matrix       tau
    """
    # sites from one slice are connected to the
    # next slice with same labels
    tau = -np.eye(mlat,dtype=complex)
    h = np.zeros((mlat,mlat), dtype=complex)
    
    for i in range(mlat-1):
        h[i,i+1] = -1.0
        h[i+1,i] = np.conjugate(h[i,i+1])

    return h, tau

############################################################
def make_Gr(mlat, J1=1, J2=1, J3=1):
    """ Constructs the Hamiltonian and the connection 
    matrix of an armchair graphene strip
    0--o  0--o
    |  |  |  |
    o  0--o  0
    |  |  |  |
    0--o  0--o
    |  |  |  |
    o  0--o  0
    |  |  |  |
    0--o  0--o
    returns: unitcell hamiltonian h
             hopping matrix       tau
    """
    NN = 2*mlat                 # # of sites in one super unitcell
    tau = -np.zeros((NN, NN),dtype=complex)
    h = np.zeros((NN,NN), dtype=complex)

    # translational cell's Hamiltonian
    for i in range(mlat-1):
        if (i%2==0):
            h[i,i+1] = J1
            h[mlat+i,mlat+i+1] = J2
            h[i,mlat+i] = J3    # horizoltal connection
        elif (i%2==1):
            h[i,i+1] = J2
            h[mlat+i,mlat+i+1] = J1
            
    h = h + h.conj().T          # make it hermitian

    # Hopping matrix
    for i in range(1,mlat,2):
        tau[i+mlat,i] = J3

    return h, tau
