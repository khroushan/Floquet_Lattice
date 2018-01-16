# Program to calculate the Band Structure of Floquet operator
# in a honeycomb lattice with stroboscopic driven field, here the
# field just affects the on-site energy
# for a square lattice
# Author: Amin Ahmdi
# Date: Oct 11, 2017
# Date: oct 18, 2017         on-site energy
# ################################################################
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as pl

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

    # h[0,mlat-1] = -1.0
    # h[mlat-1,0] = np.conjugate(h[0,mlat-1])
                               
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
############################################################
def make_Gr_onsite(mlat, enut=0, J=1):
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

    # on-site energy
    for i in range(NN):
        h[i,i] = enut

    J1 = J2 = J3 = J
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



############################################################
##############         Main Program     ####################
############################################################
N_k = 100                                # Num of k-points
N_t = 2                                  # Num of time intervals
T = 1.                                   # One period of driven field
# During each iterval T/N_t, the Hamiltonian is time-independent
mlat = input("Graphene strip width: ")                # width of strip
NN = 2*mlat
H_k = np.zeros((NN,NN), dtype=complex)   # k-representation H
E_k = np.zeros((NN), dtype=complex)      # eigenenergies
psi_k = np.zeros((NN,NN), dtype=complex) # matrix of eigenvectors

# different hopping amplitude
on_site_E = input("Enter the on-site energy difference: ") 
J = np.pi/16                             # hopping amplitude 
data_plot = np.zeros((N_k, NN+1), dtype=float)

# loop over k, first BZ 
for ik in range(N_k):

    ka = -np.pi/2. +  ik*(np.pi/N_k)
    # ka = ik*(np.pi/N_k)
    M_eff = np.eye((NN), dtype=complex)   # aux matrix
    for it in range(N_t):
        if   (it==0 ):
            enut = on_site_E
        elif (it==1):
            enut = 0

        # construct the Hamiltonian and hopping matrices
        h, tau = make_Gr_onsite(mlat, enut)
        tau_dg = tau.conj().T 

        # Construct matrix: [h + tau*exp(ika) + tau^+ * exp(-ika)]
        # and diagonalization
        H_k = h + np.exp(1.j*ka)*tau + np.exp(-1.j*ka)*tau_dg

        # return eigenenergies and vectors
        E_k, U = lg.eig(H_k)    

        # U^-1 * exp(H_d) U
        U_inv = lg.inv(U)

        # construct a digonal matrix out of a vector
        #H_M= np.diag(np.exp((-1j/3.)*E_k*T))
        M1 = (np.exp((-1j/2.)*E_k*T) * U_inv.T).T
        #MM = np.dot(U_inv,np.dot(H_M, U))
        MM = np.dot(U,M1)
        M_eff = np.dot(M_eff,MM)


    E_Fl, UF = lg.eig(M_eff)    
    data_plot[ik,0] = ka
    data_plot[ik,1:] = np.log(E_Fl).imag/T
    # print ka,
    # for a in E_k:
    #     print a.real,

    # print
for i in range(1,NN):
    pl.plot(data_plot[:,0], data_plot[:,i], '.k', markersize=1)
    
pl.show()
        
