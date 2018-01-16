# Program to calculate the quasi-energies associated with
# Floquet operator in a honeycomb lattice with stroboscopic
# driven field. This code is implemented for a system with
# priodic boundary condition.
# Author: Amin Ahmadi
# Date(i): Oct 11, 2017 
# Date(1): Oct 18, 2017
# Date(2): Dec 23, 2017
# ################################################################
import numpy as np
import numpy.linalg as lg

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
def make_Gr_wPBC(mlat, J1=1, J2=1, J3=1):
    """ Constructs the Hamiltonian and the connection 
    matrix of an armchair graphene strip with PBC
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

    # PBC
    h[mlat-1,0] = J2
    h[2*mlat-1,mlat] = J1
    
    h = h + h.conj().T          # make it hermitian
    # Hopping matrix
    for i in range(1,mlat,2):
        tau[i+mlat,i] = J3

    return h, tau



############################################################
##############         Main Program     ####################
############################################################
N_k = 101                                # Num of k-points
N_t = 3                                  # Num of time intervals
T = 1.                                   # One period of driven field
# During each iterval T/N_t, the Hamiltonian is time-independent
mlat = input("Graphene strip width: ")                # width of strip
if (mlat%2 == 1):
    print "Error: To have PBC width must be even number"
    print "Try again!"
    print "============================================"
    exit()
    
NN = 2*mlat
H_k = np.zeros((NN,NN), dtype=complex)   # k-representation H
E_k = np.zeros((NN), dtype=complex)      # eigenenergies
E_real = np.zeros((NN), dtype=float)      # eigenenergies
psi_k = np.zeros((NN,NN), dtype=complex) # matrix of eigenvectors

# different hopping amplitude
delta = input("Enter the hopping difference coefficient: ") 
J = np.pi/16.                             # hopping amplitude 
data_plot = np.zeros((N_k, NN+1), dtype=float)

# loop over k, first BZ 
for ik in range(N_k):

    ka = -np.pi/3. +  ik*(2.*np.pi)/(3.*N_k)
    # ka = ik*(np.pi/N_k)
    M_eff = np.eye((NN), dtype=complex)   # aux matrix
    for it in range(N_t):
        if   (it==0 ):
            J1=delta*J; J2=J; J3=J
        elif (it==1):
            J1=J; J2=delta*J; J3=J
        elif (it==2):
            J1=J; J2=J; J3=delta*J

        # construct the Hamiltonian and hopping matrices
        h, tau = make_Gr_wPBC(mlat , J1, J2, J3)
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
        M1 = (np.exp((-1.j)*E_k*T) * U_inv.T).T
        #MM = np.dot(U_inv,np.dot(H_M, U))
        MM = np.dot(U,M1)
        M_eff = np.dot(M_eff,MM)


    E_Fl, UF = lg.eig(M_eff)    
    E_real = np.log(E_Fl).imag
    E_sort = np.sort(E_real)
    data_plot[ik,0] = ka
    data_plot[ik,1:] = E_sort/(T)


## save data
np.savetxt("./Result/Fl_dis_WPB.dat", data_plot, fmt='%0.3e')
############################################################
############              Plot           ###################
############################################################
# Use plot_FL.py file for plotting

import matplotlib.pyplot as pl
fig, ax = pl.subplots(1)
mm = ['-r', '-k', '-c', '-b', '-y', '-g']
for i in range(1,NN+1):
    pl.plot(data_plot[:,0], data_plot[:,i], '-k', markersize=1)
    
pl.show()
        
