# Program to calculate the Bandstructure of "augmented" effective Floquet 
# Hamiltonian in a honeycomb lattice with stroboscopic driven field.

# Refrences:
# (1) T. Kitagawa, et. al. "Topological characterization of periodically driven
# Study of Floquet system based on paper Phys. Rev. B 82, 235114 (2010)

# (2) Mark S. Rudner, et. al. "Anomalous Edge States and the Bulk-Edge 
# Correspondence for Periodically Driven Two-Dimensional Systems", 
# Phys. Rev. X 3, 031005 (2013)
#
# (3)  Netanel H. Lindner, et. al. "Quantized large-bias current in the 
# anomalous Floquet-Anderson insulator", arXiv:1708.05023 (2017) 


# Author: Amin Ahmdi
# Date: May 22, 2018
#################################################################
import numpy as np
import numpy.linalg as lg
import Floquet_2D as FL         # external functions

############################################################
##############         Functions        ####################
############################################################
def H_aug_generate(N_harmonic, mlat, J):
    """ construct the augmented field periodic Hamiltonian.
    input:
    ------
    N_harmonic: int, number of harmonic that H(t) will be expanded to
    mlat: int, width of armchair graphene
    J: float, hopping amplitude

    return:
    -------
    H_aug: (aug_dim,aug_dim) complex, augmented Hamiltonian
    tau_aug: (aug_dim,aug_dim) complex, augmented Hopping matrix
    taudg_aug: (aug_dim,aug_dim) complex, augmented Hopping matrix (H.C.) 
    """
    T = 1
    Omega = 2*np.pi/T                    # Frequency of the driven field
    NN = 2*mlat                          # number of sites in one translational unitcell
    aug_dim = NN*(2*N_harmonic+1)        # dimension of augmented Hamiltonian
    H_aug = np.zeros((aug_dim, aug_dim), dtype=complex)
    tau_aug = np.zeros((aug_dim, aug_dim), dtype=complex)
    taudg_aug = np.zeros((aug_dim, aug_dim), dtype=complex)   

    for i, im in enumerate(range(-N_harmonic,N_harmonic+1)):
        H_aug[i*NN:(i+1)*NN,i*NN:(i+1)*NN] += -im*Omega*np.eye(NN) 
        for j, il in enumerate(range(-N_harmonic,N_harmonic+1)):
            if (il == im):
                aux = 1./N_t 
                aux2 = 0
            else:
                aux = (2.j*np.sin( (il-im)*np.pi/N_t ) ) / ((il-im)*Omega)
                aux2 = 2.j*np.pi*(il-im)/N_t
                
            haux = np.zeros((NN,NN), dtype=complex)
            taux = np.zeros((NN,NN), dtype=complex)
            tdgaux = np.zeros((NN,NN), dtype=complex)
                
            for it in range(N_t):
                if   (it==0 ):
                    Jtup = (delta*J,J,J)
                elif (it==1):
                    Jtup = (J,delta*J,J)
                elif (it==2):
                    Jtup = (J,J,delta*J)

                # construct the Hamiltonian and hopping matrices
                h, tau = FL.make_Gr(mlat, *Jtup)
                tau_dg = tau.conj().T

                haux += h*np.exp(aux2*(it+1))
                taux += tau*np.exp(aux2*(it+1))
                tdgaux += tau_dg*np.exp(aux2*(it+1))
            # End of it-loop
            # print('i,j are: ', i, j, im, il)
            # print('M dim: ', i*NN, (i+1)*NN,j*NN, (j+1)*NN)
            H_aug[i*NN:(i+1)*NN,j*NN:(j+1)*NN] += aux*haux
            tau_aug[i*NN:(i+1)*NN,j*NN:(j+1)*NN] = aux*taux
            taudg_aug[i*NN:(i+1)*NN,j*NN:(j+1)*NN] = aux*tdgaux
        # endof il-loop
    #endof im-loop

    return H_aug, tau_aug, taudg_aug
##################################################

############################################################
##############         Main Program     ####################
############################################################
N_k = 100                                # Num of k-points, 
N_t = 3                                  # Num of time intervals
T = 1.                                   # One period of driven field
N_harmonic = 3
mlat = int(input("Graphene strip width: "))   # width of strip
NN_aug = 2*mlat*(2*N_harmonic+1)

print('Dimension of augmented Hamiltonian: ', NN_aug)
H_aug = np.zeros((NN_aug,NN_aug), dtype=complex)   # k-representation H
E_k = np.zeros((NN_aug), dtype=complex)      # eigenenergies

# different hopping amplitude
delta = float(input("Enter the hopping difference coefficient: ")) 
J = np.pi/16.                             # hopping amplitude 
data_plot = np.zeros((N_k, NN_aug), dtype=float)

ik = 0
for ka in np.linspace(-np.pi, np.pi, N_k):

    H_aug, tau_aug, tau_augdg  = H_aug_generate(N_harmonic, mlat, J)

    # Construct matrix: [h + tau*exp(ika) + tau^+ * exp(-ika)]
    # and diagonalization
    H_k = H_aug + np.exp(1.j*ka)*tau_aug + np.exp(-1.j*ka)*tau_augdg

    # return eigenenergies and vectors
    E_k = lg.eigvals(H_k)    
    
    data_plot[ik] = np.sort(E_k.real)
    ik += 1
############################################################
############              Plot           ###################
############################################################
# Use plot_FL.py file for plotting

import matplotlib.pyplot as pl
fig, ax = pl.subplots(1)

ka = np.linspace(-np.pi, np.pi, N_k)
mm = ['-r', '-k', '-c', '-b', '-y', '-g']
for i in range(NN_aug):
    pl.plot(ka, data_plot[:,i], '-k', markersize=1)
    
pl.show()
        
