# Program to calculate the Band Structure of Floquet operator
# in a honeycomb lattice with stroboscopic driven field
# for a square lattice
# This program is written to regenerate the results of Ref:
# T. Kitagawa, et. al. "Topological characterization of periodically driven
# Study of Floquet system based on paper Phys. Rev. B 82, 235114 (2010)
# 
# Author: Amin Ahmdi
# Date(i): Oct 11, 2017
# Date(3): Nov  6, 2017   more structured version
# Date(4): Dec 20, 2017   fix the time interval and
# the sorting problem. In Ref paper instead of T/3
# for each time-interval T had been considered.

# note that the square lattice part is not ready to use 
# ################################################################
import numpy as np
import numpy.linalg as lg
import Floquet_2D as FL
############################################################
##############         Main Program     ####################
############################################################
N_k = 100                                # Num of k-points, 
                                         # odd number to exclude zero
N_t = 3                                  # Num of time intervals
T = 1.                                   # One period of driven field

# During each iterval T/N_t, the Hamiltonian is time-independent
mlat = int(input("Graphene strip width: "))   # width of strip
NN = 2*mlat
H_k = np.zeros((NN,NN), dtype=complex)   # k-representation H
E_k = np.zeros((NN), dtype=complex)      # eigenenergies
E_real = np.zeros((NN), dtype=float)     # eigenenergies
psi_k = np.zeros((NN,NN), dtype=complex) # matrix of eigenvectors
# UF_F = np.zeros((mlat,mlat), dtype=complex) # U up to Fermi level
W_loop = np.eye((mlat), dtype=complex)# matrix to to Wilson loop calculation

# different hopping amplitude
delta = float(input("Enter the hopping difference coefficient: ")) 
J = np.pi/16.                             # hopping amplitude 
data_plot = np.zeros((N_k, NN+1), dtype=float)

# loop over k, first BZ 
for ik in range(N_k):

    ka = -np.pi + ik*(2.*np.pi)/(N_k)
    # ka = ik*(np.pi/N_k)
    M_eff = np.eye((NN), dtype=complex)   # aux matrix
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
    indx = np.argsort(E_real)
    E_sort = E_real[indx]
    UF_F = UF[indx[:int(NN/2)]]
    UF_inv = UF_F.T.conj()
    # print("UF shape is ", UF_F.shape)
    # print("UF_inv shape is ", UF_inv.shape)
    # print("W_loop shape is ", W_loop.shape)
    # Wilson loop winding number calculation
    if (ik%2 == 0):
        W_loop = np.dot(W_loop, UF_F)
    elif (ik%2==1):
        W_loop = np.dot(W_loop, UF_inv)
    
    data_plot[ik,0] = ka
    data_plot[ik,1:] = E_sort/(T)

wind_num = (np.log(lg.det(W_loop)).imag)/(2.*np.pi)
# wind_num = lg.det(W_loop)
print(wind_num)
# save the data
np.savetxt("./Result/FL_disert.dat", data_plot, fmt="%.2e")
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
        
