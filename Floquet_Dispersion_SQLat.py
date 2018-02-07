# Program to calculate the Band Structure of Floquet operator
# in a Square lattice with stroboscopic driven field
# This program is written to regenerate the results of Ref:
# M. Rudner, et. al. "Anomalous Edge States and the Bulk-Edge
# Correspondence for Periodically Driven 2D Systems."
# PRX, 3, 031005 (2013)
# 
# Author: Amin Ahmdi
# Date(i): Feb 5, 2018

# ################################################################
import numpy as np
import numpy.linalg as lg
import Floquet_2D as FL
############################################################
##############         Main Program     ####################
############################################################
N_k = 100                                # Num of k-points, 
                                         # odd number to exclude zero
N_t = 5                                  # Num of time intervals
T = 1.                                   # One period of driven field

# During each iterval T/N_t, the Hamiltonian is time-independent
mlat = int(input("Graphene strip width: "))   # width of strip
NN = 2*mlat
H_k = np.zeros((NN,NN), dtype=complex)   # k-representation H
E_k = np.zeros((NN), dtype=complex)      # eigenenergies
E_real = np.zeros((NN), dtype=float)     # eigenenergies
psi_k = np.zeros((NN,NN), dtype=complex) # matrix of eigenvectors

# different hopping amplitude
dAB = 0.*np.pi
J = 1.5*np.pi                             # hopping amplitude 
data_plot = np.zeros((N_k, NN+1), dtype=float)

# loop over k, first BZ 
for ik in range(N_k):

    ka = ik*2*np.pi/N_k
    
    M_eff = np.eye((NN), dtype=complex)   # aux matrix
    for it in range(N_t):
        if   (it==0 ):
            Jtup = (J,0,0,0)
        elif (it==1):
            Jtup = (0,J,0,0)
        elif (it==2):
            Jtup = (0,0,J,0)
        elif (it==3):
            Jtup = (0,0,0,J)
        elif (it==4):
            Jtup = (0,0,0,0)

        # construct the Hamiltonian and hopping matrices
        h, tau = FL.make_sq(mlat,dAB, *Jtup)
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
        M1 = (np.exp((-1.j/5.)*E_k*T) * U_inv.T).T
        #MM = np.dot(U_inv,np.dot(H_M, U))
        MM = np.dot(U,M1)
        M_eff = np.dot(M_eff,MM)


    E_Fl, UF = lg.eig(M_eff)
    E_real = np.log(E_Fl).imag
    E_sort = np.sort(E_real)
    data_plot[ik,0] = ka
    data_plot[ik,1:] = E_sort/(T)
    
# save the data
np.savetxt("./Result/FL_disSQ.dat", data_plot, fmt="%.2e")
############################################################
############              Plot           ###################
############################################################
# Use plot_FL.py file for plotting

import matplotlib.pyplot as pl
fig, ax = pl.subplots(figsize=(6,6))
mm = ['-r', '-k', '-c', '-b', '-y', '-g']
for i in range(1,NN+1):
    pl.plot(data_plot[:,0], data_plot[:,i], '-k', markersize=1)

ax.set_xlim(0,2*np.pi)
ax.set_xticks([0, 2*np.pi])
ax.set_xticklabels([r'$0$', r'$2\pi$'])
ax.set_xlabel(r'$k_\|$')

ax.set_ylim(-np.pi,np.pi)
ax.set_yticks([-np.pi, np.pi])
ax.set_yticklabels([r'$-\pi$', r'$\pi$'])
ax.set_ylabel('Quasienergy')

pl.show()
        
