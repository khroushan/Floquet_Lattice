# Program to calculate the dispesion relation of
# square and honeycomb lattices
# Authur: Amin Ahmdi
# Date: Oct 11, 2017
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
    # hopping between sites in a column
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

############################################################
def make_Gr_R(mlat, t_so):
    """ Constructs the Hamiltonian and the connection 
    matrix of an armchair graphene strip including Rahba 
    term.
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
##############         Main Program     ####################
############################################################
N_k = 200                                # Num of k-points
mlat = int(input("Enter width of graphene strip : "))
NN = 2*mlat                              # # of site in the super-unitcell 
H_k = np.zeros((NN,NN), dtype=complex)   # k-representation H
E_k = np.zeros((NN), dtype=complex)      # eigenenergies
psi_k = np.zeros((NN,NN), dtype=complex) # matrix of eigenvectors

data_plot = np.zeros((N_k, NN+1), dtype=float)

h, tau = make_Gr(mlat)
tau_dg = tau.conj().T 
# loop over k, first BZ 
for ik in range(N_k):

    ka = -np.pi/2. + ik*(np.pi/N_k)
    
    # Construct matrix: [h + tau*exp(ika) + tau^+ * exp(-ika)]
    # and diagonalization
    H_k = h + np.exp(1.j*ka)*tau + np.exp(-1.j*ka)*tau_dg 
    E_k, psi_k = lg.eig(H_k)    # return eigenenergies and vectors

    # sort the eigenenergies
    E_sort = np.sort(E_k)
    data_plot[ik,0] = ka
    data_plot[ik,1:] = E_sort.real


#############################################################
# save data into file
np.savetxt("./Result/E_k_Gr_100.dat", data_plot, fmt='%5e')

########################################
###########      Plot      #############
########################################
import matplotlib.pyplot as pl
import matplotlib.ticker as tk

pl.rc("font", family='monospace')

# Use system latex instead of builtin
pl.rc('text', usetex=True) # It's slower


fig = pl.figure(figsize=(6,8))
ax = fig.add_subplot(1,1,1)

ax.set_xlabel(r'$ka$', fontsize=16)
ax.set_ylabel(r'$E$', fontsize=16)

ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_xticklabels([r'$-\pi/2$', r'$-\pi/4$', r'$0$',
                    r'$\pi/4$', r'$\pi/2$'], fontsize=16)
ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_yticklabels(['$-3$', '$-2$', '$-1$', '$0$', '$1$',
                    '$2$', '$3$'], fontsize=16)

ax.set_xlim([-np.pi/2, np.pi/2])


for i in range(1,NN):
    pl.plot(data_plot[:,0], data_plot[:,i], color='0.4',
            linestyle='-', markersize=1)

fig.savefig("./Result/E_k_Gr_100.pdf", bbox_inches='tight')
pl.show()
        
