# Python functions for computing pressure and density vertical modes
#   and vertical structure of SQG solutions
# Cesar B Rocha
# SIO, Summer 2014

import numpy as np
import scipy as sp

def Dm(n):
    """ it creates forward difference matrix
            n: number of rows/columns """

    a = [1.,0.]
    b = sp.zeros(n-2)
    row1 = np.concatenate([a,b])
    a = [1.,-1.]
    col1 = np.concatenate([a,b])

    # Dn: 1st order centered differences
    D = sp.linalg.toeplitz(col1,row1)

    return D

# four special matrices
def kctb(n):
    """ it creates four 'special' matrices for
            second order 1d problem.
            n: number of rows/columns """

    a = [2.,-1.]
    b = sp.zeros(n-2)
    row1 = np.concatenate([a,b])

    # Kn: Stiffness matrix (second order difference)
    K = sp.linalg.toeplitz(row1)

    # Cn: Circulant  matrix 
    C = sp.copy(K); C[0,n-1] = -1; C[n-1,0] = -1

    # Tn: Kn with changed the upper bc
    T = sp.copy(K); T[0,0] = 1

    # Bn: Tn with changed the lower bc
    B = sp.copy(T); B[n-1,n-1] = 1

    return K, C, T, B

# orthonormal eigenvector
def normal_evector(E,z):
    """ it normalizes eigenvectors such that 
           sum(Ei^2 x dz/H) = 1    """
    ix,jx =  E.shape
    dz = np.float(np.abs(z[1]-z[0]))
    H = dz*ix
    for j in range(jx):
        s = np.sqrt( (E[:,j]**2).sum()*(dz/H) )
        E[:,j] = E[:,j]/s
    return E

# pressure N2 modes
def pmodes(N2,z,lat,nm):

    """ it computes the pressure stratification (N2)
       modes in a equispaced z-grid by solving
       d/dz( f2/N2(z) * d/dz ) F + rd^2 F = 0
       subject to dF/dz  = 0 @ z = 0, -H
     lat = local latitude and 
     nm  = number of modes to return """

    # settings
    dz = np.float(np.abs(z[1]-z[0]))
    f2 = (2.*(7.29e-5)*np.sin(lat*(np.pi/180)))**2

    # assembling matrices
    C = np.matrix(np.diag(f2/N2))
    D = Dm(N2.size)

    K = (D.T*C*D)/(dz**2)
    
    # upper bc  (bottom bc is already satisfied)
    K[0,0] = -K[0,1]

    # eigenvalue problem
    w,v = np.linalg.eigh(K) 
    w = np.sort(w)
    v = v[:,w.argsort()]
    w = np.array(w[:nm])
    v = np.array(v[:,:nm])

    # normalize to make eigvecs orthonormal
    v = normal_evector(v,z)

    # deformation radius [km]
    rd = np.zeros(w.size)
    rd[1:] = 1/np.sqrt(w[1:])/1.e3
    rd[0] = np.sqrt(9.81*np.abs(z[-1]))/np.sqrt(f2)

    return v,rd

# density modes
def rhomodes(N2,z,lat,nm):

    """ it computes the density stratification (N2)
       modes in a equispaced grid z by solving
       d2/dz2 F + N2(z)/(g Dn) F = 0
       subject to F  = 0 @ z = 0, -H
     lat = local latitude and 
     nm  = number of modes to return 
     Dn is the eigenvalue which is, by denifition,
        an 'equivalent depth' """

    # settings
    dz = np.float(np.abs(z[1]-z[0]))
    f = 2.*np.abs((7.29e-5)*np.sin(lat*(np.pi/180)))
    g = 9.81 # [m/s^2]

    # assembling matrices
    C = np.matrix(np.diag(N2/g))

    K,_,_,_ = kctb(N2.size)

    # eigenvalue problem
    w,v = sp.linalg.eigh(K/(dz**2),C) 
    w = np.sort(w)
    v = v[:,w.argsort()]
    w = np.array(w[:nm])
    v = np.array(v[:,:nm])

    # normalize to make eigvecs orthonormal
    v = normal_evector(v,z)

    # deformation radius [km]
    rd = np.sqrt(g/w)/f/1.e3

    return v,rd

# vertical structure of SQG solution 
def sqgz(N2,z,lat,k):

    """ it computes vertical structure of SQG solutions
       in a regular grid z [m] by solving
       d/dz( f2/N2 * d/dz )F - k^2 F = 0
       subject to dF/dz  = 1 @ z = 0
       and dF/dz = 0 @ z = -H
     N2 = stratification squared [(cps)^2]
     lat = local latitude
     k = wavenumber [cpm] """

    # settings
    dz = np.float(np.abs(z[1])-np.abs(z[0]))
    f2 = (2.*(7.29e-5)*np.sin(lat*(np.pi/180)))**2

    # assembling matrices
    C = np.matrix(np.diag(f2/N2))
    D = Dm(N2.size)

    K = -(D.T*C*D)/(dz**2) - np.matrix(np.eye(N2.size))*(k**2)

    # upper bc  (bottom bc is already satisfied)
    K[0,0] = -K[0,1]

    f = np.zeros(N2.size)

    # point load (surface buoyancy)
    A = dz/(C[0,0]-(k**2))
    f[0] = A 
    
    v = sp.linalg.solve(K,f)
    
    # normalize such that v[0] = 1
    v = v/(v[0])

    return  v
