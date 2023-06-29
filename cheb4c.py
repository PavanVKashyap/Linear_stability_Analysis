import numpy as np

def cheb4c(N):
    
    """    
    Original Matlab function downloaded from https://appliedmaths.sun.ac.za/~weideman/research/differ.html
    Article : Weideman, J. A. & Reddy, S. C. A MATLAB differentiation matrix suite. ACM Trans. Math. Softw. 26, 465–519 (2000).
    
    -------------------------------
    The function [x, D4] =  cheb4c(N) computes the fourth 
    derivative matrix on Chebyshev interior points, incorporating 
    the clamped boundary conditions u(1)=u'(1)=u(-1)=u'(-1)=0.
  
    Input:
    N:     N-2 = Order of differentiation matrix.  
                 (The interpolant has degree N+1.)
  
    Output:
    x:      Interior Chebyshev points (vector of length N-2)
    D4:     Fourth derivative matrix  (size (N-2)x(N-2))
    --------------------------------------
    
    The code implements two strategies for enhanced 
    accuracy suggested by W. Don and S. Solomonoff in 
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The two strategies are (a) the use of trigonometric 
    identities to avoid the computation of differences 
    x(k)-x(j) and (b) the use of the "flipping trick"
    which is necessary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.
    
    J.A.C. Weideman, S.C. Reddy 1998.
    
    """    
    I = np.identity(N-2)                  # Identity Matrix
    L = I==1                            # Logical Identity
    
    n1 = np.floor(N/2-1).astype(int)                  # Indices used for flipping trick.
    n2 = np.ceil(N/2-1).astype(int)
    
    k = np.arange(1,N-1).reshape(N-2,1)     # Compute theta vecor
    th = k*np.pi/(N-1)
    
    x = np.sin(np.pi*np.arange(N-3,1-N,-2).reshape(N-2,1) / (2*(N-1)))   # Compute interior Chebyshev points
    
    s = np.vstack(((np.sin(th[0:n1])),                                  # s = sin(th))
                   (np.flipud(np.sin(th[0:n2])))))
    
    alpha = s**4                                                    # Compute the weight functions and its derivatives
    beta1 = -4*s**2*x/alpha
    beta2 = 4*(3*x**2-1)/alpha 
    beta3 = 24*x/alpha
    beta4 = 24/alpha 
    
    B = np.vstack(((beta1.T),
                   (beta2.T),
                   (beta3.T),
                   (beta4.T)))
    
    
    T = np.tile(th/2,[1,N-2])
    DX = 2*np.sin((np.transpose(T)+T))*np.sin(np.transpose(T)-T)    # Trignometric Identity
    DX = np.vstack(((DX[0:n1,:]),
                    (-np.flipud(np.fliplr(DX[0:n2,:])))))           # Flipping trick
    DX[L] = np.ones(N-2)                                              # Put 1's on the main diagonal of DX.
    
    ss = s**2*((-1*np.ones(N-2).reshape(N-2,1))**k)                     # Compute the matrix entrie c(k)/c(j)
    S = np.tile(ss,[1,N-2])
    C = S/S.T

    Z = 1/DX                                                      # Z contains entries 1/(x(k)-x(j)) with zeros on the diagonal.
    Z[L] = np.zeros(N-2)
    
    X = Z.T
    indx=np.setxor1d(np.arange(0,(N-2)**2),np.arange(0,(N-2)**2,N-1))
    X = np.reshape(np.ravel(X,order='F')[indx],(N-3,N-2),order='F')
    
    Y = np.ones([N-3,N-2])                                          # Initialize Y and D vectors. Y contains matrix of cumulative sums
    D = np.identity(N-2)                                            # D scaled differaentiation matrices
    DM = np.zeros([N-2,N-2,4])
    
    for ell in range(1,5):
        
        Y = np.cumsum(np.vstack(((B[ell-1,:]),
                                (ell*Y[0:N-3,:]*X))),axis=0)
        
        D = ell*Z*(C*np.tile(np.diag(D).reshape(N-2,1),[1,N-2]) - D)                 # Off-diagonals
        D[L] = Y[N-3,:]                                  # Correct main diagonal of D
        DM[:,:,ell-1] = D                                             # Store current D in DM
        
    return DM[:,:,3]