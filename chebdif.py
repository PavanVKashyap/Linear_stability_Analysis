import numpy as np
from scipy.linalg import toeplitz

def chebdif(N,M):
    
    """    
    Original Matlab function downloaded from https://appliedmaths.sun.ac.za/~weideman/research/differ.html
    Article : Weideman, J. A. & Reddy, S. C. A MATLAB differentiation matrix suite. ACM Trans. Math. Softw. 26, 465–519 (2000).
    
    -------------------------
    The function [x, DM] =  chebdif(N,M) computes the differentiation 
    matrices D1, D2, ..., DM on Chebyshev nodes. 
   
    Input:
    N:        Size of differentiation matrix.        
    M:        Number of derivatives required (integer).
    Note:     0 < M <= N-1.
  
    Output:
    DM:       DM(1:N,1:N,ell) contains ell-th derivative matrix, ell=1..M.
    ---------------------------
    
    The code implements two strategies for enhanced 
    accuracy suggested by W. Don and S. Solomonoff in 
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The two strategies are (a) the use of trigonometric 
    identities to avoid the computation of differences 
    x(k)-x(j) and (b) the use of the "flipping trick"
    which is necessary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.
    Note added May 2003:  It may, in fact, be slightly better not to
    implement the strategies (a) and (b).   Please consult the following
    paper for details:   "Spectral Differencing with a Twist", by
    R. Baltensperger and M.R. Trummer, to appear in SIAM J. Sci. Comp. 
  
    J.A.C. Weideman, S.C. Reddy 1998.  Help notes modified by 
    JACW, May 2003.
    
    """
    
    I = np.identity(N)                  # Identity Matrix
    L = I==1                            # Logical Identity
    
    n1 = np.floor(N/2).astype(int)                  # Indices used for flipping trick.
    n2 = np.ceil(N/2).astype(int)
    
    k = np.arange(0,N).reshape(N,1)     # Compute theta vecor
    th = k*np.pi/(N-1)
    
    x = np.sin(np.pi*np.arange(N-1,-1-N,-2).reshape(N,1) / (2*(N-1)))   # Compute Chebyshev points
    
    T = np.tile(th/2,[1,N])
    DX = 2*np.sin((np.transpose(T)+T))*np.sin(np.transpose(T)-T)    # Trignometric Identity
    DX = np.vstack(((DX[0:n1,:]),
                    (-np.flipud(np.fliplr(DX[0:n2,:])))))           # Flipping trick
    DX[L] = np.ones(N)                                              # Put 1's on the main diagonal of DX.
    
    C = toeplitz((-1*np.ones(N).reshape(N,1))**k)                   # C is the matrix with entries c(k)/c(j)
    C[0,:] = C[0,:]*2
    C[N-1,:] = C[N-1,:]*2
    C[:,0] = C[:,0]/2
    C[:,N-1] = C[:,N-1]/2
    
    Z = 1/DX                                                      # Z contains entries 1/(x(k)-x(j)) with zeros on the diagonal.
    Z[L] = np.zeros(N)
    
    D = np.identity(N)                                              # D contains diff. matrices.
    DM = np.zeros([N,N,M])
    
    for ell in range(1,M+1):
        
        D = ell*Z*(C*np.tile(np.diag(D).reshape(N,1),[1,N]) - D)                 # Off-diagonals
        D[L] = -np.sum(D.T,axis=0)                                  # Correct main diagonal of D
        DM[:,:,ell-1] = D                                             # Store current D in DM
        
    return x,DM