import numpy as np
from chebdif import chebdif
from cheb4c import cheb4c

"""
Buld the Orr-Sommerfeld-Squire system using the Chebyshev collocation


"""

def ossrans(N,alpha,beta,u,nuT):
    
    ## Gather diffrentiation matrices
    y,DM = chebdif(N+2,3)
    
    ## Get the fourth order differentiation matrix with boundary conditions
    D4 = cheb4c(N+2)
    
    ## Differentiate the base flow
    up = np.matmul(DM[:,:,0],u)
    upp = np.matmul(DM[:,:,1],u)
    
    ## Derivatives of nu
    nuTp=np.matmul(DM[:,:,0],nuT)
    nuTpp=np.matmul(DM[:,:,1],nuT)
    
    ## Implement the homogeneous boundary conditions
    D1 = DM[1:N+1,1:N+1,0]
    D2 = DM[1:N+1,1:N+1,1]
    D3 = DM[1:N+1,1:N+1,2]
        
    ## Wave vector
    k2 = alpha**2 + beta**2
    
    ## substitutions
    I=np.identity(N)
    K2 = k2*I
    K4 = k2**2*I
    U=u[1:N+1]*I
    Up=up[1:N+1]*I
    Upp=upp[1:N+1]*I
    
    nu=nuT[1:N+1]*I
    nup=nuTp[1:N+1]*I
    nupp=nuTpp[1:N+1]*I
    
    ## Build the Orr-Sommerfeld and Squire terms
    LOS = 1j*alpha*np.matmul(U,(D2-K2)) - np.matmul(nupp,(D2+K2)) - 2*np.matmul(nup,(D3 - np.matmul(K2,D1))) - 1j*alpha*Upp - np.matmul(nu,(K4 + D4 - 2*np.matmul(K2,D2)))
    LSQ = 1j*alpha*U - np.matmul(nup,D1) - np.matmul(nu,(D2-K2)) 
    
    
    ## Build the OSS matrix
    OSSRANS=np.block([[np.matmul(LOS,np.linalg.inv(D2-K2)), np.zeros([N,N])],
                 [1j*beta*Up , LSQ]
                ])
    
    return OSSRANS