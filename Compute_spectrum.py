import numpy as np
import sys
import os
import h5py as hp
from scipy.interpolate import interp1d as intp1d
import matplotlib.pyplot as plt
import matplotlib

from chebdif import chebdif
from OSSRANS import ossrans  


## Define Cess function
def cess_fit(Retau):
        
    def cess(y,k,A):
        
        Re=Retau**2/2
        nu=1/Re
        
        yp=Retau*(np.abs(y)-1)
        
        nut=(nu/2)*np.sqrt( 1 + (k**2*Retau**2/9) * (1-y**2)**2 * (1+2*y**2)**2 * (1 - np.exp(yp/A))**2  ) - (nu/2)
        
        return nut
    
    return cess



## Retau 
Retau=[110,102,100,98,96]
dm=[100,50,25,'4pi']

# Domain sizes
Lx=np.array([100,50,25,4*np.pi])
Lz=np.array([100,50,25,4*np.pi])/2

# Numerical Resolution
Nx=np.array([682,340,170,84])
Ny=65
Nz=Nx.copy()

## File names
fls=["RANS","Cess","Mean"]
mrks=['X','^','o']
flstyle=['full','none','none']

###  Plot details
figsize=[7,5]
## Set the clor scheme
clrs=['blue','purple','orange','green','red']

## Select the domain for sample plotting
lxp=100

## Select the domain
lindx=np.argmin(np.abs(Lx-lxp))
lbls=[r"$\nu_t(DNS)$",r"$\nu_{Cess}$",r"$\nu$"]

## Wavenumber for computing the spectra
alpha=0.18
beta=0.42

for i in range(0,len(Retau)):
    
    with hp.File("Mean_data.h5","r") as fp:
        
        y=fp[f'Lx=2Lz={dm[lindx]}/Re_tau_{Retau[i]}/y'][:]
        y=np.reshape(y,[len(y),1])
        ubar=np.reshape(fp[f'Lx=2Lz={dm[lindx]}/Re_tau_{Retau[i]}/ubar'][:],[len(y),1])
        uv=np.reshape(fp[f'Lx=2Lz={dm[lindx]}/Re_tau_{Retau[i]}/uv'][:],[len(y),1])
            
    
    ## Laminar flow
    ulam=1-y**2  
    
    ## Mean flow
    umean=ubar+ulam
    
    ## Derivative matrix
    _,DM = chebdif(len(y),2)
    
    ## Compute the derivatives
    Up=np.matmul(DM[:,:,0],umean)
    Upp=np.matmul(DM[:,:,1],umean)
    uvp=np.matmul(DM[:,:,0],uv)
    
    ## Nut
    nut=uv/Up
    nut=nut.ravel()
    
    ## Nut with the L'Hospitals rule
    nut0=uvp/Upp
    
    ## Ravel the data for interpolation
    nut0=nut0.ravel()
    y=y.ravel()
    
    ## Indices of the center and its immediate neighbours
    indx=np.argmin(np.abs(y-0))
    indx1,indx2=indx-2,indx+3
    
    ## Construct nut with the center as l'Hospital
    nutlh=np.concatenate((nut[:indx1],np.array([nut0[indx]]),nut[indx2:]))
    ylh=np.concatenate((y[:indx1],np.array([y[indx]]),y[indx2:]))
    
    ## Interpolate for the immediate neighbours
    nuintp=intp1d(ylh,nutlh,kind='cubic')
    
    ## Interpolated values
    y1=np.arange(indx1,indx)
    nut1=nuintp(y[y1])
    
    y2=np.arange(indx+1,indx2)
    nut2=nuintp(y[y2])
    
    ## Assemble nuT
    nuT=np.concatenate((nut[:indx1],nut1,np.array([nut0[indx]]),nut2,nut[indx2:])) + (2.0/Retau[i]**2)
    nuT=np.reshape(nuT,[len(y),1])
    
    ## Compute nuT Cess
    nuCess=cess_fit(Retau[i])(y,0.426,25.4) + (2.0/Retau[i]**2)
    nuCess=np.reshape(nuCess,[len(y),1])
    
    ## nu without Eddy viscosity
    nuMean=(2.0/Retau[i]**2)*np.ones(len(y))
    nuMean=np.reshape(nuMean,[len(y),1])
    
    
    ## Assemble the OSS matrix for Re stress data from DNS and compute the eigen values
    OS_RANS = ossrans(Ny-2,alpha,beta,umean,nuT) 
    eigRANS,_ = np.linalg.eig(OS_RANS)
    wRANS=-1j*eigRANS
    
    ## Assemble the OSS matrix for Cess profile and compute the eigen values
    OS_Cess = ossrans(Ny-2,alpha,beta,umean,nuCess) 
    eigCess,_ = np.linalg.eig(OS_Cess)
    wCess=-1j*eigCess
    
    ## Assemble the OSS matrix for Turbulent mean flow with constant viscosity and compute the eigen values
    OS_Mean = ossrans(Ny-2,alpha,beta,umean,nuMean) 
    eigMean,_ = np.linalg.eig(OS_Mean)
    wMean=-1j*eigMean
    
    ## Plot the spectrum
    fig,axs=plt.subplots()
    axs.plot(np.real(wRANS),np.imag(wRANS),ls='None',marker=mrks[0],fillstyle=flstyle[0],color=clrs[0],label=lbls[0])
    axs.plot(np.real(wCess),np.imag(wCess),ls='None',marker=mrks[1],fillstyle=flstyle[1],color=clrs[1],label=lbls[1])
    axs.plot(np.real(wMean),np.imag(wMean),ls='None',marker=mrks[2],fillstyle=flstyle[2],color=clrs[2],label=lbls[2])
    
    axs.set_xlabel(r"$\omega_r$",fontsize=14)
    axs.set_ylabel(r"$\omega_i$",fontsize=14)
    axs.set_xlim([0,0.1])
    axs.set_ylim([-1,0.05])
    axs.legend(loc='best')
    
    fig.tight_layout()
    fig.savefig(f"ISpectrum_Retau_{Retau[i]}.png")
    plt.close('all')
