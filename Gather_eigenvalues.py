import numpy as np
import sys
import os
import h5py as hp
from scipy.interpolate import interp1d as intp1d

# Path to store data
fpath='D:\Work_from_Home\PPF_Mean_instability\LSA\Python_scripts\OSS_RANS\eigenvalue_data'

os.chdir(fpath)
from OSSRANS import ossrans    
from chebdif import chebdif

## Define Cess function
def cess_fit(Retau):
        
    def cess(y,k,A):
        
        Re=Retau**2/2
        nu=1/Re
        
        yp=Retau*(np.abs(y)-1)
        
        nut=(nu/2)*np.sqrt( 1 + (k**2*Retau**2/9) * (1-y**2)**2 * (1+2*y**2)**2 * (1 - np.exp(yp/A))**2  ) - (nu/2)
        
        return nut
    
    return cess


## Retau and domain sizes
Retau=[110,102,100,98,96]
dm=[100,50,25,'4pi']

Lx=np.array([100,50,25,4*np.pi])
Lz=np.array([100,50,25,4*np.pi])/2

Nx=np.array([682,340,170,84])
Ny=65
Nz=Nx.copy()

## Consildate data and construct single file
os.chdir(fpath)

for i in range(0,len(Retau)):
    
    for j in range(0,len(dm)):
        
        alpha=np.fft.fftfreq(Nx[j],Lx[j]/Nx[j])*2*np.pi
        beta=np.fft.fftfreq(Nz[j],Lz[j]/Nz[j])*2*np.pi
        
        ## Take only positive alpha and beta
        alpha=alpha[alpha>=0]
        beta=beta[beta>=0]
        
        if j==0:
            
            ## variable for recording the data
            wRANS=np.zeros([len(alpha),len(beta),2*(Ny-2)],dtype='complex128')
            wCess=np.zeros([len(alpha),len(beta),2*(Ny-2)],dtype='complex128')
            wMean=np.zeros([len(alpha),len(beta),2*(Ny-2)],dtype='complex128')

        os.chdir(fpath)
        # read the data
        with hp.File("Mean_data.h5","r") as fp:
                        
            y=fp[f'Lx=2Lz={dm[j]}/Re_tau_{Retau[i]}/y'][:]
            y=np.reshape(y,[len(y),1])
            ubar=np.reshape(fp[f'Lx=2Lz={dm[j]}/Re_tau_{Retau[i]}/ubar'][:],[len(y),1])
            uv=np.reshape(fp[f'Lx=2Lz={dm[j]}/Re_tau_{Retau[i]}/uv'][:],[len(y),1])
            
        
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
        
        pcount=0
        for m in range(0,len(alpha)):
            
            for n in range(0,len(beta)):
                
                OS_RANS = ossrans(Ny-2,alpha[m],beta[n],umean,nuT) 
                eigRANS,_ = np.linalg.eig(OS_RANS)
                wRANS[m,n,:]=eigRANS
                                
                OS_Cess = ossrans(Ny-2,alpha[m],beta[n],umean,nuCess) 
                eigCess,_ = np.linalg.eig(OS_Cess)
                wCess[m,n,:]=eigCess
                
                OS_Mean = ossrans(Ny-2,alpha[m],beta[n],umean,nuMean) 
                eigMean,_ = np.linalg.eig(OS_Mean)
                wMean[m,n,:]=eigMean
                
                pcount+=1
                
                print(f"Completed {dm[j]}: {Retau[i]} -> {(pcount/(len(alpha)*len(beta)))*100}%")
                #print(f"Completed {dm[j]}: {Retau[i]} -> alpha={alpha[m]}; beta={beta[n]}")
                sys.stdout.flush()
                

        ## Write to file
        ## File names
        fls=["RANS","Cess","Mean"]
        os.chdir(fpath)
        for fp in range(0,len(fls)):
            
            with hp.File(f"{fls[fp]}_Retau_{Retau[i]}_Lx_{dm[j]}.hdf5", "w") as f:
                
                f.create_dataset("Retau", np.shape(Retau), dtype='float64',data=Retau)
                f.create_dataset("Lx", np.shape(Lx), dtype='float64',data=Lx)
                f.create_dataset("alpha", np.shape(alpha), dtype='float64',data=alpha)
                f.create_dataset("beta", np.shape(beta), dtype='float64',data=beta)
                
                if fp==0:
                    f.create_dataset("Eignevalues", np.shape(wRANS), dtype='complex128',data=wRANS)
                if fp==1:
                    f.create_dataset("Eignevalues", np.shape(wCess), dtype='complex128',data=wCess)
                if fp==2:
                    f.create_dataset("Eignevalues", np.shape(wMean), dtype='complex128',data=wMean)
                