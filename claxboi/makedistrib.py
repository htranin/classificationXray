### ----------------------------------------------
# Usage: python makedistrib.py, python makedistrib.py [float between 0 and 1]
# computes the distributions (normalized histograms) of each property and each class from the Golden Sample sources
# distributions are smoothed unless you comment the line "d=dsm"
# in test mode (when the variable "fraction" is set by the used in the command line or when testSample==1 in classify_montecarlo), use only a fraction of the GS and save the other part in a file used to test the classification (retrieval fractions, false positive rates)
# see classify_montecarlo.py for more details
# When equipart==1, the other part is tuned to respect the proportions of each class given by (nCO, nSTAR, nAGN)
### ----------------------------------------------

import numpy as np
import os
import sys
# from scipy.interpolate import splrep, splev,interp1d
from sklearn.neighbors import KernelDensity

n=100 #number of bins
equipart=False #False or list with length=len(Classes)
verbose=0
plotdistrib=0

def make(SourcesCsv, properties, Classes=[0,1,2], equipart=False, fraction=1, plotdistrib=False,N=100,verbose=False,dirout='classif/distribtest/', custom_pty=[],dumb=False, scale=None):

    if dirout:
        os.system('mkdir -p %s'%dirout)
    
    def get_peak(y):
        dy=y[1:]-y[:-1]
        cdy=dy[1:]*dy[:-1]
        return np.where(cdy<0)[0]+1

    if plotdistrib:
        import matplotlib.pyplot as plt

    selection=sum([SourcesCsv['class']==C for C in Classes]).astype(bool) #select rows with class in Classes
    SourcesCsv=SourcesCsv[selection]
    l=len(SourcesCsv)
    takehalf=np.arange(l)
    np.random.shuffle(takehalf)

    #bright=sum(np.isnan(np.vstack((SourcesCsv[takehalf][p] for p in ['logFxFIR','logFxFopt'])).T).T)!=0

    #choose partition of CO, AGN and Stars in the Test Sample (i.e. in the file otherhalf.dat)
    if np.array(equipart).any():
        #ex: [0.5,0.4,0.06,0.04]
        equipart=np.array(equipart)/sum(equipart)
        limiting_class=np.argmin([sum(SourcesCsv['class']==Classes[i])/equipart[i] for i in range(len(Classes))])
        print('limiting class: %s'%str(Classes[limiting_class]))
        #class with lowest ratio proportion/desired_proportion
        nClasses=[int(sum(SourcesCsv['class']==Classes[limiting_class])*equipart[i]/equipart[limiting_class]) for i in range(len(Classes))]
        print('counts of each class in the properly proportionned sample:',nClasses,'(total %d)'%sum(nClasses))
        partsClasses=[takehalf[SourcesCsv[takehalf]['class']==Classes[i]][-int(nClasses[i]):] for i in range(len(Classes))]
        partsClasses=[np.where(selection)[0][line] for line in partsClasses]
        
    #takehalf=takehalf[np.argsort(SourcesCsv[takehalf]['class'])[::-1]]
    SourcesCsv=SourcesCsv[takehalf[:int(fraction*l)]]
    if np.array(equipart).any():
        np.savetxt('otherhalf.dat',np.sort(np.concatenate([partsClasses[i] for i in range(len(Classes))])),fmt='%d',header='Reference Sample index')
    else:
        np.savetxt('otherhalf.dat',np.sort(takehalf[int(fraction*l):]),fmt='%d',header='Reference Sample index')

    classes=SourcesCsv['class']


    #properties=SourcesCsv.dtype.names[2:]
    #all except name,class


    Nbins=np.arange(50,100,2)
    
    for ip in range(len(properties)):
        ymax = 0
        p = properties[ip]
        data=SourcesCsv[p]
        igood=np.where(~np.isnan(data)) # index sources for which p!=NULL
        Data=data[igood]
        igoodClasses=[(classes==C)[igood] for C in Classes]
        #data=Data    # remove nan values
        nD=len(Data)
        #Data=Data[np.argsort(Data)[nD//(10*n):(10*n-1)*nD//(10*n)]]
        #later imp: generalize by removing float('%.1e'%...)
        x1=float('%.1e'%np.nanmax(Data))
        x0=x1-float('%.1e'%(np.nanmax(Data)-np.nanmin(Data)))
        if len(np.unique(Data))>=10 and dumb: #apply kde
            x0kde=x1-float('%.1e'%(np.nanmax(Data)-np.nanmin(Data[Data>np.nanmin(Data)])))
        else:
            x0kde=x0
        dx=(x1-x0kde)/N
        try:
            Bfinal=np.linspace(x0-dx,x1+dx,int(N*(x1-x0)/(x1-x0kde)))
        except:
            print(p,np.array(igoodClasses).shape,nD,((x0,x0kde),x1),dx,N, dumb)
            print("Error")
            sys.exit()
            
        #print(x0,x1,x0kde,p)
            
        xfinal=(Bfinal[:-1]+Bfinal[1:])/2
        
        dl=[]
        
        D=(x1-x0)/Nbins  # array of test binwidth
        
        
        if plotdistrib:
            plt.figure(p)
            
        for ic in range(len(Classes)):
            ind=igoodClasses[ic] # without NaNs
            B=np.linspace(x0,x1,N+5)
            d=np.histogram(Data[ind],B)[0]
            x=(B[1:]+B[:-1])/2
            d_interp=np.interp(xfinal,x,d) 

            #KDE method to smooth the histogram into a PDF
            kde=KernelDensity(bandwidth=(x1-x0kde)/(5*len(Data)**(1/5)), kernel='exponential') #rule of thumb
            try:
                kde.fit(Data[ind][:,None])
            except:
                print("Error: property %s does not contain any value for class %s"%(p,Classes[ic]))
                sys.exit()
                    
            logprob=kde.score_samples(xfinal[:,None])
            dsm_kde=np.exp(logprob)*sum(ind)
                


            if len(np.unique(Data[ind]))<10:
                #property can be considered discrete
                #=> do not smooth its distributions
                dsm_kde=np.histogram(Data[ind],Bfinal)[0]

            if [p,ic] in custom_pty:
                if p=="b":
                    # Galactic latitude => a uniform prior is a cosine
                    dsm_kde=0.5*dsm_kde+0.5*np.cos(xfinal*np.pi/180)/sum(np.cos(xfinal*np.pi/180))*sum(dsm_kde)
                    #half the original distribution + half a uniform prior
                else:
                    dsm_kde=np.ones(len(dsm_kde))/len(dsm_kde)

            if plotdistrib:
                plt.step(xfinal,d_interp/sum(d_interp),color='C%d'%ic,ls=':',label=' C%d'%Classes[ic],alpha=0.5)
                plt.plot(xfinal,dsm_kde/sum(dsm_kde),color='C%d'%ic,alpha=1)
                if len(np.unique(Data[ind]))>=10:
                    ymax = max(ymax,np.nanmax(dsm_kde/sum(dsm_kde)))

            d=dsm_kde           # to apply smoothing to the histograms
            dl.append(d)
            
        if plotdistrib:
            plt.legend()
            if scale is not None:
                plt.xlabel(p+" %s"%((scale[ip]==2)*"(log)"+(scale[ip]==0)*"(sigmoid)"))
            else:
                plt.xlabel(p)
            plt.ylabel('Probability density')
            plt.ylim((-0.01,ymax*1.5))
            plt.savefig('%s%s.png'%(dirout,p))
    

        D=np.array([Bfinal[:-1],Bfinal[1:]]+dl).T
        if verbose:
            h,B,i=plt.hist(data,B,width=dx)
            print("property, range, len(data):\n",p,B[0],B[-1],sum(h))
            plt.show()


        if dirout:
            # LATER IMP: also save a recuperation file from which all distrib can be generated.
            np.savetxt('%s%s.dat'%(dirout,p),D,header='LOW   HIGH '+' '.join('Cl%d_COUNT'%C for C in Classes),fmt='%.3f')
        
                    
                
