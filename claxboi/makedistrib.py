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
from scipy.interpolate import splrep, splev,interp1d
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

n=100 #number of bins
equipart=False
#False or list with length=len(Classes)
verbose=0
fraction=1
if len(sys.argv)>1 and __name__=="main":
    fraction=float(sys.argv[1])
plotdistrib=0
dirout='classif/distributions/'

#catalog='catalogs/relGoldenSample_thin.csv'
#SourcesCsv=np.genfromtxt(catalog,delimiter=',',names=True)
#properties=['b','HR2','logFxFIR','logFxFopt','logFIR','FittedPowGamma']


def make(SourcesCsv, properties, Classes=[0,1,2], equipart=False, fraction=1, plotdistrib=False,N=100,verbose=False,dirout='classif/distribtest/', custom_pty=[],dumb=False):

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

    iClasses=[np.where(classes==C) for C in Classes]

    Nbins=np.arange(50,100,2)
    
    for p in properties:
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
        Bfinal=np.linspace(x0-dx,x1+dx,int(N*(x1-x0)/(x1-x0kde)))
        #print(x0,x1,x0kde,p)
            
        xfinal=(Bfinal[:-1]+Bfinal[1:])/2
        
        dl=[]
        
        D=(x1-x0)/Nbins  # array of test binwidth
        
        
        for ic in range(len(Classes)):
            ind=igoodClasses[ic] # without NaNs
            Ind=iClasses[ic]     # with NaNs
            B=np.linspace(x0,x1,N+5)
            d=np.histogram(Data[ind],B)[0]
            x=(B[1:]+B[:-1])/2
            d_interp=np.interp(xfinal,x,d) 

            #different methods to smooth the histogram into a PDF
            #method 1 : KDE
            kde=KernelDensity(bandwidth=(x1-x0kde)/(5*len(Data)**(1/5)), kernel='exponential') #rule of thumb
            kde.fit(Data[ind][:,None])
            logprob=kde.score_samples(xfinal[:,None])
            dsm_kde=np.exp(logprob)*sum(ind)
                
            # #method 2 : optimal binwidth + phase average
            # CostFunction=np.zeros(len(Nbins))
            # for i in range(len(Nbins)):
            #     x_range=np.histogram(Data[ind],bins=Nbins[i])[1]
            #     subdx=(x_range[1]-x_range[0])/5
            #     for j in range(5): #5 bin phases to smooth cost function
            #         d=np.histogram(Data[ind],x_range+j*subdx)[0]
            #         CostFunction[i]+=(2*np.mean(d)-np.var(d))/(D[i]**2)

            # n = Nbins[np.argmin(CostFunction)]
            # print("optimal bin number=%d for %d data points"%(n,sum(ind)))
            
            # B_opt=np.linspace(x0,x1,n+1)
            # x_opt=(B_opt[:-1]+B_opt[1:])/2
            # d_opt=np.zeros(n)
            # for i in np.linspace(0.5*(B_opt[0]-B_opt[1]),0.5*(B_opt[1]-B_opt[0]),10):
            #     d_opt+=np.histogram(data[Ind],B_opt+i)[0]

            # d_opt=d_opt/10
            # # d_old=np.histogram(data[Ind],B_opt)[0]

            # # n2=50
            # # B2=np.linspace(x0,x1,n2)
            # # x2=(B2[:-1]+B2[1:])/2
            # # d2=np.zeros(n2-1)
            # # dx_d2=(B2[1]-B2[0])/(int((B2[1]-B2[0])/dx)+1)
            # # nphase2=0
            # # for i in np.arange(0.5*B2[0]-0.5*B2[1],0.5*B2[1]-0.5*B2[0],dx_d2):
            # #     d2+=np.histogram(data[Ind],B2+i)[0]
            # #     nphase2+=1
                
            # # d2=d2/nphase2

            # #method 3 : smoothing of this average histogram
            
            # for s in range(1,9):
            #     try:
            #         bspl=splrep(x_opt,d_opt,s=s*sum(d_opt),k=3)
            #     except:
            #         print(len(x_opt),len(d_opt))
            #     #s=smoothing "degree" ; chisq(d,dsmooth)=sum(d)*s
            #     # bspl2=splrep(x2,d2,s=s*sum(d2),k=3)
            #     # bst=np.argmin([np.var(splev(x,bspl)),np.var(splev(x2,bspl2))])
            #     # n,x,d,bspl=[n,n2][bst],[x,x2][bst],[d,d2][bst],[bspl,bspl2][bst]
            #     dsm=splev(xfinal,bspl)
            #     if len(get_peak(dsm[dsm>max(dsm)/4]))<=4:
            #         #smooth contains few extrema => no need to smooth further
            #         break

            # dsm[dsm<0]=0          # counts can't be negative

            # d_interp_opt=np.interp(xfinal,x_opt,d_opt) 


            # #remove artefacts
            # l=[]
            # for k in range(len(dsm)):
            #     if dsm[k]>2*d_interp_opt[k]:
            #         # smooth>2*data: if not an outlier in the data, it is often an artefact in the fit, due to polynomial fitting (Runge's phenomenon)
            #         l.append(k)
            #     elif len(l)>2:# and np.mean(dsm[np.array(l)])<max(dsm)/10:
            #         # remove them if several in a row (discard outliers)
            #         dsm[np.array(l)]=d_interp_opt[np.array(l)]
            #         l=[]
            #     else:
            #         l=[]


            if len(np.unique(Data[ind]))<10:
                #property can be considered discrete
                #=> do not smooth its distributions
                dsm_kde=np.histogram(Data[ind],Bfinal)[0]

            if [p,ic] in custom_pty:
                if p=="b":
                    dsm_kde=0.5*dsm_kde+0.5*np.cos(xfinal*np.pi/180)/sum(np.cos(xfinal*np.pi/180))*sum(dsm_kde)
                    #half the original distribution + half a uniform prior
                else:
                    dsm_kde[:]=sum(d)*1/N
                    

            # print(sum(dsm)/sum(d_interp_opt))
            if plotdistrib:
                plt.figure(p)
                plt.plot(xfinal,d_interp/sum(d_interp),color='C%d'%ic,ls=':',label=' C%d'%Classes[ic],alpha=0.2)
                # plt.plot(xfinal,d_interp_opt/sum(d_interp_opt),color='C%d'%ic,ls='-.',label=' C%d smooth %d nbins %d'%(Classes[ic],s,n),alpha=0.2)
                # plt.plot(xfinal,dsm/sum(dsm),color='C%d'%ic,alpha=0.2,ls='--')
                plt.plot(xfinal,dsm_kde/sum(dsm_kde),color='C%d'%ic,alpha=1)

            d=dsm_kde           # to apply smoothing to the histograms
            
            dl.append(d)

        if plotdistrib:
            plt.legend()
            plt.show()
        D=np.array([Bfinal[:-1],Bfinal[1:]]+dl).T
        if verbose:
            h,B,i=plt.hist(data,B,width=dx)
            print(p,B[0],B[-1],sum(h))
            plt.show()

        if dirout:
            np.savetxt('%s%s.dat'%(dirout,p),D,header='LOW   HIGH '+' '.join('Cl%d_COUNT'%C for C in Classes),fmt='%.3f')


if __name__=="__main__":
    make(SourcesCsv,properties,equipart,plotdistrib,fraction)
