import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
from os.path import isfile
from matplotlib import rcParams
from astropy.table import Table

rcParams.update({'font.size':11})

Xmmclassified="../catalogs/4XMMDR10_classified.fits"
Xmmcla=Table.read(Xmmclassified)
Xmmcla=Xmmcla[np.asarray(Xmmcla['class']==99) & np.asarray(Xmmcla['SC_EXTENT']==0) & np.asarray(Xmmcla['quality']>=2)]
Xmmcla['WAPO_GAMMA']=np.log10(Xmmcla['WAPO_GAMMA'])
Swiclassified="../2SXPS_classified_new.fits"
Swicla=Table.read(Swiclassified)
Swicla=Swicla[np.asarray(Swicla['class']==99) & np.asarray(Swicla['quality']>=2)]
Swicla['FittedPowGamma']=(Swicla['FittedPowGamma'])/(1+abs(Swicla['FittedPowGamma']))

datadir='classif/distrib_KDE/'
showhist=0

coSwi=['b','HR1','HR2','FittedPowGamma','logFmaxFmin','logFmaxFminCons','logFxFr','logFxFw1']
#coSwi=coSwi[3:4]
coXmm=['b','SC_HR2','SC_HR3','WAPO_GAMMA','logFratio','logFratio_cons','logFxFr','logFxFw1']
#coXmm=coXmm[3:4]
coSwi=[datadir[:-1]+'_Swift/'+x+'.dat' for x in coSwi]
coXmm=[datadir+x+'.dat' for x in coXmm]

files=glob.glob(datadir+'*dat')
files=list(np.array([coSwi,coXmm]).T.reshape(-1))

print([isfile(f) for f in files])



if len(sys.argv)>1 and '%s%s.dat'%(datadir,sys.argv[1]) in files:
    files=['%s%s.dat'%(datadir,sys.argv[1])]
print(files)        
    
Class=['AGN','Star','XRB','CV']
col=['C0','C1','C2','C3']
    
for f in files:
    plt.figure(figsize=(7.5,4.5))
    d=np.loadtxt(f).reshape((-1,6))
    for i in [2,3,4,5]:#range(2,len(d[0])):
        plt.plot((d[:,0]+d[:,1])/2,d[:,i]/np.sum(d[:,i])/(d[1,0]-d[0,0]),label=Class[i-2],
         color=col[i-2],alpha=0.8,ls="-")
         
    if showhist and not("logFmaxFmed" in f):
        h=np.loadtxt('histograms/'+f.split('/')[-1])
        for i in [2,3,4,5]:#range(2,len(h[0])):
            plt.bar(h[:,1],h[:,i]/np.sum(h[:,i])/(h[1,0]-h[0,0]),color=col[i-2],alpha=0.1, width=(h[-1,0]-h[0,0])/(len(h)-1))
            plt.step(h[:,1]+(h[:,1]-h[:,0])/2,h[:,i]/np.sum(h[:,i])/(h[1,0]-h[0,0]),color=col[i-2],ls=':',alpha=0.4,label=Class[i-2])
        
    
    plt.legend()
    xlab=f.split('/')[-1].split('.')[0]
    prop=xlab
    if xlab=="WAPO_GAMMA":
        xlab=r"$log_{10}(FitPowGamma)$"
    elif xlab[:5]=="SC_HR":
        xlab="HR"+xlab[5:]
    elif xlab=="logFratio":
        xlab="max(Flux)/min(Flux)"
    elif xlab=="logFratio_cons":
        xlab="max(Flux-err)/min(Flux+err)"


    plt.xlabel(xlab)
    plt.subplots_adjust(bottom=0.18,top=0.99,left=0.1,right=0.99)
    if "Swift" in f:
        plt.savefig("../plots/2SXPS_%s_in.png"%prop)
    else:
        plt.savefig("../plots/4XMM_%s_in.png"%prop)
    plt.show()
    plt.figure(figsize=(7.5,4.5))
    if "Swift" in f:
        print(d[:,0])
        for i in range(4):
            t=np.histogram(Swicla[Swicla['prediction']==i][prop],bins=np.linspace(d[0,0],d[-1,1],min(60,int(sum(Swicla['prediction']==i)/6))),density=1)#,fc='none',histtype='step')
            plt.plot(0.5*(t[1][1:]+t[1][:-1]),t[0],ls='--',label=Class[i])
            
    else:
        for i in range(4):
            t=np.histogram(Xmmcla[Xmmcla['prediction']==i][prop],bins=np.linspace(d[0,0],d[-1,1],min(60,int(sum(Xmmcla['prediction']==i)/6))),density=1)#,fc='none',histtype='step')
            plt.plot(0.5*(t[1][1:]+t[1][:-1]),t[0],ls='--',label=Class[i])
    plt.legend()
    plt.xlabel(xlab)
    plt.subplots_adjust(bottom=0.18,top=0.99,left=0.1,right=0.99)
    if "Swift" in f:
        plt.savefig('../plots/2SXPS_%s_out.png'%prop)
    else:
        plt.savefig('../plots/4XMM_%s_out.png'%prop)
    plt.show()
