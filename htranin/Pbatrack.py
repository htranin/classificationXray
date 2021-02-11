### ----------------------------------------------
# Usage: python Pbatrack.py SRCID or JXXX..., python Pbatrack.py listofsources.txt
# plots the probabilities (likelihood) of each class given by each property of a given source (or list of sources)
# they are computed in the classification process (program classify.py)
# for each class, the product of all likelihoods is used to determine the class
### ----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size':11})
import sys
import glob

inst="XMM"
Classes=["AGN","Star","XRB","CV"]
itoplot=0#[1,4,6,11,13,15,17,19,21,23,24,25,26,28,30,31]
#[1,4,6,12,14,17,19,22,24,26,28,31,33,34]
#[1,4,6,11,13,17,19,21,23,24,25,27,29,30]

log=1
shuffle=0
if inst=="XMM":
    reference='../catalogs/4XMMDR10_toclassify.csv' # may be the same as $classification
    classification='../htranin/classification_4xmmdr10.csv'
    col_iauname='IAUNAME'
    col_srcid='SRCID'    #used for reference to match $reference and $classification
else:
    reference='/home/htranin/2SXPS_toclassify_041120.csv' # may be the same as $classification
    classification='/home/htranin/classification_2SXPS_new3.csv'
    col_iauname='IAUName'
    col_srcid='2SXPS_ID'    #used for reference to match $reference and $classification

refcat=np.genfromtxt(reference,delimiter=',',usecols=[0,1],dtype=None,encoding='utf-8',names=True)   # because usecols=[srcid, optionally iauname]


if len(sys.argv)>1:
    if sys.argv[1][-3:]=='txt':
        srcid=np.genfromtxt(sys.argv[1],delimiter=',',dtype=None,encoding='utf-8',usecols=[0])
    else:
        if sys.argv[1][:1]=='J':
            ind_iaunames=[i for i in range(len(refcat)) if sys.argv[1] in refcat[col_iauname][i]]
            srcid=refcat[col_srcid][ind_iaunames]
        else:
            try:
                srcid=[int(sys.argv[1])]
            except:
                print('Error: invalid argument')
                print('Please enter a .txt file, the SRCID of the target, or their position in the format "JXXXX..." (to be matched with IAUName)')
                exit()
                    
else:
    print('Error: invalid argument')
    print('Please enter a .txt file with SRCIDs, or the SRCID of the target, or their position in the format "JXXXX..." (to be matched with IAUName)')
    exit()

print('%d probability track%s to plot'%(len(srcid),'s'*(len(srcid)>1)))
if len(srcid)==0:
    exit()
if sys.argv[1][-3:]=="txt" and shuffle:
    np.random.shuffle(srcid)
    
classifsrc=list(np.genfromtxt(classification,usecols=[0],delimiter=',',dtype=None,encoding='utf-8'))
isrcid=np.array([classifsrc.index(s) for s in srcid if s in classifsrc])
print(np.array(classifsrc)[isrcid])
try:
    firstrows=np.genfromtxt(reference,delimiter=',',names=True,max_rows=2)
    properties=np.array(firstrows.dtype.names)
    inputfile=np.genfromtxt(reference.replace('csv','in'),dtype=None,encoding='utf-8',names=True)
    proptoplot=inputfile['property'][inputfile['to_use']==1]
    iprop=[i for i in range(len(properties)) if properties[i] in proptoplot]
    properties=properties[iprop]
    plotprop=0  # whether to plot the source properties
except:
    plotprop=0
firstrows=np.genfromtxt(classification,delimiter=',',names=True,max_rows=2)
colnames=np.array(firstrows.dtype.names)
if 'PbaTestSample' in colnames:
    skip1=1
else:
    skip1=0
    
ipbas=np.array([i for i in range(len(colnames)) if colnames[i][:3]=='Pba'])
if skip1:
    nClasses=(ipbas[2:]-ipbas[1:-1]!=1).argmax()+1
else:
    nClasses=(ipbas[1:]-ipbas[:-1]!=1).argmax()+1
    
print(nClasses)

for isrc in isrcid:
    if refcat[col_srcid][isrc]==classifsrc[isrc] and plotprop:
        prop=np.genfromtxt(reference,delimiter=',',skip_header=isrc+1,max_rows=1,usecols=iprop)
        plt.figure('Properties %s'%str(classifsrc[isrc]))
        c=np.array(prop[~np.isnan(prop)])
        plt.scatter(range(len(c)),c)        
        plt.xticks(range(len(c)),labels=properties[~np.isnan(prop)],rotation='vertical')
        plt.ylabel('value')
        plt.title('Properties of %s = %s'%(col_srcid,classifsrc[isrc]))        
        plt.grid(axis='x')
        plt.subplots_adjust(bottom=0.4)

    classifcat=np.genfromtxt(classification,delimiter=',',skip_header=isrc+1,max_rows=1)  #isrc+(size of the header)
    inan=np.where(~np.isnan(classifcat))[0]
    classifcat=classifcat[inan]
    cnames=colnames[inan]
    ipred=np.where(cnames=="prediction")[0][0]
    iclass=np.where(cnames=="class")[0][0]
    ipbas=np.array([i for i in range(len(cnames)) if cnames[i][:3]=='Pba'])
    iweights=np.array([i for i in range(len(cnames)) if cnames[i][:6]=='Weight'])

    
    plt.figure('Probability track %s'%str(classifsrc[isrc]),figsize=(10,7))
    if log:
        plt.gca().set_yscale('log')

    if not(itoplot):
        itoplot=range(len([classifcat[min(len(classifcat)-1,p)] for p in [ipbas[0]]+list(ipbas[1::nClasses])]))

    if skip1:
        for j in range(nClasses):
            c=np.array([classifcat[min(len(classifcat)-1,p)] for p in [ipbas[0]]+list(ipbas[j+1::nClasses])])  
            plt.scatter(range(len(c[itoplot])),np.maximum(c[itoplot],1e-7), label=Classes[j])
    else:
        for j in range(nClasses):
            c=np.array([classifcat[p] for p in ipbas[j::nClasses]])  
            plt.scatter(range(len(c)),np.maximum(c,1e-7))
    xlab=[''.join(p.split('_')[int('_' in p):]) for p in cnames[ipbas][::nClasses]]
    xlab[1]="Final probability"
    
    plt.xticks(range(len(c[itoplot])),labels=np.array(xlab)[itoplot],rotation='vertical')
        
    plt.ylabel('probability')

    if classifcat[iclass]!=99:
        plt.title('Probability track for %s = %s (%s classified as %s)'%(col_srcid,classifsrc[isrc],Classes[int(classifcat[iclass])],Classes[int(classifcat[ipred])]))
    else:        plt.title('Probability track for %s = %s (classified as %s)'%(col_srcid,classifsrc[isrc],Classes[int(classifcat[ipred])]))

    plt.grid(axis='x')
    plt.subplots_adjust(bottom=0.4)
    xlab=[''.join(p.split('_')[1:]) for p in cnames[iweights]]

    
    if 0:#len(iweights)>0:
        plt.figure('Weights %s'%str(classifsrc[isrc]))
        if log:
            plt.gca().set_yscale('log')

        c=np.array([classifcat[p] for p in iweights])  
        plt.scatter(range(len(c)),np.maximum(c,1e-7))
        plt.xticks(range(len(c)),labels=xlab,rotation='vertical')
        plt.ylabel('weight')

        if classifcat[iclass]!=99:
            plt.title('Weight track for %s = %s (%s predicted %s)'%(col_srcid,classifsrc[isrc],'C_%d'%int(classifcat[iclass]),'C_%d'%int(classifcat[ipred])))
        else:
            plt.title('Weight track for %s = %s (predicted %s)'%(col_srcid,classifsrc[isrc],'C_%d'%int(classifcat[ipred])))
        
        plt.grid(axis='x')
        plt.subplots_adjust(bottom=0.4)

    plt.legend()    
    plt.show()
