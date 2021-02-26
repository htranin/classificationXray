### ----------------------------------------------
# Usage: python3 classify.py
# Please edit the file configfile.ini before running the command above
# classifies the sources of the catalog stored in the var SourcesCsv
# probabilistic classification, the predicted class is the one giving the maximum product likelihood*prior
# the likelihood of a class C is the probability that the source belongs to C given its properties, using the distributions of each property from the objects identified as C in the reference sample
# displays the results of the classification on the whole reference sample
# if save == 1 the result is stored in a table named after the var fileout
### ----------------------------------------------

totest = []

import numpy as np
import scipy.stats as st
import sys
import select
import os
import makedistrib
from scipy.optimize import basinhopping, minimize, differential_evolution
import yaml


configfile = "configfile.ini"
with open(configfile) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

dirref = config['dirref']
# directory in which are (or will be) stored the probability densities (.dat files)

fileout = config['fileout']
save = config['save']

#fileout= the .csv file to make if save == 1
if save:
    print('output catalog: %s\n'%fileout)

categories = config['categories']
ncat = len(categories)
global_coeffs = config['global_coeffs']
# 1 weighting coefficient for missing values + 1 per category

# Prior proportions of the different classes (here AGN, Star, XRB, CV)
trueprop = config['trueprop'] #from CLAXSON users
# trueprop = [0.66, 0.23, 0.02, 0.02, 0.01, 0.06] #classification with 6 classes (AGN, Star, V/Y*, CV, gXB, eXB)
# trueprop = [0.93, 0.07] #binary classification (non-XB, XB)

compute_distrib = config['compute_distrib']
# compute the probability densities of the input catalog thanks to its known objects. See line "makedistrib.make(...)"
plotdistrib = 0

custom_pty = config['custom_pty']
# [Property, Class] for which the probability density is customed, to limit bias
# ['b', XRB] is set to 0.5*trainingsample+0.5*uniform, to work around the spikes at the galactic latitude of some nearby galaxies, which are most studied in the literature and hence more present in the reference sample

optimize_coeffs = config['optimize_coeffs']
# optimize the weighting coeffs of each category of properties to maximize of the f1-score of class C
C = config['C']
# Class on which the classification will be optimized

equipart = optimize_coeffs
# adapt the reference sample to the prior proportions? Useful to get proper false positive rates. [filename]_ifnullpba.csv must exist beforehand (always run once with equipart = 0)

misval_strategy = 'splitpba'
# strategy to handle missing values; one of 'splitpba', 'dumbval', 'ignore'
# splitpba: L(class|pty)=L(class|whether pty exist)*(L(class|pty value) or 1 if pty missing). Works well.
# dumbval: replace missing values by a dumb value e.g. -99 (hence present in the distributions). Works as well.
# ignore: L(class|missing value)=cst for every class. Less performant.


if len(sys.argv) > 1 and sys.argv[1][-3:] == 'csv':
    filename = sys.argv[1]
else:
    filename = config['filename']

print('input catalog:', filename)
inputfilename = filename.replace('csv', 'in')
# file detailing how to handle each column

# Load the catalog
SourcesCsv = np.genfromtxt(filename, delimiter=',', names=True)
icol_name = 0
Names = np.genfromtxt(filename, usecols=[icol_name], delimiter=',', names=True, dtype=None, encoding='utf-8')[SourcesCsv.dtype.names[icol_name]]

# Load the file detailing how to handle each column
if os.path.isfile(inputfilename):
    inputfile = np.genfromtxt(inputfilename, names=True, dtype=None, encoding='utf-8')
else:
    print('catalog columns: %s'%', '.join(SourcesCsv.dtype.names))
    dtypes = [('property', 'U32'), ('to_use', 'i4'), ('weight', 'U6'), ('category', 'i4'), ('pba_ifnull', 'i4'), ('scale', 'i4')]

    # description:
    # property: the name of the colums of the input catalog
    # to_use: whether or not the algorithm should use them in the classification (e.g. use spectral information, do not use the source name)
    # weight: the weight of the property, representing its quality. It will be used as an exponent to the likelihood P(value given class). "auto" means 1/e_property will be used if e_property is a column of the catalog, 1 otherwise.
    # category: the category of the property, e.g. whether it concerns location, X-ray spectrum, X-ray variability, multiwavelength ratio... The weights of each category are treated independently.
    # pba_ifnull: whether the algorithm should compute and use probabilities that the property exist given each class: hence missing values are taken into account.
    # scale: defines the x-axis scale to use when computing the distributions of the reference sample. For instance the histogram of a flux is better represented in logscale.

    inputfile = np.empty(len(SourcesCsv.dtype.names), dtype=np.dtype(dtypes))
    os.system('clear')
    print('\n\t\tPLEASE FILL IN THE INFORMATION BELOW (you have to do it once)')
    print('\t\t\tThey will be stored in %s\n'%inputfilename)
    print('Columns of the input catalog:',' '.join(SourcesCsv.dtype.names))
    print('\nto_use:(no=0|yes=1)\nweight:(auto=""|fixed=[float])\tcategory:('+'|'.join('%s=%d'%(categories[i], i+1) for i in range(ncat))+')\npba_ifnull:(no=0|yes=1)\tscale:(lin=1|log=2|{x/(1+|x|)}=0)')
    for icol in range(len(SourcesCsv.dtype.names)):
        col=SourcesCsv.dtype.names[icol]
        print('\t\t=== %s ===\t\t(%d/%d)'%(col, icol, len(SourcesCsv.dtype.names)))
        u = 0
        while not(u in ['0', '1', '']):
            u = input('to_use [0]? ')
        if u and int(u):
            u = 1
            not_filled = 1
            while not_filled:
                w, c, p, s = input('%s weight, category, pba_ifnull, scale? '%col).split(',')
                try:
                    w, c, p, s = w.strip(), int(c), int(p), int(s)
                    if w == '':
                        w = 'auto'
                    else:
                        w = str(float(w))
                    not_filled=0
                except:
                    pass            
        else:
            u = 0
            w, c, p, s = '0', 0, 0, 1
            

        inputfile[icol] = (col, u, w, c, p, s)

    np.savetxt(inputfilename, inputfile, delimiter='\t', 
               header='\t'+'\t'.join([dt[0] for dt in dtypes]), 
               fmt=['%32s', '%1d', '%s', '%2d', '%1d', '%1d'])

        
# Treatment of properties
inputfile = inputfile[inputfile["to_use"] == 1]
properties = inputfile['property']
nprop = len(properties)
for p in properties[inputfile['scale'] == 0]:
    SourcesCsv[p] = SourcesCsv[p]/(1+abs(SourcesCsv[p]))
for p in properties[inputfile['scale'] == 2]:
    SourcesCsv[p] = np.log10(SourcesCsv[p])


print('This program will classify X-ray sources from %s using %d of their properties:\n'%(filename, nprop)+',\n'.join([', '.join(properties[i:i+7]) for i in range(0, nprop, 7)])+'\n')

# Define the classes of the catalog. Must agree with trueprop
classes = SourcesCsv['class']
Classes = np.unique(classes[~np.isnan(classes)]).astype(int)
ncla = len(Classes)
#e.g. 0: AGN, 1:STAR, 2:CO or 2: XRB, 3: CV

if misval_strategy == 'dumbval':
    for p in properties:
        SourcesCsv[p][np.isnan(SourcesCsv[p])] = -20
        # dumb value that will be present when computing distributions
        if 'e_%s'%p in SourcesCsv.dtype.names:
            SourcesCsv['e_%s'%p][np.isnan(SourcesCsv['e_%s'%p])] = 1


if compute_distrib:
    # Estimate the probability densities and save them in dirref
    makedistrib.make(SourcesCsv, properties=properties, Classes=Classes, equipart=0, fraction=1, plotdistrib=plotdistrib, custom_pty=custom_pty, dirout=dirref, dumb=(misval_strategy == 'dumbval'))

if misval_strategy == 'ignore':
    for p in properties:
        SourcesCsv[p][np.isnan(SourcesCsv[p])] = -20
        # dumb value that won't be present when computing distributions
        if 'e_%s'%p in SourcesCsv.dtype.names:
            SourcesCsv['e_%s'%p][np.isnan(SourcesCsv['e_%s'%p])] = 1

# in-category coefficients to give more weight to safer properties (the ones having lower error e_pty)
coeffs = np.empty((nprop, len(SourcesCsv)))
for ip in range(nprop):
    if inputfile['weight'][ip] == 'auto':
        if 'e_%s'%properties[ip] in SourcesCsv.dtype.names:
            coeffs[ip] = 1/SourcesCsv['e_%s'%properties[ip]]
            #print(properties[ip], np.mean(coeffs[~np.isnan(coeffs)]))
        else:
            coeffs[ip] = np.ones(len(SourcesCsv))
    else:
        coeffs[ip] = float(inputfile['weight'][ip])*np.ones(len(SourcesCsv))


       


if equipart:
    # make the file otherhalf.dat, storing indexes of a subsample proportioned as trueprop
    makedistrib.make(SourcesCsv, properties=[], Classes=Classes, equipart=trueprop, fraction=1, dirout='')
    # restrict the sample to this subsample
    selection = np.loadtxt('otherhalf.dat').astype(int)
    SourcesCsv = SourcesCsv[selection]
    Names = Names[selection]
    coeffs = coeffs[:,selection]
    classes = SourcesCsv['class']
    print('counts of each class in the properly proportioned sample:', [sum(classes == C) for C in Classes])


# Functions for treatment of the probability densities
def normalize(histo):
    histo[:,2] = histo[:,2]/sum(histo[:,2])
    return histo

def fillzeros(histo):
    ind = np.where(histo[:,2] == 0)[0]
    histo[:,2] = (histo[:,2]+0.01/len(histo))/(1+0.02/len(histo))
    return histo

def rebin(histo):
    # new x-axis = histogram bar centers
    histo[:,1] = (histo[:,0]+histo[:,1])/2
    return histo[:,1:]

# Load and normalize the probability densities for later use

Distrib = [[] for ic in range(ncla)]
for ip in range(nprop):
    d = np.loadtxt('%s%s.dat'%(dirref, properties[ip]))
    # import the estimated distributions (unit:counts) of each class
    for ic in range(ncla):
        Distrib[ic].append(rebin(normalize(fillzeros(normalize(d[:,(0, 1, ic+2)])))))


prediction = np.empty(len(classes))
predictionDaria = np.empty(len(classes))

# Spot all missing values
SourcesNan = np.isnan(np.vstack([SourcesCsv[p] for p in properties]).T)
print("total number of missing values:", sum(sum(SourcesNan)))

# 3D (Class, Pty, Source) Matrix of likelihoods L(class|pty) for each source
ainterp = np.array([np.array([np.interp(SourcesCsv[properties[ip]], Distrib[ic][ip][:,0], Distrib[ic][ip][:,1], left=1, right=1) for ip in range(nprop)]) for ic in range(ncla)])

# List of vectors flagging source indexes, one per class
# 3 sources (AGN, nan, Star) => [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]
refsample = [classes == Classes[ic] for ic in range(ncla)]
if not(equipart):
    ifnullpba = np.ones(np.shape(ainterp))
    # properties for which a pba_if_null is computed = all properties
    ifnullpba = np.array([abs(SourcesNan-(sum(1.-SourcesNan[refsample[ic]])/sum(refsample[ic])+0.001)/1.002).T for ic in range(ncla)])
    # 0.001 to prevent zero probabilities
    #later imp: ifnullpba[:, inputfile['pba_ifnull'] == 0]=1
    #later imp: ifnullpba[np.isnan(SourcesCsv['RA_ir']), properties.index('logFxFw1')]=1
    #           ifnullpba[np.isnan(SourcesCsv['RA_ir']), properties.index('logFxFw2')]=1
    #           ifnullpba[np.isnan(SourcesCsv['RA_opt']), properties.index('logFxFr')]=1
    #           ifnullpba[np.isnan(SourcesCsv['RA_opt']), properties.index('logFxFb')]=1
    #later imp: do not use nullpba if only one band is missing in optical and/or infrared
    #later imp: only use nullpba if all properties of a subcategory are missing (e.g. logFxFr, logFxFb)
    #later imp (?): do not use nullpba if property is not null?
    #later imp: do not use nullpba if a counterpart was found but no flux is available

    #print(inputfile['property'][inputfile['pba_ifnull'] == 0])
    np.savetxt(filename.replace('.csv', '_ifnullpba.csv'), np.concatenate(ifnullpba), delimiter=',')
else:
    ifnullpba = np.loadtxt(filename.replace('.csv', '_ifnullpba.csv'), delimiter=',').reshape((ncla, nprop, -1))[:,:,selection]



# List of vectors flagging property indexes, one per category
icat = [inputfile['category'] == cat+1 for cat in range(ncat)]


expo = [[np.mean(coeffs[~SourcesNan[isrc]*icat[inputfile['category'][ip]-1], isrc]) for ip in np.arange(nprop)[~SourcesNan[isrc]]] for isrc in range(len(SourcesCsv))]


# 3D (Source, Pty, Class) Matrix of weighted likelihoods
Pbgood = [[ainterp[ic, ~SourcesNan[isrc], isrc]**(coeffs[~SourcesNan[isrc], isrc]/expo[isrc])*ifnullpba[ic, ~SourcesNan[isrc], isrc] for ic in range(ncla)] for isrc in range(len(SourcesCsv))] #later imp: try without multiplying by ifnullpba[,~SourcesNan[isrc],]
Pbnull = [[ifnullpba[ic, SourcesNan[isrc], isrc] for ic in range(ncla)] for isrc in range(len(SourcesCsv))]


# Computes the posterior probability of a source,
# given the weights to apply per category
def proba_fast(isrc, global_weights):
    # alpha weigh missing values
    alpha, global_weights = global_weights[0], global_weights[1:]
    pbCl = np.ones(ncla)
    pbgood = Pbgood[isrc]
    pbnull = Pbnull[isrc]
    pbCats = []

    for cat in range(ncat): #location, spectrum, multiwv, variabty
        try:
            pbCat = np.array([(np.nanprod(pbgood[ic][icat[cat][~SourcesNan[isrc]]])*np.product(pbnull[ic][icat[cat][SourcesNan[isrc]]])**alpha)**(5*global_weights[cat]/(2*len(pbgood[0])+alpha*len(pbnull[0]))) for ic in range(ncla)])
            pbCats.append(pbCat/sum(pbCat))
            pbCl = pbCl*pbCat #+alpha*len(pbnull[0])
        except:
            pass

    pbCl = pbCl**(8/sum(global_weights))
    pbCl = pbCl*trueprop
            
    alt = ""
    for cat in range(ncat):
        pbCat2, gw2 = pbCats[:], list(global_weights[:])
        pbCat2.pop(cat)
        gw2.pop(cat)
        pbCl2 = np.product(np.array(pbCat2), axis=0)**(8/sum(gw2))
        pbCl2 = pbCl2*trueprop
        if np.argmax(pbCl2) != np.argmax(pbCl):
            alt += categories[cat][:3]+'%d '%np.argmax(pbCl2)

    #later imp: add alternative when given by uniform priors
    #pbCl = np.product(pbCat, axis=0)

    if isrc in totest:#classes[isrc] == 2 and isrc%200 == np.random.randint(200):# isrc%3000 == 13:
        # sanity checks
        ic = np.argmax(Classes == classes[isrc])
        print('\n Source,  category,  class:', isrc, cat, ic)
        print(alt)
        #vprint(len(pbgood[ic]), pbgood[ic])
        # print(len(pbnull[ic]), pbnull[ic])
        # print(len(exp), exp)
        string = '\t'
        for cat in range(ncat):
            for p in properties[icat[cat]]:
                string += ' %s-%s %.1f'%(p[:3], p[-3:], SourcesCsv[p][isrc])
            string += ' %s %s'%(pbgood[ic][icat[cat][~SourcesNan[isrc]]], pbnull[ic][icat[cat][SourcesNan[isrc]]])
            string += ' %s'%(np.product(pbgood[ic][icat[cat][~SourcesNan[isrc]]])/sum([np.product(pbgood[iC][icat[cat][~SourcesNan[isrc]]]) for iC in range(ncla)]))
            if np.isnan(np.product(pbgood[ic][icat[cat][~SourcesNan[isrc]]])/sum([np.product(pbgood[iC][icat[cat][~SourcesNan[isrc]]]) for iC in range(ncla)])):
                try:
                    string += ' %s %s %s'%(str(ainterp[ic,:,isrc][icat[cat][~SourcesNan[isrc]]]), str(coeffs[:,isrc][icat[cat][~SourcesNan[isrc]]]), str(np.array(expo[isrc])[icat[cat][~SourcesNan[isrc]]]))
                except:
                    print(ainterp[ic,:,isrc], type(ainterp[ic,:,isrc]), [icat[cat][~SourcesNan[isrc]]], type(icat[cat][~SourcesNan[isrc]]))
            string += ' %s\n'%(np.product(pbnull[ic][icat[cat][SourcesNan[isrc]]])/sum([np.product(pbnull[iC][icat[cat][SourcesNan[isrc]]]) for iC in range(ncla)]))
        pbCl2 = np.ones(ncla)
        for cat in range(ncat):
            pbCl2 = pbCl2*np.array([(np.nanprod(pbgood[ic][icat[cat][~SourcesNan[isrc]]])*np.nanprod(pbnull[ic][icat[cat][SourcesNan[isrc]]]**alpha))**(5*global_weights[cat]/(2*len(pbgood[0])+alpha*len(pbnull[0]))) for ic in range(ncla)])
        pbCl2 = pbCl2**(8/sum(global_weights))*trueprop
        pbCl2 = pbCl2/sum(pbCl2)
        print(string)
        print(pbCl/sum(pbCl), pbCl2)



    if not(save):
        return [pbCl]
    else:
        weights_cat = [np.nansum(coeffs[inputfile['category'] == cat+1, isrc]) for cat in range(ncat)]
        recordPb = np.concatenate(pbCats)  #np.concatenate(list(np.concatenate((ainterp[:,:,isrc]/sum(ainterp[:,:,isrc]), ifnullpba[:,:,isrc]/sum(ifnullpba[:,:,isrc]), [coeffs[:,isrc]])).T)+pbCats)

        # comment ", recordPb" to save memory
        return pbCl, weights_cat, recordPb, alt

    



# Build the empty result
# these will be the columns of the output catalog!
# Structure: source name, general classification indicators, and then one probability per class and per property. 30/10/20: added one probability per class per category.

dtypes = [('Name', 'U22'), ('class', 'i4'), ('prediction', 'i4'), ('alt', 'U24'), ('ClMargin', 'f8'), ('outlier', 'f8'), ('N_missing', 'i4')]+[('PbaC%d'%Classes[ic], 'f8') for ic in range(ncla)]+[('Weight_%s'%cat, 'f8') for cat in categories]

### comment this section if "recordPb" is commented below
#for ip in range(nprop):
#    dtypes += [('PbaC%d_%s'%(Classes[ic], properties[ip]), 'f8') for ic in range(ncla)]+[('PbaC%d_Existence%s'%(Classes[ic], properties[ip]), 'f8') for ic in range(ncla)]+[('Weight_%s'%properties[ip], 'f8')]
###
for cat in categories:
    dtypes += [('PbaC%d_%s'%(Classes[ic], cat), 'f8') for ic in range(ncla)]

classification = np.empty(len(SourcesCsv), dtype=np.dtype(dtypes))

if optimize_coeffs:
    fout = open(filename.replace('.csv', '_optimsteps.dat'), 'w')
else:
    fout = None

def f1score(global_weights, outfile=fout):
    # computes the f1-score (2*recall*precision/(recall+precision)) of the classification for class C. Recall=Retrieval fraction, Precision=True positive rate
    # f1score = 2*(#predicted_C_&_actually_C)/(#predicted_C + #actually_C)
    p1C1 = 0 #prediction == class == C
    p1C0 = 0 #predicted as C but not C
    p0C1 = 0 #C but not predicted as C

    #lpbCl=np.array([proba_fast(i, global_weights)[0].argmax() for i in range(len(SourcesCsv))])
    #prc = np.array([sum(lpbCl == c && classes == c)/(sum(lpbCl == c) for c in Classes])
    #rec = np.array([sum(lpbCl == c && classes == c)/sum(classes == c) for c in Classes])
    #f1 = np.mean(2/(1/prc+1/rec)) #f1 averaged over all classes
    #print(' '.join(['%.3f'%gw for gw in global_weights]+list(rec)+list(prc)+list(f1)), file=outfile)
    #return f1
    
    for i in range(len(SourcesCsv)):
        pbCl = proba_fast(i, global_weights)[0].argmax()
        p1C1 += (Classes[pbCl] == C and classes[i] == C)
        p0C1 += (Classes[pbCl] != C and classes[i] == C)
        for oc in Classes[Classes != C]:
            p1C0 += (Classes[pbCl] == C and classes[i] == oc)

    print(' '.join(['%.3f'%gw for gw in global_weights]), p0C1, p1C1, p1C0, 2*p1C1/(2*p1C1+p0C1+p1C0), file=outfile)
    return 1-2*p1C1/(2*p1C1+p0C1+p1C0)


# Define 1 weighting coefficient for missing values (usually 1) + 1 per category

# global_coeffs=[1, 0.7, 1.4, 5.5, 9.9] # CLAXSON proportions + latest variability + quality >= 2
# global_coeffs=[1, 1, 2, 8, 10] # CLAXSON proportions + latest variability + quality >= 2
# global_coeffs=[1, 0, 6.5, 7, 8.5] #2SXPS
# global_coeffs=[1, 0, 3, 6, 9] #4XMM
# global_coeffs=[1, 0.5, 4, 7, 9.5] #DR10
# global_coeffs=[1, 9.5, 4, 8, 4.5] #DR10 6 classes
# global_coeffs=[1, 9, 4, 8, 5] #DR10 6 classes #2
# global_coeffs=[1, 7.5, 2.5, 9, 7.5] #DR10 binary classif

if optimize_coeffs:
    fout.write('# pba_null  location  hardness  spectrum  multiwavelength  variability  FN TP FP f1\n')
    # Coeff optimizition, to maximize the f1-score of class C
    try:
        res = differential_evolution(f1score, [(len(global_coeffs)-len(categories))*[0,1]]+(len(categories))*[[0,10]], disp=1)
        print(res.x, res.fun, res.message, res.nit)
        global_coeffs = res.x
    except:
        fout.close()
        outfile2 = '4XMMDR10_toclassify_sample_optimsteps.dat'
        evol_coeffs = np.loadtxt(outfile2)
        global_coeffs = evol_coeffs[np.argmax(evol_coeffs[:,-1])][:-4]
        
    str_coeffs = str(list(np.round(global_coeffs, 2)))
    os.system('sed -i "s/global_coeffs:.*/global_coeffs: %s/g" %s'%(str_coeffs, configfile))
    
    fout.close()

# Compute the posterior probability of each class for every source
    
for i in range(len(SourcesCsv)):
    if save:
        # comment ", recordPb =" to save memory
        pbCl, weights_cat, recordPb, alt = proba_fast(i, global_coeffs)#, recordPb, alt
        outlier = -np.log10(max(pbCl))
        # high if the object lies in the distribution tails of all classes
        Nnan = sum(SourcesNan[i])
        # number of missing values for this source
        prediction[i] = Classes[np.argmax(pbCl)]
        # predicted class, the one with higher probability
        
        if not(np.isnan(classes[i])):
            # comment "+list(recordPb)" if ", recordPb" was previously commented
            classification[i] = tuple([Names[i], classes[i], prediction[i], alt, 2*max(pbCl)/sum(pbCl)-1, outlier, Nnan]+[pbCl[ic]/sum(pbCl) for ic in range(ncla)]+weights_cat+list(recordPb))
        else:
            classification[i] = tuple([Names[i], 99, prediction[i], alt, 2*max(pbCl)/sum(pbCl)-1, outlier, Nnan]+[pbCl[ic]/sum(pbCl) for ic in range(ncla)]+weights_cat+list(recordPb))
    else:
        pbCl = proba_fast(i, global_coeffs)[0]
        prediction[i] = Classes[np.argmax(pbCl)]
        #reliable XRB prediction if pbaAGN < threshold?
        #Good hint but does not surpass another choice of coeff
        #(example: RF=0.71, TPR=0.83 becomes RF=0.55, TPR=0.88, 
        #other choice of coeff able to give RF=0.59, TPR=0.88)
        # if prediction[i] == 2 and pbCl[0]/sum(pbCl) > 0.2:
        #     prediction[i] = 0
        #     print('not very reliable XRB => reclassified as AGN')
    


# Print general results

print('trueprop = %s\t global_coeffs = %s'%(str(trueprop), str(global_coeffs)))
print(', '.join(['NC%s=%d'%(Classes[ic], sum(classes == Classes[ic])) for ic in range(ncla)]))
print(', '.join(['NpC%s=%d'%(Classes[ic], sum(prediction == Classes[ic])) for ic in range(ncla)]))


results = np.zeros((ncla+1, ncla+1))

for i in range(ncla):
    for j in range(ncla):
        results[i,j] = sum((classes == Classes[j])*(prediction == Classes[i]))

results[:-1,-1] = [100*round(results[i,i]/sum(results[:,i]), 3) for i in range(ncla)]                        # Retrieval Fraction
results[-1,:-1] = [100*round(1-results[i,i]/sum(results[i,:-1]), 3) for i in range(ncla)]# False Positive rate

print('Truth --->\tC'+'\tC'.join(Classes.astype(str))+'\tretrieval fraction (%)')
for i in range(ncla+1):
    if i < ncla:
        print('P%s'%str(Classes[i]), '\t\t'+'\t'.join(results[i,:-1].astype(int).astype(str))+'\t%.1f'%results[i,-1])
    else:
        print('false pos. rate\t'+'\t'.join(['%.1f'%r for r in results[i,:-1]]))


        
if save:
    # Save these results
    with open(fileout.replace('csv', 'metrics'), 'w') as f:
        print('trueprop = %s\t global_coeffs = %s'%(str(trueprop), str(global_coeffs)), file=f)
        print(', '.join(['NC%s=%d'%(Classes[ic], sum(classes == Classes[ic])) for ic in range(ncla)]), file=f)
        print(', '.join(['NpC%s=%d'%(Classes[ic], sum(prediction == Classes[ic])) for ic in range(ncla)]), file=f)
        print('# Truth --->\tC'+'\tC'.join(Classes.astype(str))+'\tretrieval fraction (%)', file=f)
        for i in range(ncla+1):
            if i < ncla:
                print(Classes[i], '\t\t'+'\t'.join(results[i,:-1].astype(int).astype(str))+'\t%.1f'%results[i,-1], file=f)
            else:
                print('false pos. rate\t'+'\t'.join(['%.1f'%r for r in results[i,:-1]]), file=f)

    # comment (2*ncla+1) and (ncla+1) (replace by 1 and 0) if recordPb is commented above
    np.savetxt(fileout, classification, delimiter=',', 
               header=','.join([dt[0] for dt in dtypes]), 
               fmt=['%22s', '%2d', '%2d', '%24s', '%.3e', '%.3e', '%2d']+(ncla+ncat*(ncla+1)+0*nprop)*['%.3e']) #(ncla+1) (2*ncla+1)

        
