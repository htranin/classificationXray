#!/home/hugo/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:55:39 2021

@author: htranin
"""

import os
from astropy.table import Table, vstack, hstack
from sys import exit
import numpy as np
import sys
os.chdir('/home/htranin/')

#####
#CDS catalogues [optical, infrared] to be matched with the X-ray catalogue

cds_tables = [["Gaia EDR3","PanSTARRS DR1","DES DR1","USNO-B1.0"],
              ["2MASS","AllWISE","UnWISE"]]
# sky coverages in squared degrees
cds_cov = [[41253,31512,5255,41253],[41253,41253,41253]]
# number of sources. Used to compute average source densities
cds_ntot = [[1.81e9,1.92e9,3.99e8,1.05e9],[4.71e8,7.48e8,2.21e9]]
# User-friendly names
cds_names = [[cd.replace('_',' ').replace('-',' ').split()[0] for cd in cds_tables[0]],
             [cd.replace('_',' ').replace('-',' ').split()[0] for cd in cds_tables[1]]]
# Maximum matching radius
radius = 8
# False positive rates used to calibrate p_any. The last value is used as calibration.
nwaycutoffs=[0.01,0.03,0.05,0.1,0.15]
nwaycutoffs=str(nwaycutoffs)
# Column names to use in secondary catalog
optircolnames = ["ra","dec","phot_rp_mean_mag","phot_bp_mean_mag","phot_g_mean_mag","pm","rmag","bmag","gmag","imag","rkmag","rmag1","rmag2","bmag1","bmag2","kmag","w1mag","w2mag","fw1","fw2"]


# Following parameters can be used for reruns and debugging
skip_actual = False #skip to fake positions
skip_cds = False #skip cdsskymatch
skip_nway = False #skip nway runs
nonmatchonly = False #remove secure matches before matching again with another catalogue

# Input file
if len(sys.argv)>1:
    input_fname = sys.argv[1]
else:
    input_fname = "classificationXray-main/data/4XMM_DR12cat_slim_v1.0.fits"   # IMPORTANT: first restrict the catalogue to less than ~10 columns to avoid kill    


param = ' '.join(list(np.array([skip_actual,skip_cds,skip_nway,nonmatchonly,radius,input_fname]).astype(str)))



#Tune nway calibrate cutoff to return a text file
path = os.popen("which nway-calibrate-cutoff.py").read()[:-1]
if path=="":
    print("error in the installation of nway. Please check that nway is well installed")
    exit()
else:
    f = open(path,"r")
    lines = f.readlines()
    line = 'numpy.savetxt(args.realfile + "_p_any_cutoffquality.txt", numpy.array([%s,[cutoffs[numpy.argmin(abs(numpy.array(error_rate)-rate))] for rate in %s]]).T, fmt="%.3f", header="fp_rate p_any_cutoff")\n'%(nwaycutoffs,nwaycutoffs)
    if not(line in lines):
        f.close()
        f = open(path,"a")
        f.write("\n")
        f.write(line)
        print("added one line of code to", path)
    f.close()

# Use the default settings (err_c1, coverage) of X-ray catalogues (2SXPS, 4XMM or CSC2)
whole_X_catalog = 1

if not(whole_X_catalog):
    # Put the sky coverage of your catalog, in deg²
    coverage = 602000/502000*1239
    # Put the position error column of your catalog below, in arcsec
    error_column = "SC_POSERR"

# Save auxiliary table with less columns, to work faster
reduce_table = 1


def find_id_ra_dec(table,fname,return_flux=False):
    colnames = table.colnames
    id_colname = ''
    ra_colname = ''
    dec_colname = ''
    flux_colname = ''
    for col in colnames:
        if table[col].unit is None:  #first column (most often)
            id_colname = col
            break        
    if 'ra' in [col.lower() for col in colnames]:
        icol = [col.lower() for col in colnames].index('ra')
        ra_colname = colnames[icol]
    else:
        for col in colnames:
            if (col.lower()[:2]=="ra" or col.lower()[-2:]=="ra") and str(table[col].unit)=="deg":
                ra_colname = col
                break
    if 'dec' in [col.lower() for col in colnames]:
        icol = [col.lower() for col in colnames].index('dec')
        dec_colname = colnames[icol]
    else:
        for col in colnames:
            if (col.lower()[:2]=="de" or col.lower()[-3:]=="dec") and str(table[col].unit)=="deg":
                dec_colname = col
                break            
    if '' in [id_colname, ra_colname, dec_colname]:
        imiss = [id_colname, ra_colname, dec_colname].index('')
        print("Error: %s column is missing in %s"%(["ID","RA","DEC"][imiss],fname))
        exit()
    if not(return_flux):
        return id_colname, ra_colname, dec_colname
    else:
        flux_colnames=[]
        flux_colname =" "
        for col in colnames:
            if "flux" in col.lower() and not(col.lower()[-3:] in ['err','lim','pos','neg']):
                flux_colnames.append(col)
        print("Flux columns found:", flux_colnames)
        if len(flux_colnames)==1:
            flux_colname=flux_colnames[0]
        colmaxflux = flux_colnames[np.nanargmax(list(table[0][flux_colnames]))]
        while not(flux_colname in flux_colnames):
            flux_colname = input("Column to use for flux? [%s] "%colmaxflux)
            if flux_colname == '':
                flux_colname = colmaxflux
        print("Selected column", flux_colname)
        return id_colname, ra_colname, dec_colname, flux_colname
 
    

def mag_computations(cds_name,cdsout_fname):
    """Function to calibrate all optical and infrared magnitudes to the Gaia and WISE photometric bands.
    The following linear relations between magnitudes were calibrated empirically using safe matches (better than 0.5 arcsec).
    The resulting magnitudes are only a proxy of Gaia and WISE magnitudes and should not be used for optical studies.
    """
    cds = Table.read(cdsout_fname)
    if cds_name=="Gaia":
        cds['phot_rp_mean_mag'].name='Rmag'
        cds['phot_bp_mean_mag'].name='Bmag'
        cds['phot_g_mean_mag'].name='Gmag'
        cds['Rmag'][np.isnan(cds['Rmag'])]=cds['Gmag'][np.isnan(cds['Rmag'])]-0.5
        cds['Rmag'][np.isnan(cds['Rmag'])]=cds['Bmag'][np.isnan(cds['Rmag'])]-0.8
        cds['Bmag'][np.isnan(cds['Bmag'])]=cds['Gmag'][np.isnan(cds['Bmag'])]+0.3
        cds['Bmag'][np.isnan(cds['Bmag'])]=cds['Rmag'][np.isnan(cds['Bmag'])]+0.8
    elif cds_name=="PanSTARRS":
        cds['rmag'].name='Rmag'
        cds['gmag'].name='Bmag'
        cds['Rmag'][np.isnan(cds['Rmag'])]=cds['rKmag'][np.isnan(cds['Rmag'])]
        cds['Rmag'][np.isnan(cds['Rmag'])]=cds['imag'][np.isnan(cds['Rmag'])]+0.4
        cds['Rmag'][np.isnan(cds['Rmag'])]=cds['Bmag'][np.isnan(cds['Rmag'])]-0.3
        cds['Bmag'][np.isnan(cds['Bmag'])]=cds['Rmag'][np.isnan(cds['Bmag'])]+0.3
        cds['Rmag']-=0.75
        cds['Bmag']-=0.2
    elif cds_name=="DES":
        cds['rmag'].name='Rmag'
        cds['gmag'].name='Bmag'
        cds['Rmag']-=0.5
        cds['Bmag']-=0.1
    elif cds_name=="USNO":
        cds['Rmag2'].name='Rmag'
        cds['Bmag2'].name='Bmag'
        cds['Rmag'][np.isnan(cds['Rmag'])]=cds['Rmag1'][np.isnan(cds['Rmag'])]
        cds['Bmag'][np.isnan(cds['Bmag'])]=cds['Rmag'][np.isnan(cds['Bmag'])]
        cds['Rmag'][np.isnan(cds['Rmag'])]=cds['Bmag'][np.isnan(cds['Rmag'])]-1
        cds['Bmag'][np.isnan(cds['Bmag'])]=cds['Rmag'][np.isnan(cds['Bmag'])]+1
        cds['Rmag']-=0.4
        cds['Bmag']-=0.2
    elif cds_name=="2MASS":
        cds['Kmag'].name='W1mag'
        cds['W1mag']-=0.1
        cds.add_column(cds['W1mag'],name='W2mag')
    elif cds_name=="AllWISE":
        cds['W1mag'][np.isnan(cds['W1mag'])]=cds['W2mag'][np.isnan(cds['W1mag'])]
        cds['W2mag'][np.isnan(cds['W2mag'])]=cds['W1mag'][np.isnan(cds['W2mag'])]
    elif cds_name=="UnWISE":
        cds['FW1'].name='W1mag'
        cds['W1mag'][cds['W1mag']==0] = np.nan
        cds['W1mag']=22.5-2.5*np.log10(cds['W1mag'])
        cds['FW2'].name='W2mag'
        cds['W2mag'][cds['W2mag']==0] = np.nan
        cds['W2mag']=22.5-2.5*np.log10(cds['W2mag'])
        cds['W1mag'][np.isnan(cds['W1mag'])]=cds['W2mag'][np.isnan(cds['W1mag'])]
        cds['W2mag'][np.isnan(cds['W2mag'])]=cds['W1mag'][np.isnan(cds['W2mag'])]
    cds.write(cdsout_fname, overwrite=True)
    


## BEGIN ##
if __name__ == "__main__":
    ####
    #Load the X-ray catalogue
    
    
    xname = input_fname.split('/')[-1].replace('_',' ').replace('-',' ').split()[0]
    if whole_X_catalog:
        if xname=="2SXPS":
            coverage = 3790
            err_c1 = "Err63"
        elif xname[:4]=="4XMM":
            coverage = 1283       #DR10: 1192
            err_c1 = "SC_POSERR"
        elif xname[:4]=="CSC2":
            coverage = 550
            err_c1 = "err_ellipse_r0"
    else:
        coverage = coverage
        err_c1 = error_column
        
    input_table = Table.read(input_fname)
        
    id_c1, ra_c1, dec_c1 = find_id_ra_dec(input_table,input_fname)
    input_table[ra_c1].name = "RA"
    ra_c1 = "RA"
    input_table[dec_c1].name = "DEC"
    dec_c1 = "DEC"
            
    if not("id" in id_c1.lower()):
        input_table[id_c1].name=id_c1+"_ID"
        id_c1+="_ID"
        input_table.write(input_fname,overwrite=True)
    
    if reduce_table and len(input_table.colnames)>15:
        try:
            fx_c1 = find_id_ra_dec(input_table,input_fname,return_flux=1)[-1]
            if ra_c1+'_fake' in input_table.colnames and dec_c1+'_fake' in input_table.colnames:
                input_table = input_table[id_c1,ra_c1,dec_c1,ra_c1+'_fake',dec_c1+'_fake',err_c1,fx_c1]
            else:
                input_table = input_table[id_c1,ra_c1,dec_c1,err_c1,fx_c1]
        except:
            if ra_c1+'_fake' in input_table.colnames and dec_c1+'_fake' in input_table.colnames:
                input_table = input_table[id_c1,ra_c1,dec_c1,ra_c1+'_fake',dec_c1+'_fake',err_c1]
            else:
                input_table = input_table[id_c1,ra_c1,dec_c1,err_c1]
        
        input_fname = ".".join(input_fname.split(".")[:-1])+"_thin.fits"
        input_table.write(input_fname, overwrite=True)
        
        
    id_c3 = '%s_%s'%(xname,id_c1)
    ra_c4 = "AUX_%s"%ra_c1
    dec_c4 = "AUX_%s"%dec_c1
    input_table.write("%s_with_counterparts.fits"%xname,overwrite=True)
    
    #Check prerequisites on the fake X-ray positions required to properly run Nway
    fake=False
    # if os.path.isfile(input_fname.replace('.fits','-fake.fits')):
    #     input_fake = Table.read(input_fname.replace('.fits','-fake.fits'))
    #     id_cf, ra_cf, dec_cf = find_id_ra_dec(input_fake,input_fname.replace('.fits','-fake.fits'))
    #     fake=True
    #     if not(ra_c1+'_fake' in input_table.colnames) or not(dec_c1+'_fake' in input_table.colnames):
    #         input_table.add_column(input_fake[ra_cf],name=ra_c1+'_fake')
    #         input_table.add_column(input_fake[dec_cf],name=dec_c1+'_fake')
    #         input_table.write(input_fname,overwrite=True)
    
    if not(ra_c1+'_fake' in input_table.colnames and dec_c1+'_fake' in input_table.colnames): #without not()
        print("adding fake coordinates for nway-calibrate-cutoff")
        aux = Table.read(input_fname)
        aux[ra_c1+'_fake'] = (aux[ra_c1]+2*np.random.random(len(aux))-1)%360
        aux[dec_c1+'_fake'] = (aux[dec_c1]+90+2*np.random.random(len(aux))-1)%180-90
        aux.write(input_fname, overwrite=True)
         
    fake=True
    # if not(os.path.isfile(input_fname.replace('.fits','-fake.fits'))):
    aux = Table.read(input_fname)
    aux[ra_c1]=aux[ra_c1+'_fake']
    aux[dec_c1]=aux[dec_c1+'_fake']
    aux.remove_column(ra_c1+'_fake')
    aux.remove_column(dec_c1+'_fake')
    aux.write(input_fname.replace('.fits','-fake.fits'), overwrite=True)
    
        
    
    
    
    for iwv in range(2):
        aux = Table.read(input_fname)
        aux_new = aux[ra_c1,dec_c1,ra_c1+'_fake',dec_c1+'_fake']
        aux_new[ra_c1].name = ra_c4
        aux_new[dec_c1].name = dec_c4
        aux_new[ra_c1+'_fake'].name = ra_c4+'_fake'
        aux_new[dec_c1+'_fake'].name = dec_c4+'_fake'
        aux_new.write('aux.fits',overwrite=True)
        print(len(aux))
        for icds in range(len(cds_tables[iwv])):
            #Do the external CDS crossmatch, to retrieve a subset close enough to X-ray sources
            cds_name = cds_names[iwv][icds]
            print("Working on %s..."%(cds_name))
            if not(skip_actual):
                cdsout_fname = "%s_near_%s.fits"%(cds_name,xname)
                try:
                    skip_this = (os.path.isfile(cdsout_fname) and Table.read(cdsout_fname).meta['PAR_NWAY']==param)
                except:
                    skip_this = 0
                
                
                if not(skip_this):
                    if not(skip_cds):
                        cmd='stilts cdsskymatch in="%s" out="%s" ra="%s" dec="%s"  cdstable="%s" find=all  radius=%.1f'%(input_fname, 'result.fits', ra_c1, dec_c1, cds_tables[iwv][icds],radius*1.25)
                        print(cmd)
                        read=os.popen(cmd).read() 
                        
                    
                    ##First run on actual positions
                    #Remove duplicates
                    result = Table.read("result.fits")
                    if not(skip_cds):
                        result.remove_columns(result.colnames[:len(aux.colnames)])
                        id_c2, ra_c2, dec_c2 = find_id_ra_dec(result,'result.fits')
                        print("Columns identified in CDS table:",id_c2, ra_c2, dec_c2)
                        result[ra_c2].name = "RA"
                        result[dec_c2].name = "DEC"
                        result.remove_columns([c for c in result.colnames if not(c.lower() in [id_c2.lower()]+optircolnames)])
                        result.write('result.fits', overwrite="True")
                    else:
                        ra_c2, dec_c2 = "RA", "DEC"
                    
                    if not(skip_cds):
                        if os.path.isfile(cdsout_fname):
                            os.remove(cdsout_fname)                       
                        cmd='stilts tmatch1 in="result.fits" matcher="exact" values="%s" action="keep1" out="%s"'%(id_c2,cdsout_fname)
                        print(cmd)
                        read=os.popen(cmd).read()
                        if not(os.path.isfile(cdsout_fname)):
                            cmd='cp result.fits %s'%cdsout_fname
                            read=os.popen(cmd).read()
                        # do magnitude calibration between catalogues
                        mag_computations(cds_name, cdsout_fname)
                        # restrict to a few columns
                        result = Table.read(cdsout_fname)
                        result.remove_columns([c_rem for c_rem in result.colnames if not(c_rem in [id_c2,"RA","DEC","Rmag","Bmag","W1mag","W2mag","pm"])])
                        result.meta.update({"PAR_NWAY":param})
                        result.write(cdsout_fname,overwrite=True)
                    if nonmatchonly:
                        cmd='stilts tmatch2 out="%s" matcher="sky" params="15" values1="%s %s" values2="%s %s" find="best1" in1="%s" in2="aux.fits"'%(cdsout_fname, ra_c2,dec_c2,ra_c4,dec_c4,cdsout_fname)
                        print(cmd)
                        read=os.popen(cmd).read()
                        result = Table.read(cdsout_fname)
                        result.remove_columns(result.colnames[-len(aux_new.colnames)+3:])
                        result.meta.update({"PAR_NWAY":param})
                        result.write(cdsout_fname,overwrite=True)
                    
                nmatch = len(Table.read(cdsout_fname))
                
                #Run Nway
                try:
                    skip_this = 0#(os.path.isfile('example-%s-%s.fits'%(cds_name,xname)) and Table.read('example-%s-%s.fits'%(cds_name,xname)).meta['PAR_NWAY']==param)
                except:
                    skip_this = 0
                
                
                if not(skip_nway) and not(skip_this):
                    cmd='nway-write-header.py %s %s %.2f'%(input_fname,xname,coverage)
                    read=os.popen(cmd).read()
                    cmd='nway-write-header.py %s %s %.2f'%(cdsout_fname,cds_name,cds_cov[iwv][icds]*nmatch/cds_ntot[iwv][icds])
                    read=os.popen(cmd).read()
                    
                    cmd='nway.py %s :%s %s 0.1 --out=example-%s-%s.fits --radius %.1f --mag %s:%s auto'%(input_fname, err_c1, cdsout_fname, cds_name, xname, radius, cds_name.upper(), ["Rmag","W1mag"][iwv])
                    print(cmd)
                    read=os.popen(cmd).read()
                    result = Table.read('example-%s-%s.fits'%(cds_name,xname))
                    result.meta.update({"PAR_NWAY":param})
                    result.write('example-%s-%s.fits'%(cds_name,xname),overwrite=True)
                    
                
            if fake:
                cdsout_fname = "%s_near_%s.fits"%(cds_name,xname)
                cdsout_fname2 = cdsout_fname.replace('.fits','-fake.fits')
                try:
                    skip_this = 0#(os.path.isfile(cdsout_fname2) and Table.read(cdsout_fname2).meta['PAR_NWAY']==param)
                except:
                    skip_this = 0
                if not(skip_this):
                    ##Same on fake positions
                    if not(skip_cds):
                        cmd='stilts cdsskymatch in="%s" ra="%s" dec="%s" cdstable="%s" find=all out="result-fake.fits" radius=%.1f'%(input_fname, ra_c1+'_fake', dec_c1+'_fake', cds_tables[iwv][icds],radius*1.25)
                        print(cmd)
                        read=os.popen(cmd).read()
                    #Remove duplicates
                    result = Table.read("result-fake.fits")
                    print("Fake match terminated with success")
                    if not(skip_cds):
                        result.remove_columns(result.colnames[:len(aux.colnames)])
                        id_c2, ra_c2, dec_c2 = find_id_ra_dec(result,'result-fake.fits')
                        print("Columns identified in CDS table:",id_c2, ra_c2, dec_c2)
                        result[ra_c2].name = "RA"
                        result[dec_c2].name = "DEC"
                        result.remove_columns([c for c in result.colnames if not(c.lower() in [id_c2.lower()]+optircolnames)])
                        result.write('result-fake.fits', overwrite="True")
                    else:
                        ra_c2, dec_c2 = "RA", "DEC"
                    
                    if not(skip_cds):
                        if os.path.isfile(cdsout_fname2):
                            os.remove(cdsout_fname2)
                        cmd='stilts tmatch1 in="result-fake.fits" matcher="exact" values="%s" action="keep1" out="%s"'%(id_c2,cdsout_fname2)
                        print(cmd)
                        read=os.popen(cmd).read()
                        if not(os.path.isfile(cdsout_fname2)):
                            cmd='cp result-fake.fits %s'%cdsout_fname2
                            read=os.popen(cmd).read()
                        mag_computations(cds_name, cdsout_fname2)
                        # restrict to a few columns
                        result = Table.read(cdsout_fname2)
                        result.remove_columns([c_rem for c_rem in result.colnames if not(c_rem in [id_c2,"RA","DEC","Rmag","Bmag","W1mag","W2mag"])])
                        result.meta.update({"PAR_NWAY":param})
                        result.write(cdsout_fname2,overwrite=True)
                    if nonmatchonly:
                        cmd = 'stilts tmatch2 out="%s" matcher="sky" params="15" values1="%s %s" values2="%s %s" find="best1" in1="%s" in2="aux.fits"'%(cdsout_fname2, ra_c2,dec_c2,ra_c4+'_fake',dec_c4+'_fake',cdsout_fname2)
                        print(cmd)
                        read=os.popen(cmd).read()
                        result = Table.read(cdsout_fname2)
                        result.remove_columns(result.colnames[-len(aux_new.colnames)+3:])
                        result.meta.update({"PAR_NWAY":param})
                        result.write(cdsout_fname2,overwrite=True)
                
                nmatch = len(Table.read(cdsout_fname2))
                
                #Run Nway
                skip_this = 0#(os.path.isfile('example-%s-%s-fake.fits'%(cds_name,xname)) and Table.read('example-%s-%s-fake.fits'%(cds_name,xname)).meta['PAR_NWAY']==param)
                if not(skip_nway) and not(skip_this):
                    cmd='nway-write-header.py %s %s %.2f'%(input_fname.replace('.fits','-fake.fits'),xname,coverage)
                    read=os.popen(cmd).read()
                    cmd='nway-write-header.py %s %s %.2f'%(cdsout_fname2,cds_name,cds_cov[iwv][icds]*nmatch/cds_ntot[iwv][icds])
                    read=os.popen(cmd).read()
                    
                    use_mag = 'bias_%s_%s'%(cds_name.upper(),['Rmag','W1mag'][iwv]) in Table.read('example-%s-%s.fits'%(cds_name,xname)).colnames
                    if use_mag:
                        cmd='nway.py %s :%s %s 0.1 --out=example-%s-%s-fake.fits --radius %.1f --mag %s:%s %s_%s_fit.txt'%(input_fname.replace('.fits','-fake.fits'), err_c1, cdsout_fname2,
                                                                                                                         cds_name, xname, radius, cds_name.upper(), ["Rmag","W1mag"][iwv], cds_name.upper(), ["Rmag","W1mag"][iwv])
                    else:
                        cmd='nway.py %s :%s %s 0.1 --out=example-%s-%s-fake.fits --radius %.1f'%(input_fname.replace('.fits','-fake.fits'), err_c1, cdsout_fname2,
                                                                                               cds_name, xname,radius)
                    
                    print(cmd)
                    read=os.popen(cmd).read()
                    result = Table.read('example-%s-%s-fake.fits'%(cds_name,xname))
                    result.meta.update({"PAR_NWAY":param})
                    result.write('example-%s-%s-fake.fits'%(cds_name,xname),overwrite=True)
                    
                    ##Compare actual/fake with Nway to calibrate a p_any cutoff
                    cmd='nway-calibrate-cutoff.py example-%s-%s.fits example-%s-%s-fake.fits'%(cds_name,xname,cds_name,xname)
                    print(cmd)
                    read=os.popen(cmd).read()
                    
            print('removing secure matches')
            example_fname = 'example-%s-%s.fits'%(cds_name,xname)
            result = Table.read(example_fname)
            cutoff = np.loadtxt('example-%s-%s.fits_p_any_cutoffquality.txt'
                                %(cds_name,xname))[-1,-1]
            result = np.asarray(result[id_c3][((result['p_any']>max(cutoff,0.001))*(result['match_flag']==1))])
            aux_keep = np.intersect1d(np.asarray(aux[id_c1]),
                                      np.setxor1d(np.asarray(aux[id_c1]),result),
                                      return_indices=True)[1]
            aux = aux[aux_keep]
            aux_new = aux[ra_c1,dec_c1,ra_c1+'_fake',dec_c1+'_fake']
            aux_new[ra_c1].name = ra_c4
            aux_new[dec_c1].name = dec_c4
            aux_new[ra_c1+'_fake'].name = ra_c4+'_fake'
            aux_new[dec_c1+'_fake'].name = dec_c4+'_fake'
            aux_new.write('aux.fits',overwrite=True)
                
        
        results = []
        print("Stacking matches...")
        for icds in range(len(cds_tables[iwv])):
            cds_name = cds_names[iwv][icds]
            print(cds_name)
            example_fname = 'example-%s-%s.fits'%(cds_name,xname)
            result = Table.read(example_fname)
            print('applying cutoff')
            cutoff = np.loadtxt('example-%s-%s.fits_p_any_cutoffquality.txt'
                                %(cds_name,xname))[-1,-1]
            result = result[(result['p_any']>max(cutoff,0.001))*(result['match_flag']==1)]
            print('removing unnecessary columns')
            
            result[id_c3].name = id_c1
            if iwv==0:
                result['%s_RA'%(cds_name.upper())].name = 'RA_Opt'
                result['%s_DEC'%(cds_name.upper())].name = 'DEC_Opt'
                result['%s_Bmag'%( cds_name.upper())].name = 'Bmag'
                result['%s_Rmag'%( cds_name.upper())].name = 'Rmag'
                result['Separation_%s_%s'%(cds_name.upper(),xname)].name = 'angdist_Opt'
                result.add_column(cds_name, name = 'Ref_Opt')
                result = result[[id_c1,'RA_Opt','DEC_Opt','angdist_Opt','Bmag','Rmag','p_single','p_any','Ref_Opt']]
                result['p_any'].name = 'p_any_Opt'
                result['p_single'].name = 'p_single_Opt'
            if iwv==1:
                result['%s_RA'%(cds_name.upper())].name = 'RA_IR'
                result['%s_DEC'%(cds_name.upper())].name = 'DEC_IR'
                result['%s_W1mag'%(cds_name.upper())].name = 'W1mag'
                result['%s_W2mag'%( cds_name.upper())].name = 'W2mag'
                result['Separation_%s_%s'%(cds_name.upper(),xname)].name = 'angdist_IR'
                result.add_column(cds_name, name = 'Ref_IR')
                result = result[[id_c1,'RA_IR','DEC_IR','angdist_IR','W1mag','W2mag','p_single','p_any','Ref_IR']]
                result['p_any'].name = 'p_any_IR'
                result['p_single'].name = 'p_single_IR'
                
            results.append(result)
        print('stacking, removing duplicates and saving')
        result = vstack(results)
        firstind = np.unique(result[id_c1],return_index=True)[1]
        result = result[firstind]
        output_table = Table.read("%s_with_counterparts.fits"%xname)
        id_c1_ind = np.intersect1d(np.asarray(output_table[id_c1]),
                                   np.asarray(result[id_c1]),
                                   return_indices=True)[1:]
        result = hstack([output_table[id_c1_ind[0]],result[id_c1_ind[1]]])
        result['%s_1'%id_c1].name = id_c1
        result.remove_column('%s_2'%id_c1)
        nonmatch = np.intersect1d(np.asarray(output_table[id_c1]),
                                  np.setxor1d(np.asarray(output_table[id_c1]),np.asarray(result[id_c1])),
                                  return_indices=True)[1]
        aux_new = output_table[nonmatch]
        for col in np.setxor1d(output_table.colnames,result.colnames):
            if isinstance(result[col][0],str):
                aux_new.add_column('',name=col)
            else:
                aux_new.add_column(np.nan,name=col)
        result = vstack([result,aux_new])
        result.write('%s_with_counterparts.fits'%(xname), overwrite=True)
        print('done')
    
  
        
        
