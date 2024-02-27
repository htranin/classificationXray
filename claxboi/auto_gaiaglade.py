#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 21:23:19 2021

@author: htranin
"""

import os
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from auto_nway import find_id_ra_dec
import numpy as np
import sys
skip_glade = 0
skip_pm = 0
skip_dist = 0

if len(sys.argv)>1:
    input_fname = sys.argv[1]
else:
    input_fname = "4XMM_with_counterparts.fits"
    
xname = input_fname.replace('_',' ').replace('-',' ').split()[0]
input_table = Table.read(input_fname)

id_c1, ra_c1, dec_c1, fx_c1 = find_id_ra_dec(input_table, input_fname,
                                             return_flux=True)

output_fname = input_fname.replace(".fits","_loc.fits")

# GLADE
if not(skip_glade):    
    glade_fname = "../data/GLADE2016_corrected_0123.fits"
    glade_table = Table.read(glade_fname)
    ## RECENT UPDATE 11/1/22    
    #glade_table = glade_table[abs(glade_table['Dist']-175.5)<=174.5]
    glade_table = glade_table[glade_table['to_keep']]
    ## OLD WAY
    #glade_table = glade_table[glade_table['R1']<6000] #remove LMC,SMC
    #glade_table = glade_table[~glade_table['2MASX_flag']] #remove 2MASS spurious galeeeee
    id_c2, ra_c2, dec_c2 = find_id_ra_dec(glade_table, glade_fname)
    glade_table[ra_c2].name = 'RA_GLADE'
    glade_table[dec_c2].name = 'DEC_GLADE'
    glade_table['Bmag'].name = 'Bmag_GLADE'
    glade_table.write('aux_glade.fits', overwrite=True)
    ## OLD WAY: params="300"
    cmd = 'stilts tmatch2 matcher="skyellipse" find="best1" join="all1" out="%s" in1="%s" in2="aux_glade.fits" params="60" values1="%s %s %s %s %s" values2="%s %s %s %s %s"'%(
        output_fname, input_fname, ra_c1, dec_c1, 1.5, 1.5, 0, 'RA_GLADE', 'DEC_GLADE', '1.265*R1', '1.265*R2', 'PA')
    print(cmd)         
    os.system(cmd)
    # LATER CORRECTION: prevent this match to mix up column descriptions
    res = Table.read(output_fname)
    res['Separation'].name = 'Separation_GLADE'
    theta = np.asarray(SkyCoord(res[ra_c1],res[dec_c1],unit="deg").position_angle(SkyCoord(res['RA_GLADE'],res['DEC_GLADE'],unit="deg")))
    pa = res['PA']*np.pi/180
    angdist= SkyCoord(res[ra_c1],res[dec_c1],unit="deg").separation(SkyCoord(res['RA_GLADE'],res['DEC_GLADE'],unit="deg")).arcsec
    rad_at_ang = (res['R1']*res['R2'])/np.sqrt((res['R2']*np.cos(theta-pa))**2+(res['R1']*np.sin(theta-pa))**2)
    SepToRadius = angdist / rad_at_ang
    res.add_column(np.asarray(SepToRadius), name="SepToRadius")
    res.add_column(np.asarray(res[fx_c1])*4*np.pi*(np.asarray(res['Dist'])*3.086e24)**2,name="Lx_1")
    if not('b' in res.colnames):
        res.add_column(180/np.pi*np.arcsin(np.cos(res[dec_c1]/180*np.pi)*np.cos(27.128336/180*np.pi)*np.cos((res[ra_c1]-192.859508)/180*np.pi)+np.sin(res[dec_c1]/180*np.pi)*np.sin(27.128336/180*np.pi)),name='b')
    if not('l' in res.colnames):
        res.add_column((122.9320-(180/np.pi)*np.arctan2(np.cos(res[dec_c1]/180*np.pi)*np.sin((res[ra_c1]-192.8595)/180*np.pi),(np.sin(res[dec_c1]/180*np.pi)-np.sin(27.1284/180*np.pi)*np.sin(res['b']/180*np.pi))/np.cos(27.1284/180*np.pi))+360)%360,name='l')
    res.add_column(np.asarray(np.log10(res[fx_c1]))+5.37+np.asarray(res['Bmag']/2.5),name='logFxFb')
    res.add_column(np.asarray(np.log10(res[fx_c1]))+5.37+np.asarray(res['Rmag']/2.5),name='logFxFr')
    res.add_column(np.asarray(np.log10(res[fx_c1]))+6.95+np.asarray(res['W1mag']/2.5),name='logFxFw1')
    res.add_column(np.asarray(np.log10(res[fx_c1]))+6.95+np.asarray(res['W2mag']/2.5),name='logFxFw2')
    res["Lx_1"][res[fx_c1]==0] = np.nan
    res["logFxFb"][res[fx_c1]==0] = np.nan
    res["logFxFr"][res[fx_c1]==0] = np.nan
    res["logFxFw1"][res[fx_c1]==0] = np.nan
    res["logFxFw2"][res[fx_c1]==0] = np.nan
    if 'fratio' in res.colnames and 'SNR' in res.colnames:
        res.add_column((~np.isnan(res['logFxFr'])).astype(int)+
                       (~np.isnan(res['logFxFw1'])).astype(int)+
                       (res['fratio']>0).astype(int)+
                       (res['SNR']>10).astype(int), name="qual")
        
    colout = input_table.colnames
    for cnew in ['l','b','pgc','RA_GLADE','DEC_GLADE','R1','R2','PA','Dist','type','t','e_t','spiral','elliptical','prob_sp','prob_el','stellar_mass','stellar_mass_err','logSFR_12u','Separation_GLADE','SepToRadius','Lx_1','logFxFb','logFxFr','logFxFw1','logFxFw2','qual']:
        if not(cnew in input_table.colnames) and cnew in res.colnames:
            colout.append(cnew)
    res = res[colout]
    
    res.write(output_fname,overwrite=True)



# GAIA EDR3 proper motions
if not(skip_pm):
    res = Table.read(output_fname)
    result = Table.read('example-%s-%s.fits'%('Gaia',xname))
    cutoff = np.loadtxt('example-%s-%s.fits_p_any_cutoffquality.txt'
                        %('Gaia',xname))[-1,-1]
    result = result[(result['p_any']>max(cutoff,0.001))*(result['match_flag']==1)]
    result = result["%s_%s"%(xname,id_c1),"GAIA_pm"]
    result.write("aux_Gaia.fits", overwrite=True)            
    os.system('stilts tmatch2 matcher="exact" find="best1" join="all1"'
               +' ocmd="delcols %d" out="%s" in1="%s" in2="aux_Gaia.fits" values1="%s" values2="%s"'
               %(len(res.colnames)+1,output_fname,output_fname,
                 id_c1,'%s_%s'%(xname,id_c1)))

# GAIA EDR3 distances
if not(skip_dist):
    output_table = Table.read(output_fname)
    cmd ='stilts cdsskymatch in="%s" ra="%s" dec="%s" cdstable="%s" find=best out="result.fits" radius=2'%(output_fname, 'RA_Opt', 'DEC_Opt', "Gaia EDR3 distances")
    print(cmd)
    os.system(cmd)
    res = Table.read('result.fits')
    res['rpgeo'].name = 'GAIA_Dist'
    res.add_column(np.asarray(res[fx_c1])*4*np.pi*(np.asarray(res['GAIA_Dist'])*3.086e18)**2,name="Lx_2")
    res["Lx_2"][res[fx_c1]==0] = np.nan
    res = res[output_table.colnames+["GAIA_Dist","Lx_2"]]
    


    # complete with nonmatches of the last step

    nonmatch = np.intersect1d(np.asarray(output_table[id_c1]),
                              np.setxor1d(np.asarray(output_table[id_c1]),np.asarray(res[id_c1])),
                              return_indices=True)[1]
    aux_new = output_table[nonmatch]
    for col in np.setxor1d(output_table.colnames,res.colnames):
        if isinstance(res[col][0],str):
            aux_new.add_column('',name=col)
        else:
            aux_new.add_column(np.nan,name=col)


    result = vstack([res,aux_new])
    result.write(output_fname,overwrite=True)