#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:01:37 2021

@author: htranin
"""

from astropy.table import Table
from astropy.io import fits
import numpy as np
from auto_nway import find_id_ra_dec
import os
import sys


# IDENTIFICATION


if len(sys.argv)>1:
    input_fname = sys.argv[1]
else:
    input_fname = 'CSC2_with_counterparts_x_loc.fits'
    
xname = input_fname.replace('_',' ').replace('-',' ').split()[0]

input_table = Table.read(input_fname)
id_c1, ra_c1, dec_c1 = find_id_ra_dec(input_table,input_fname)
if xname=="2SXPS":
    err_c1 = 'Err63'
elif xname[:4]=="4XMM":
    err_c1 = "SC_POSERR"
else:
    err_c1 = "err_halfmaj_63"

ncol = len(input_table.colnames)

#List of external catalogues

vizcat = {'AGN':['J/ApJS/221/12/table1', 'VII/258/vv10', 'VII/283/catalog'],
          'Star':['I/280B/ascc'],
          'XRB':['B/cb/lmxbdata','J/A+A/469/807/lmxb','J/ApJ/689/983/table6','J/ApJ/662/525/Xbin','J/A+A/533/A33/table6',
                 'J/A+A/455/1165/table1','J/MNRAS/419/2095/hmxb','J/MNRAS/466/1019/table2',
                 'J/ApJS/222/15/table12','J/A+A/587/A61/tablea1'],
          'CV':['B/cb/cbdata','V/123A/cv']}

#vizref = {'AGN':['Miragn', 'Veron', 'Milliquas'],
#          'Star':['Ascc'],
#          'XRB':['Ritter','Liu07','Humphrey','Kundu','Zhang',
#                 'Liu06','Mineo','Sazonov',
#                 'Watchdog','Blackcat'],
#          'CV':['Ritter','Downes']}

simcat = {'AGN':['AGN','Seyfert_1','Seyfert_2','BLLac','Blazar','QSO'],
          'Star':['YSO','TTau*','Em*','Orion_V*','PM*','WR*','RotV*','EB*'],
          'XRB':['XB','LMXB','HMXB'],
          'CV':['CataclyV*','Nova']}

for typ in vizcat.keys():
    input_table['is%s'%typ] = 0
    for cat in vizcat[typ]:
        cmd = 'stilts cdsskymatch in="%s" ra="%s" dec="%s" cdstable="%s" find=best out="result.fits" radius=3 ocmd="select angDist<3*%s+%.1f"'%(
            input_fname,ra_c1,dec_c1,cat,err_c1,0.1+0.9*(typ in "XRB CV"))
        os.system(cmd)
        res = fits.open('result.fits')[1].data
        if cat=="VII/283/catalog":
            res = res[[res['Qpct'][i]>80 and res[res.names[5+ncol]][i].replace('WISE','J')[0]=="J" for i in range(len(res))]]
        i,i1,i2 = np.intersect1d(input_table[id_c1],res[id_c1],return_indices=1)
        input_table['is%s'%typ][i1] += 2**vizcat[typ].index(cat)

cmd = 'stilts cdsskymatch in="%s" ra="%s" dec="%s" cdstable="simbad" find=best out="result.fits" radius=3 ocmd="select angDist<3*%s+%.1f"'%(
            input_fname,ra_c1,dec_c1,err_c1,0.5)
os.system(cmd)
res = fits.open('result.fits')[1].data

for typ in simcat.keys():
    for typ2 in simcat[typ]:
        res2 = res[res['main_type']==typ2]
        i,i1,i2 = np.intersect1d(input_table[id_c1],res2[id_c1],return_indices=1)
        input_table['is%s'%typ][i1] += 1024

input_table['class'] = np.nan
input_table['class'][np.logical_and(abs(input_table['isAGN']-512)!=512,input_table['isStar']+input_table['isXRB']+input_table['isCV']==0)] = 0
input_table['class'][np.logical_and(abs(input_table['isStar']-512)!=512,input_table['isAGN']+input_table['isXRB']+input_table['isCV']==0)] = 1                           
input_table['class'][np.logical_and(abs(input_table['isXRB']-512)!=512,input_table['isStar']+input_table['isAGN']+input_table['isCV']==0)] = 2
input_table['class'][np.logical_and(abs(input_table['isCV']-512)!=512,input_table['isStar']+input_table['isXRB']+input_table['isAGN']==0)] = 3
if "qual" in input_table.colnames:
    input_table['class'][input_table['qual']<2] = np.nan

input_table.write(input_fname.replace('.','_typ.'), overwrite=True)