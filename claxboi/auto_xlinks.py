#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:00:14 2023

@author: htranin
"""

from astropy.table import Table, vstack
import numpy as np
import os


os.chdir('/home/htranin/')

reduce_detections = 1

if reduce_detections:
    xmmdet = Table.read("Downloads/4XMM_DR12cat_v1.0.fits")
    xmmdet = xmmdet[np.logical_and(xmmdet["EP_EXTENT"]==0,xmmdet["SUM_FLAG"]<=1)]
    xmmdet["SRCID"].name="srcid"
    xmmdet["DETID"].name="detid"
    xmmdet["SC_RA"].name="ra"
    xmmdet["SC_DEC"].name="dec"
    xmmdet["SC_POSERR"].name="err63"
    xmmdet["EP_8_FLUX"].name="flux"
    xmmdet["flux_err"]=xmmdet["SC_EP_8_FLUX_ERR"]
    xmmdet.remove_columns([c for c in xmmdet.colnames if not(c in ["srcid","detid","ra","dec","err63","flux","flux_err"])])
    xmmdet.write("4XMM_DR12_detections_thin.fits", overwrite=True)
    xmmdet = xmmdet[np.argsort(xmmdet['srcid'])]
    xmmsrc = xmmdet[np.unique(xmmdet['srcid'], return_index=1)[1]]
    #xmmsrc.write("4XMM_DR12_udet_thin.fits", overwrite=True) # like the unique source catalogue
    
    swidet = Table.read("Downloads/LSXPS_Detections.fits")
    swidet = swidet[np.logical_and(swidet["DetFlag"]<=1,np.logical_and(swidet["ObsID"]<1e10,swidet['Band']))]
    swidet["LSXPS_ID"].name="srcid"
    swidet["DetectionID"].name="detid"
    swidet["RA"].name="ra"
    swidet["Decl"].name="dec"
    swidet["err63"] = swidet["Err90"]/1.52
    swidet["flux"] = swidet["Rate"]*4.3e-11
    swidet["flux_err"] = (swidet["Rate_pos"]-swidet["Rate_neg"])/2*4.3e-11
    swidet.remove_columns([c for c in swidet.colnames if not(c in ["srcid","detid","ra","dec","err63","flux","flux_err"])])
    swidet.write("LSXPS_detections_thin.fits", overwrite=True)
    swidet = swidet[np.argsort(swidet['srcid'])]
    swisrc = swidet[np.unique(swidet['srcid'], return_index=1)[1]]
    #swisrc.write("2SXPS_udet_thin.fits", overwrite=True)
    
    
    chadet = Table.read("CSC2_detections.fits")
    chadet = chadet[np.logical_and(~chadet['sat_src_flag'],np.logical_and(~chadet['streak_src_flag'],np.logical_and(chadet['flux_aper_b']>0,np.logical_and(chadet['conf_code']<256,chadet['extent_code']<256))))]
    chadet["name"].name="srcid"
    chadet["detid"]=np.int64(chadet['ra']*1e11+(chadet['dec']+180)*1e6+chadet['obsid']+chadet['obi']*1e4)
    chadet["err63"] = chadet["err_ellipse_r0"]/1.73
    chadet["flux_aper_b"].name="flux"
    chadet["flux_err"] = (chadet["flux_aper_hilim_b"]-chadet["flux_aper_lolim_b"])/2
    chadet.remove_columns([c for c in chadet.colnames if not(c in ["srcid","detid","ra","dec","err63","flux","flux_err"])])
    chadet.write("CSC2_detections_thin.fits", overwrite=True)
    chadet = chadet[np.argsort(chadet['srcid'])]
    chasrc = chadet[np.unique(chadet['srcid'], return_index=1)[1]]
    #chasrc.write("CSC2_udet_thin.fits", overwrite=True)


"""
   This program identifies common sources in different X-ray catalogues,
   assigning a unique MASTER_ID to each unique source, and computes the
   maximum-to-miminum flux ratios (with or without flux error). A special
   attention is given to the identification and removal of any ambiguous
   match, where a source is associated to several other sources at both
   1 and 3 sigma (position errors). Spurious and flagged detections may 
   be discarded as well.
"""

for names in []:#"4XMM_DR12 CSC2", "2SXPS CSC2", "4XMM_DR12 2SXPS"]:
    name1, name2 = names.split()[0], names.split()[1]
    cmd = 'stilts tmatch2 matcher="skyerr" find="all" join="1and2" out="match3sig_%s_%s.fits" in1="%s_udet_thin.fits" in2="%s_udet_thin.fits" params="1" values1="ra dec 3*err63" values2="ra dec 3*err63"'%(
        name1, name2, name1, name2)
    print(cmd)
    os.popen(cmd).read()

    cmd = 'stilts tmatch2 matcher="skyerr" find="all" join="1and2" out="match1sig_%s_%s.fits" in1="%s_udet_thin.fits" in2="%s_udet_thin.fits" params="1" values1="ra dec err63" values2="ra dec err63"'%(
        name1, name2, name1, name2)
    print(cmd)
    os.popen(cmd).read()

# Next steps :
    # select np.isnan(groupid) in match3sig and np.isnan(groupid) in match1sig being not nan in match3sig
    # create links : MASTERID assignment
    # save new udet catalogues
    # compute flux ratios