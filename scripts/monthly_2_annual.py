#!usr/bin/env python
# import modules here
import h5py
import os
import numpy as np
from math import isinf
import sys
import pandas as pd

fn_pf = sys.argv[1] # file prefix

print(fn_pf)

ET = 0.
GPP = 0.
NPP = 0.
LEAF_RESP = 0.
STEM_RESP = 0.
ROOT_RESP = 0.
GROWTH_RESP = 0.
LEAF_MAIN, ROOT_MAIN, LEAF_DROP = 0., 0., 0.
LAI = 0.
BA = 0.


for month in np.arange(1,12+1):
    fn_name = '{:s}-{:02d}-00-000000-g01.h5'.format(fn_pf,month)
    h5in = h5py.File(fn_name,'r')


    MMEAN_ET = np.array(h5in['MMEAN_TRANSP_PY'][:]
                        +h5in['MMEAN_VAPOR_LC_PY'][:]
                        +h5in['MMEAN_VAPOR_WC_PY'][:]
                        +h5in['MMEAN_VAPOR_GC_PY'][:])[0] * 86400 * 30. # mm/month
    MMEAN_GPP = np.array(h5in['MMEAN_GPP_PY'][:])[0] # kgC/m2/yr
    MMEAN_NPP = np.array(h5in['MMEAN_NPP_PY'][:])[0] # kgC/m2/yr
    MMEAN_LEAF_RESP = np.array(h5in['MMEAN_LEAF_RESP_PY'][:])[0] # kgC/m2/yr
    MMEAN_STEM_RESP = np.array(h5in['MMEAN_STEM_RESP_PY'][:])[0] # kgC/m2/yr
    MMEAN_ROOT_RESP = np.array(h5in['MMEAN_ROOT_RESP_PY'][:])[0] # kgC/m2/yr
    MMEAN_LAI = np.sum(h5in['MMEAN_LAI_PY'][:])
    MMEAN_BA = np.sum(h5in['BASAL_AREA_PY'][:])
    MMEAN_GROWTH_RESP = np.array(
        h5in['MMEAN_LEAF_GROWTH_RESP_PY'][:] + 
        h5in['MMEAN_ROOT_GROWTH_RESP_PY'][:] + 
        h5in['MMEAN_SAPA_GROWTH_RESP_PY'][:] + 
        h5in['MMEAN_SAPB_GROWTH_RESP_PY'][:] 
                        )[0] # kgC/m2/yr
    MMEAN_LEAF_MAINTENANCE = np.sum(h5in['MMEAN_LEAF_MAINTENANCE_PY'][:])
    MMEAN_ROOT_MAINTENANCE = np.sum(h5in['MMEAN_ROOT_MAINTENANCE_PY'][:])
    MMEAN_LEAF_DROP = np.sum(h5in['MMEAN_LEAF_DROP_PY'][:])

    h5in.close()
    ET += (MMEAN_ET / 12.)
    GPP += (MMEAN_GPP / 12.)
    NPP += (MMEAN_NPP / 12.)
    LEAF_RESP += (MMEAN_LEAF_RESP / 12.)
    ROOT_RESP += (MMEAN_ROOT_RESP / 12.)
    STEM_RESP += (MMEAN_STEM_RESP / 12.)
    GROWTH_RESP += (MMEAN_GROWTH_RESP / 12.)
    LEAF_MAIN += (MMEAN_LEAF_MAINTENANCE / 12.)
    ROOT_MAIN += (MMEAN_ROOT_MAINTENANCE / 12.)
    LEAF_DROP += (MMEAN_LEAF_DROP / 12.)
    LAI += (MMEAN_LAI / 12.)
    BA += (MMEAN_BA / 12.)

print(LEAF_MAIN,ROOT_MAIN,LEAF_DROP)
print('''Ecosystem Level Variables:
         ET = {:f}
         GPP = {:f}
         NPP = {:f}
         LEAF_RESP = {:f} ROOT_RESP = {:f}  STEM_RESP = {:f}
         GROWTH_RESP = {:f}
         LEAF_MAIN = {:f} ROOT_MAIN = {:f} LEAF_DROP = {:f}
         LAI = {:f}
         BA = {:f}'''.format(
             ET,GPP,NPP,LEAF_RESP,ROOT_RESP,STEM_RESP,GROWTH_RESP,
             LEAF_MAIN,ROOT_MAIN,LEAF_DROP,
             LAI,BA
         ))


