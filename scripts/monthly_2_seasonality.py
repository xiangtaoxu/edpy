import h5py
import os
import numpy as np
from math import isinf
import sys
sys.path.append('/n/home10/xiangtao/git_repos/edpy/')
import edpy as edpy
import pandas as pd

fn_pf = sys.argv[1] # file prefix

print(fn_pf)

ET = np.zeros((12,1),dtype=float)
GPP, NPP, LAI, BA = np.zeros_like(ET), np.zeros_like(ET), np.zeros_like(ET), np.zeros_like(ET)
LEAF_RESP, STEM_RESP, ROOT_RESP, GROWTH_RESP = np.zeros_like(ET), np.zeros_like(ET), np.zeros_like(ET), np.zeros_like(ET)
LEAF_MAIN, ROOT_MAIN, LEAF_DROP = np.zeros_like(ET), np.zeros_like(ET), np.zeros_like(ET)

for imonth, month in enumerate(np.arange(1,12+1)):
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
    ET[imonth] = MMEAN_ET
    GPP[imonth] = MMEAN_GPP
    NPP[imonth] = MMEAN_NPP
    LEAF_RESP[imonth] = MMEAN_LEAF_RESP
    ROOT_RESP[imonth] = MMEAN_ROOT_RESP
    STEM_RESP[imonth] = MMEAN_STEM_RESP
    GROWTH_RESP[imonth] = MMEAN_GROWTH_RESP
    LAI[imonth] = MMEAN_LAI
    BA[imonth] = MMEAN_BA

print('Seasonality:\n')
print('ET = ',ET)
print('GPP = ',GPP)
print('NPP = ',NPP)
print('LAI = ',LAI)
print('BA = ',BA)


