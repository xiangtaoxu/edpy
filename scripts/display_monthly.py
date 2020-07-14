#!usr/bin/env python
# import modules here
import h5py
import os
import numpy as np
from math import isinf
import sys
sys.path.append('/n/home10/xiangtao/git_repos/edpy/')
import edpy as edpy
import pandas as pd

fn_name = sys.argv[1] # file name

print(fn_name)

h5in = h5py.File(fn_name,'r')

# print the LAI profile for every 5 meter

hite_dict = {}
htop_dict = {}
var_list = ['LAI','MMEAN_DMIN_LEAF_PSI','MMEAN_DMAX_LEAF_PSI']
hite_size_list = ('H',np.arange(0,50+1,5.))
htop_size_list = ('HTOP',np.arange(0,50+1,5.))

for var in var_list:
    # H Size vars
    for isize, size_edge in enumerate(hite_size_list[1]):
        hite_dict['{:s}_{:s}_{:d}'.format(
            var,hite_size_list[0],isize)] = []
    # HTOP Size vars
    for isize, size_edge in enumerate(htop_size_list[1]):
        htop_dict['{:s}_{:s}_{:d}'.format(
            var,htop_size_list[0],isize)] = []


edpy.extract_utils.extract_size(h5in,hite_dict,var_list,
                                hite_size_list)
edpy.extract_utils.extract_size(h5in,htop_dict,var_list,
                                htop_size_list)


hite_df = pd.DataFrame(hite_dict)
htop_df = pd.DataFrame(htop_dict)

MMEAN_PRCP = h5in['MMEAN_PCPG_PY'][0] * 86400 * 30
MMEAN_VPD = h5in['MMEAN_ATM_VPDEF_PY'][0] / 1000. # kPa
MMEAN_ET = np.array(h5in['MMEAN_TRANSP_PY'][:]
                    +h5in['MMEAN_VAPOR_LC_PY'][:]
                    +h5in['MMEAN_VAPOR_WC_PY'][:]
                    +h5in['MMEAN_VAPOR_GC_PY'][:])[0] * 86400 * 30. # mm/month
MMEAN_GPP = np.array(h5in['MMEAN_GPP_PY'][:])[0] # kgC/m2/yr
MMEAN_NPP = np.array(h5in['MMEAN_NPP_PY'][:])[0] # kgC/m2/yr
MMEAN_LEAF_RESP = np.array(h5in['MMEAN_LEAF_RESP_PY'][:])[0] # kgC/m2/yr
MMEAN_STEM_RESP = np.array(h5in['MMEAN_STEM_RESP_PY'][:])[0] # kgC/m2/yr
MMEAN_ROOT_RESP = np.array(h5in['MMEAN_ROOT_RESP_PY'][:])[0] # kgC/m2/yr
MMEAN_GROWTH_RESP = np.array(
    h5in['MMEAN_LEAF_GROWTH_RESP_PY'][:] + 
    h5in['MMEAN_ROOT_GROWTH_RESP_PY'][:] + 
    h5in['MMEAN_SAPA_GROWTH_RESP_PY'][:] + 
    h5in['MMEAN_SAPB_GROWTH_RESP_PY'][:] 
                    )[0] # kgC/m2/yr


print('''Ecosystem Level Variables:
         PRCP = {:f}
         VPD  = {:f}
         ET = {:f}
         GPP = {:f}
         NPP = {:f}
         LEAF_RESP = {:f}
         ROOT_RESP = {:f}
         STEM_RESP = {:f}
         GROWTH_RESP = {:f}
         LAI = {:f}
         BA = {:f}
         AGB = {:f}'''.format(
             MMEAN_PRCP, MMEAN_VPD,
             MMEAN_ET,MMEAN_GPP,
             MMEAN_NPP,MMEAN_LEAF_RESP,MMEAN_ROOT_RESP,
             MMEAN_STEM_RESP,MMEAN_GROWTH_RESP,
             np.nansum(h5in['MMEAN_LAI_PY'][:].ravel()),
             np.nansum(h5in['BASAL_AREA_PY'][:].ravel()),
             np.nansum(h5in['AGB_PY'][:].ravel())
         ))

print('BA by PFT')
print(np.nansum(h5in['BASAL_AREA_PY'][:],1).ravel())
print('AGB by PFT')
print(np.nansum(h5in['AGB_PY'][:],1).ravel())
print('BA by DBH')
print(np.nansum(h5in['BASAL_AREA_PY'][:],2).ravel())
print('LAI by PFT')
print(np.nansum(h5in['MMEAN_LAI_PY'][:],1).ravel())
print('LAI by height')
print(hite_df.iloc[0,0:len(hite_size_list[1])].values / 5.)
print('DMIN_PSI by height')
print(hite_df.iloc[0,len(hite_size_list[1]):(2*len(hite_size_list[1]))].values )
print('DMAX_PSI by height')
print(hite_df.iloc[0,(2*len(hite_size_list[1]))::].values )

print('DMIN_PSI by HTOP')
print(htop_df.iloc[0,len(htop_size_list[1]):(2*len(htop_size_list[1]))].values )
print('DMAX_PSI by HTOP')
print(htop_df.iloc[0,(2*len(htop_size_list[1]))::].values )

h5in.close()
