#!usr/bin/env python
# import modules here
import h5py
import os
import numpy as np
from math import isinf
import sys
import sys
file_path = os.path.abspath(__file__)

#split by /
path_list = file_path.split('/')

# get the parent folder path
edpy_path = '/'.join(path_list[0:-2])
print(edpy_path)
sys.path.append(edpy_path)

import edpy as edpy
import pandas as pd
import statsmodels.api as sm

h5_name = sys.argv[1] # file name

PFT_array = [2,3,4]
mmdry= 0.02897
# plot gsw_open vs A/ca/sqrt(vpd) for each pft
#conduct regression analysis

h5in = h5py.File(h5_name,'r')
QMEAN_LEAF_GSW_OPEN = np.array(h5in['QMEAN_LEAF_GSW_CO']) / mmdry  #[mol/m2l/s]
QMEAN_LEAF_GBW_OPEN = np.array(h5in['QMEAN_LEAF_GBW_CO']) / mmdry  #[mol/m2l/s]
QMEAN_LEAF_VPDEF = np.array(h5in['QMEAN_LEAF_VPDEF_CO']) / 1000. #[kPa]
QMEAN_A_OPEN = np.array(h5in['QMEAN_A_OPEN_CO']) #[umol/m2l/s]
PFT = np.array(h5in['PFT'])
QMEAN_PAR_LEVEL_BEAM = np.array(h5in['QMEAN_PAR_LEVEL_BEAM_CO']) # W/m2
QMEAN_PAR_LEVEL_DIFFU = np.array(h5in['QMEAN_PAR_LEVEL_DIFFU_CO']) # W/m2
QMEAN_CAN_CO2_PY = np.array(h5in['QMEAN_CAN_CO2_PY']) #ppm

MMEAN_ET = np.array(h5in['MMEAN_TRANSP_PY'][:]
                    +h5in['MMEAN_VAPOR_LC_PY'][:]
                    +h5in['MMEAN_VAPOR_WC_PY'][:]
                    +h5in['MMEAN_VAPOR_GC_PY'][:]) * 86400 * 30. # mm/month
MMEAN_GPP = np.array(h5in['MMEAN_GPP_PY'][:]) # kgC/m2/yr
h5in.close()

leaf_gsc = 1 / ((1 / (QMEAN_LEAF_GSW_OPEN / 1.6)) + (1 / (QMEAN_LEAF_GBW_OPEN / 1.4)))
#leaf_gsc = QMEAN_LEAF_GSW_OPEN / 1.6

can_co2 = np.nanmean(QMEAN_CAN_CO2_PY[QMEAN_CAN_CO2_PY < 400.])
print(QMEAN_CAN_CO2_PY)
print('CO2 = {:f}'.format(can_co2))
print('ET = {:f}'.format(MMEAN_ET[0]))
print('GPP = {:f}'.format(MMEAN_GPP[0]))

for ipft, pft in enumerate(PFT_array):
    print('PFT : {:d}'.format(pft))

    PFT_mask = PFT == pft

    sub_gsc = leaf_gsc[PFT_mask,:].ravel()
    sub_vpd = QMEAN_LEAF_VPDEF[PFT_mask,:].ravel()
    sub_A   = QMEAN_A_OPEN[PFT_mask,:].ravel()
    sub_PAR = (QMEAN_PAR_LEVEL_BEAM + QMEAN_PAR_LEVEL_DIFFU)[PFT_mask,:].ravel()

    A_mask = (sub_A > 4.)

    sub_gsc = sub_gsc[A_mask]
    sub_vpd = sub_vpd[A_mask]
    sub_A = sub_A[A_mask]
    sub_PAR = sub_PAR[A_mask]

    print('GSC : ', np.percentile(sub_gsc,[1,50,99]))
    print('A_net: ' , np.percentile(sub_A,[1,50,99]))

    reg_Y = sub_gsc - sub_A / can_co2
    reg_X = sub_A / can_co2 /(sub_vpd ** 0.5)

    reg_res = sm.OLS(reg_Y,reg_X).fit()
    print(reg_res.summary())

    # save data
    #output_df = pd.DataFrame(
    #    {'gsc' : sub_gsc,
    #     'vpd' : sub_vpd,
    #     'A'   : sub_A,
    #     'PAR' : sub_PAR,
    #     'reg_x' : reg_X,
    #     'reg_Y' : reg_Y
    #    }
    #)
    #output_df.to_csv('PFT{:d}.csv'.format(pft))


    #sub_par = (QMEAN_PAR_LEVEL_BEAM + QMEAN_PAR_LEVEL_DIFFU)[PFT_mask,:]


