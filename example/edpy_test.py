#!usr/bin/env python
# import modules here
# use mpi4py to process output parallelly
import numpy as np
import sys
import os
import pandas as pd
sys.path.append('../../edpy/')
import edpy as edpy

# default values for input
PFT_NAMES = np.array(['E','M','L'])
pf_name = './ed2_output/example'
pft_list = [2,3,4]
pft_names = PFT_NAMES
output_yeara = 1910
output_yearz = 1919
data_dir = './data/'
figure_dir = './figure/'

# create the directory
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

# get the model prefix as the prefix for the output csv file and figure file
sim_pf = pf_name.split('/')[-1] + '_'

# extract annual
edpy.extract_annual(
    pf_name,
    data_dir,
    sim_pf,
    output_yeara,
    output_yearz,
    pft_list = pft_list)

# plot annual
edpy.plot_annual(
    data_dir,
    sim_pf,
    figure_dir,
    pft_list,
    pft_names,
    include_census=True)

# extract monthly
edpy.extract_monthly(
    pf_name,
    data_dir,
    sim_pf,
    output_yeara,1,
    output_yearz,1,
    pft_list = pft_list
)

# plot monthly
edpy.plot_monthly(
    data_dir,
    sim_pf,
    figure_dir,
    pft_list,
    pft_names,
    include_census=True)



# extract tree ring
edpy.extract_treering(
    pf_name,
    data_dir,
    sim_pf,
    output_yeara,
    output_yearz,
    last_month_of_year=12)


# extract qmean
edpy.extract_monthly_diurnal(pf_name,data_dir,sim_pf,1751,1,1751,12)
# plot qmean
edpy.plot_monthly_diurnal(data_dir,sim_pf,figure_dir)
