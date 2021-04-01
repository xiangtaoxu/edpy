###############################################################
# This script is designed to be attached with any ED2 simulation
# that allows for monthly/qmean output.
# It will store all common polygon level results in a netcdf file
# use the python xarray package.
# It will also plot key monthly timeseries for model interpretation
###############################################################

# import packages
# Notes: I am trying to only import standard packages in python3
#        so that it can be used for most linux OS
# Here we need to use xarray
import numpy as np
import xarray as xr
import datetime
import os
import re
from pathlib import Path
import argparse # for processing arguments of this script


################################################
# Process script argument
################################################

parser = argparse.ArgumentParser(
    description='Options for extracting and plotting monthly ED2 output')

# add argument to store directory of data input and output
parser.add_argument('--data_dir','-dd',action='store',default='./',type=str)
parser.add_argument('--out_dir','-od',action='store',default='./',type=str)

# add argument to store output prefix
parser.add_argument('--prefix','-p',action='store',default='',type=str)

# add argument to indicate the start and end year/month
# default is None
parser.add_argument('--yeara','-ya',action='store',default=None,type=int)
parser.add_argument('--yearz','-yz',action='store',default=None,type=int)
parser.add_argument('--montha','-ma',action='store',default=None,type=int)
parser.add_argument('--monthz','-mz',action='store',default=None,type=int)

# add argument to indicate which pfts are included
parser.add_argument('--pft','-pft',action='store',default=[1,2,3,4],nargs='+')

# Feel free to add more arguments here
# e.g. output directory etc.


# Parse the argument and get its values
args = parser.parse_args()
# the values of each flag can be accessed through
#    args.submit, args.cpu etc.

# conduct some quality check for dir
if args.data_dir[-1] != '/':
    args.data_dr = args.data_dir + '/'

if args.out_dir[-1] != '/':
    args.out_dir = args.out_dir + '/'
    
# calculate pft_indexes
pft_indexes = [i-1 for i in args.pft]

#################################################




###################################################
# First, create a list of files to extract and plot
###################################################
data_path = Path(args.data_dir)
file_list = np.array(sorted(list(data_path.glob(f'{args.prefix}*.h5'))))

# modify file_list based on yeara, yearz, montha, monthz
remove_indexes = []
for i, f in enumerate(file_list):
    substrs = f.name.split('-')
    year = int(substrs[2])
    month = int(substrs[3])
    
    # decide whether this month should be skipped
    is_remove = False
    if (args.yeara is not None):
        if (year < args.yeara):
            is_remove=True
        elif (args.montha is not None) and (year == args.yeara and month < args.montha):
            is_remove=True
    if (args.yearz is not None):
        if (year > args.yearz):
            is_remove=True
        elif (args.monthz is not None) and (year == args.yearz and month > args.monthz):
            is_remove=True
    if is_remove:
        remove_indexes.append(i)
        
file_list = np.delete(file_list,remove_indexes)
###################################################

###################################################
# Initialize xarray dataset                       #
###################################################
var3d_list = ['AGB_PY','BASAL_AREA_PY','NPLANT_PY','MMEAN_LAI_PY']
var1d_list = ['MMEAN_FAST_GRND_C_PY','MMEAN_FAST_SOIL_C_PY',
              'MMEAN_SLOW_SOIL_C_PY',
              'MMEAN_STRUCT_GRND_C_PY','MMEAN_STRUCT_SOIL_C_PY']
var1d_qmean_list = [
    'QMEAN_GPP_PY','QMEAN_NPP_PY','QMEAN_NEP_PY','QMEAN_CARBON_AC_PY',
    'QMEAN_VAPOR_AC_PY','QMEAN_TRANSP_PY','QMEAN_SENSIBLE_AC_PY',
    'QMEAN_ATM_CO2_PY','QMEAN_ATM_RSHORT_PY','QMEAN_ATM_TEMP_PY',
    'QMEAN_ATM_VPDEF_PY','QMEAN_PCPG_PY',
    'QMEAN_CAN_CO2_PY','QMEAN_CAN_TEMP_PY','QMEAN_CAN_VPDEF_PY']

# soil
var_soil_qmean_list = ['QMEAN_SOIL_WATER_PY','QMEAN_SOIL_TEMP_PY','QMEAN_SOIL_MSTPOT_PY']

# create data set
dataset_py = xr.Dataset()

# create coords
ctime = []
csize = range(11)
cpft  = range(1,17+1)
chour = range(24)

# read the first file and get soil information
ds = xr.open_dataset(file_list[0])
csoil_layer = ds['SLZ'].values
nzg = len(csoil_layer)

dataset_py.coords['time'] = ctime
dataset_py.coords['size'] = csize
dataset_py.coords['pft'] = cpft
dataset_py.coords['hour'] = chour
dataset_py.coords['soil_layer'] = csoil_layer

# create data vars

for var in var3d_list:
    dataset_py[var] = (("size","pft","time"),np.reshape([],(len(csize),len(cpft),len(ctime))))
    dataset_py[var].name = var
    
for var in var1d_list:
    dataset_py[var] = (("time"),np.reshape([],(len(ctime))))
    dataset_py[var].name = var

for var in var1d_qmean_list:
    dataset_py[var] = (("hour","time")
                      ,np.reshape([],(len(chour),len(ctime))))
    dataset_py[var].name = var   
    
for var in var_soil_qmean_list:
    dataset_py[var] = (("soil_layer","hour","time")
                      ,np.reshape([],(len(csoil_layer),len(chour),len(ctime))))
    dataset_py[var].name = var        

total_vars = var1d_list + var3d_list + var1d_qmean_list + var_soil_qmean_list
###################################################


###################################################
# Loop over file_list and concatenate datasets
###################################################
for i, f in enumerate(file_list):
    # get date information
    sub_strs = f.name.split('-') # -Q-year-month-date
    year, month = int(sub_strs[2]),int(sub_strs[3])
    f_ds = xr.open_dataset(f)
    
    
    # only extract variable of interests    
    f_ds = f_ds[total_vars]
    
    # rename dims
    # loop over the dims 
    # find the ones correponding to size, pft, time, hour, and soil_layer

    for dim_name in f_ds.dims:
        if f_ds.dims[dim_name] == 11:
            # size
            size_name = dim_name
        elif f_ds.dims[dim_name] == 17:
            # pft
            pft_name = dim_name
        elif f_ds.dims[dim_name] == 1:
            # time
            time_name = dim_name
        elif f_ds.dims[dim_name] == 24:
            # hour
            hour_name = dim_name
        elif f_ds.dims[dim_name] == nzg:
            # soil_layer
            soil_layer_name = dim_name
        
            
    
    
    f_ds = f_ds.rename_dims(
        {size_name : 'size',
         pft_name  : 'pft',
         time_name : 'time',
         hour_name : 'hour',
         soil_layer_name : 'soil_layer'})
    
    
    # assign coords
    f_ds = f_ds.assign_coords(
    {'time' : [f'{year:4d}-{month:02d}-01'],
     'size' : csize,
     'pft'  : cpft,
     'hour' : chour,
     'soil_layer' : csoil_layer}
    )
    
    # concatenate dataset
    dataset_py = xr.concat([dataset_py,f_ds],join='inner',dim='time')
    
    if i == 0:
        # first time also copy attribute
        for var in var1d_list + var3d_list:
            dataset_py[var].attrs['Metadata']=f_ds[var].Metadata
        
# only select PFT 1-4 to save space
dataset_py = dataset_py.isel(pft=pft_indexes)
# transpose dataset since time is most often used
dataset_py = dataset_py.transpose('soil_layer','size','pft','hour','time')
# save the dataset as a netcdf file
output_fn = args.out_dir + args.prefix + '.nc'
dataset_py.to_netcdf(output_fn)
###################################################

###################################################
# Plot key polygon level diagnostic figure
# 1. ecosystem states by PFT and by size
# 2. key fluxes over the last 10 (or shorter) years
# 3. climate over last 10 years
# 4. soil over last 10 years
###################################################
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# first need to change the time coords to datetime
dataset_py.coords['time'] = [
    datetime.datetime.fromisoformat(datestr+'-01') 
    for datestr in dataset_py.coords['time'].values]



# Fig.1 ecosystem states by PFT and by size
pft_labels = [f'PFT {i}' for i in dataset_py.pft.values]
size_labels = (
    [f'{i*10}-{(i+1)*10}cm' for i in dataset_py.size.values[0:-1]] + 
    [f'>{dataset_py.size.values[-1]*10} cm']
)

var_plot = ['AGB_PY','BASAL_AREA_PY','MMEAN_LAI_PY','NPLANT_PY']
nx, ny = 2,4
fig, axes = plt.subplots(nx,ny,figsize=(3*ny,3*nx))
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        
        var = var_plot[j]
        try:
            unit = re.search('\[(.+?)\]', dataset_py[var].Metadata[1]).group(1)
        except AttributeError:
            # pattern not found
            unit = '' # error handling

        if i == 0:
            # by pft
            labels = pft_labels
            cmap = cm.get_cmap('tab10')
            da_plot = dataset_py[var].copy()
            
            if var == 'NPLANT_PY':
                # only include >10cm cohorts for nplant
                da_plot.loc[dict(size=0)] = 0.
                title = da_plot.name + ' >10cm'
            else:
                title = da_plot.name
            
            da_plot = da_plot.sum(dim=['size'])
            
        elif i == 1:
            # by size
            cmap = cm.get_cmap('tab20')
            labels = size_labels
            da_plot = dataset_py[var].copy()

            if var == 'NPLANT_PY':
                # set 0-10cm to zero
                da_plot.loc[dict(size=0)] = 0.
                title = da_plot.name + ' >10cm'

            else:
                title= da_plot.name
            
            da_plot = da_plot.sum(dim=['pft'])
        
        # plot the figures
        ax.stackplot(da_plot.time.values,da_plot.values,cmap=cmap,
                     labels=labels)

        if j == 0:
            ax.legend(loc='upper left',fontsize=7)

        # title and labels
        ax.set_title(title)

        ax.set_ylabel(unit)

fig.tight_layout()
fig_fn = args.out_dir + args.prefix + '_states.png'
plt.savefig(fig_fn,dpi=300)
plt.close(fig)


#Fig. 2 fluxes
var_plot = ['QMEAN_GPP_PY','QMEAN_NPP_PY','QMEAN_NEP_PY',
            'QMEAN_VAPOR_AC_PY','QMEAN_TRANSP_PY','QMEAN_SENSIBLE_AC_PY']

if len(dataset_py.time) > 120:
    # more than 10 years
    # get the last 10 year
    dataset_plot = dataset_py.isel(time=slice(-120,None))
else:
    dataset_plot = dataset_py.copy()

nx, ny = len(var_plot),2
fig, axes = plt.subplots(nx,ny,figsize=(3*ny,2*nx))
for i, row in enumerate(axes):
    var = var_plot[i]

    if var in ['QMEAN_VAPOR_AC_PY','QMEAN_TRANSP_PY']:
        # water flux
        dataset_plot[var] *= 86400
        unit = 'kgH2O/m2/d'
    elif var in ['QMEAN_SENSIBLE_AC_PY']:
        # energy flux
        unit = 'W/m2'
    else:
        # all carbon vars
        unit = 'kgC/m2/yr'

    for j, ax in enumerate(row):
        
        
        if j == 0:
            # monthly mean
            dataset_plot[var].mean(dim='hour').plot(x='time',ax=ax)
                    
        elif j == 1:
            # average diurnal cycle
            dataset_plot[var].mean(dim='time').plot(x='hour',ax=ax)

        
        ax.set_title(var[6::])
        
        ax.set_ylabel(unit)

fig.tight_layout()
fig_fn = args.out_dir + args.prefix + '_fluxes.png'
plt.savefig(fig_fn,dpi=300)
plt.close(fig)

#Fig. 3 climate

# figure for climate conditions
var_plot = ['QMEAN_ATM_CO2_PY','QMEAN_ATM_RSHORT_PY','QMEAN_ATM_TEMP_PY',
            'QMEAN_ATM_VPDEF_PY','QMEAN_PCPG_PY',
            'QMEAN_CAN_CO2_PY','QMEAN_CAN_TEMP_PY','QMEAN_CAN_VPDEF_PY']

if len(dataset_py.time) > 120:
    # more than 10 years
    # get the last 10 year
    dataset_plot = dataset_py.isel(time=slice(-120,None))
else:
    dataset_plot = dataset_py.copy()

nx, ny = len(var_plot),2
fig, axes = plt.subplots(nx,ny,figsize=(3*ny,1.5*nx))
for i, row in enumerate(axes):
    var = var_plot[i]

    if var in ['QMEAN_PCPG_PY']:
        # water flux
        dataset_plot[var] *= 86400
        unit = 'kgH2O/m2/d'
    elif var in ['QMEAN_ATM_TEMP_PY','QMEAN_CAN_TEMP_PY']:
        # temperature
        dataset_plot[var] -= 273.15 # conver to degC
        unit = 'degC'
    elif var in ['QMEAN_ATM_VPDEF_PY','QMEAN_CAN_VPDEF_PY']:
        # vpd
        dataset_plot[var] /= 1000. # convert to kPa
        unit = 'kPa'
    elif var in ['QMEAN_ATM_CO2_PY','QMEAN_CAN_CO2_PY']:
        unit = 'ppm'
    elif var in ['QMEAN_ATM_RSHORT_PY']:
        # radiation
        unit = 'W/m2'

    for j, ax in enumerate(row):
        
        
        if j == 0:
            # monthly mean
            dataset_plot[var].mean(dim='hour').plot(x='time',ax=ax)
                    
        elif j == 1:
            # average diurnal cycle
            dataset_plot[var].mean(dim='time').plot(x='hour',ax=ax)

        
        ax.set_title(var[6::])
        
        ax.set_ylabel(unit)

fig.tight_layout()

fig_fn = args.out_dir + args.prefix + '_climate.png'
plt.savefig(fig_fn,dpi=300)
plt.close(fig)

#Fig. 4 soil
var_plot = ['QMEAN_SOIL_WATER_PY','QMEAN_SOIL_MSTPOT_PY','QMEAN_SOIL_TEMP_PY']

if len(dataset_py.time) > 120:
    # more than 10 years
    # get the last 10 year
    dataset_plot = dataset_py.isel(time=slice(-120,None)).copy()
else:
    dataset_plot = dataset_py.copy()

nx, ny = len(var_plot),2
fig, axes = plt.subplots(nx,ny,figsize=(4*ny,3*nx))
for i, row in enumerate(axes):
    var = var_plot[i]

    if var in ['QMEAN_SOIL_WATER_PY']:
        # soil moisture
        unit = 'm3H2O/m3Volume'
    elif var in ['QMEAN_SOIL_TEMP_PY']:
        # temperature
        dataset_plot[var] -= 273.15 # conver to degC
        unit = 'degC'
    elif var in ['QMEAN_SOIL_MSTPOT_PY']:
        # soil water potential
        dataset_plot[var] /= 102 # convert to MPa
        unit = 'MPa'
        
    
    for j, ax in enumerate(row):
        if j == 0:
            # monthly mean
            dataset_plot[var].mean(dim='hour').plot(x='time',y='soil_layer',ax=ax)
                    
        elif j == 1:
            # average diurnal cycle
            dataset_plot[var].mean(dim='time').plot(x='hour',y='soil_layer',ax=ax)

        
        ax.set_title(unit)

fig.tight_layout()
fig_fn = args.out_dir + args.prefix + '_soil.png'
plt.savefig(fig_fn,dpi=300)
plt.close(fig)
###################################################
