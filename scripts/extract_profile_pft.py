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
import os
import re
from pathlib import Path
import argparse # for processing arguments of this script


################################################
# Process script argument
################################################

parser = argparse.ArgumentParser(
    description='Options for extracting customized profile data from qmean ED2 output')

# add argument to store directory of data input and output
parser.add_argument('--data_dir','-dd',action='store',default='./',type=str)
parser.add_argument('--out_dir','-od',action='store',default='./',type=str)

# add argument to store output prefix
parser.add_argument('--prefix','-p',action='store',default='',type=str)

# add argument to indicate the start and end year/month
# default is None
parser.add_argument('--yeara','-ya',action='store',default=None,type=int)
parser.add_argument('--yearz','-yz',action='store',default=None,type=int)
parser.add_argument('--montha','-ma',action='store',default=1,type=int)
parser.add_argument('--monthz','-mz',action='store',default=12,type=int)

# add argument to indicate which pfts are included
parser.add_argument('--pft','-pft',action='store',default=[],nargs='+',type=int)

# add argument to indicate what kind of size classes to use
# default is using height, possible options are HTOP and CTOP
parser.add_argument('--size_type','-st',action='store',default='H',type=str)


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
    
#################################################

# TODO: move line 70-625 to another utility file
#       only retain file I/O and flow control 
###################################################
# First, some constants for profile extraction
###################################################

#-------------------------------------------------
# 1.1 Variable Types
#-------------------------------------------------
var_area = ['QMEAN_LEAF_WATER','QMEAN_WOOD_WATER',
            'QMEAN_INTERCEPTED_AL','QMEAN_INTERCEPTED_AW',
            'QMEAN_SENSIBLE_LC','QMEAN_SENSIBLE_WC','QMEAN_TRANSP',
            'QMEAN_LEAF_WATER_IM2','QMEAN_WOOD_WATER_IM2',
            'NPLANT','MMEAN_LAI']


# these variables need to be scaled with area-weighted nplant
var_nplant = ['QMEAN_GPP','QMEAN_NPP','QMEAN_LEAF_RESP','QMEAN_PLRESP',
              'AGB','BA','MMEAN_BLEAF','MMEAN_BROOT','BDEADA','BDEADB',
              'BSAPWOODA','BSAPWOODB','MMEAN_BBARKA','MMEAN_BBARKB',
              'MMEAN_BSTORAGE']
# all these variables have 4 dimensions except HITE and DBH

# these variables need to be normalized by area-weighed plants
var_nplant_norm = ['DDBH_DT','MMEAN_MORT_RATE','HITE','DBH']
# all these variables have 3 dimensions time, size, pft

# these variables need to be normalized with area-eighted lai
var_lai_norm = ['QMEAN_A_NET','QMEAN_LEAF_GBW','QMEAN_LEAF_GSW',
                'QMEAN_LEAF_TEMP','QMEAN_LEAF_VPDEF']
# all these have 4 dimensions

# these variables need to be normalized with area-weighted ba
var_ba_norm = ['QMEAN_LEAF_PSI','QMEAN_WOOD_PSI','QMEAN_WOOD_TEMP']
# all these have 4 dimensions


# Here we set arrays to record variables that do not need to followed by _CO
var_noco = ['DBH','PFT','NPLANT','DDBH_DT','DBA_DT','HITE',
            'BSAPWOODA','BSAPWOODB','BDEADA','BDEADB']

# special wood-related variables that need to be distributed into different size classes
var_wood = ['BDEADA','BSAPWOODA','MMEAN_BBARKA','QMEAN_WOOD_WATER_IM2']
# WARNING! var_wood has to be a var_nplant or a var_area, not a normalized variable
# otherwise the results would be problematic

#-------------------------------------------------

#-------------------------------------------------
# 1.2 Size classes and other dimensions of the data set
#-------------------------------------------------

# default size classes
# modify as you see fit
if args.size_type == 'H':
    coord_size = np.arange(0,100+1,5) # every 5 m until 100m
elif args.size_type == 'HTOP':
    coord_size = np.arange(0,100+1,5) # every 5 m until 100m
elif args.size_type == 'CTOP':
    coord_size = np.arange(0,20+0.1,0.5) # every 0.5 kgC/m2 until 20 kgC/m2
    
# PFT
coord_pft = args.pft

###################################################


###################################################
# Second, define a few utility functions 
###################################################

#-------------------------------------------------
# 2.1 extract size information
#-------------------------------------------------
def extract_size_info(
     ds_in          : 'input xarray dataset that has all cohort-level data'
    ,size_list      : 'A list contains the edge of size bins'
    ,size_type      : 'type of calculating size classes'
):
    '''
        Return size_idx and size_hite
        size_idx is a 1-d array recording the size class of each cohort
        size_frac is a 2-d array storing the fraction of biomass/hite that 
           falls into each size class for each cohort
        
    '''
    # first get size_width from size_list
    if isinstance(size_list,list):
        size_list = np.array(size_list)
    
    # common data shared by all cases
    HITE = ds_in['HITE'].values
    PACO_ID   = ds_in['PACO_ID'].values
    PACO_N    = ds_in['PACO_N'].values
    
    SIZE = np.zeros_like(HITE)
        
    size_hite_edge = np.zeros((len(SIZE),len(size_list)),dtype=float)
    size_hite_width = np.zeros_like(size_hite_edge,dtype=float)
        
    
    #=================================
    # Now dive into different size types
    #=================================
    if size_type == 'H':
        # simplest case
        # use height to separate classes
        
        # loop over patches
        for ipa, paco_start in enumerate(PACO_ID):
            cohort_mask = (
                (np.arange(len(SIZE)) >= PACO_ID[ipa]-1) &
                (np.arange(len(SIZE)) < PACO_ID[ipa]+PACO_N[ipa]-1)
            )
            # define cohort_level hite_edge
            for isize, size_edge in enumerate(size_list):
                size_hite_edge[cohort_mask,isize] = size_edge
            
            hmax_pa = np.amax(HITE[cohort_mask])
            
            # update size_hite_width based on size_hite_edge
            size_hite_width[cohort_mask,0:-1] = (
                  size_hite_edge[cohort_mask,1::] 
                - size_hite_edge[cohort_mask,0:-1]
            )
            
            size_hite_width[cohort_mask,-1] = (
                np.maximum(0.,
                             hmax_pa
                           - size_hite_edge[cohort_mask,-1]
                          )
            )
        
        
    elif size_type == 'HTOP':
        # more complex
        # use height from the top as size classes
        # note the top of canopy is different for each patch
        
        # loop over patches
        for ipa, paco_start in enumerate(PACO_ID):
            cohort_mask = (
                (np.arange(len(SIZE)) >= PACO_ID[ipa]-1) &
                (np.arange(len(SIZE)) < PACO_ID[ipa]+PACO_N[ipa]-1)
            )
            
            hmax_pa = np.amax(HITE[cohort_mask])
            
            # define cohort_level hite_edge
            for isize, size_edge in enumerate(size_list):
                size_hite_edge[cohort_mask,isize] = (
                    np.maximum(0.,hmax_pa - size_edge)
                )
            
            # update size_hite_width based on size_hite_edge
            # these values would be NEGATIVE
            size_hite_width[cohort_mask,0:-1] = (
                  size_hite_edge[cohort_mask,1::] 
                - size_hite_edge[cohort_mask,0:-1]
            )

            # the lower bound of the last classs should be groud (0)
            size_hite_width[cohort_mask,-1] = (
                            0      
                            - size_hite_edge[cohort_mask,-1]
            )
        

            
        # finally modify site_hite_edge and site_hite_width
        # so they count from bottom to top (consistent with 'H')
        size_hite_edge = size_hite_edge + size_hite_width
        size_hite_width = -1 * size_hite_width

            
    elif size_type == 'CTOP':
        # even more complex
        # use cumulative carbon from the top as size classes
        # note the top of canopy is different for each patch
    
        
        NPLANT    = ds_in['NPLANT'].values
        BLEAF     = ds_in['MMEAN_BLEAF_CO'].values
        BDEADA    = ds_in['BDEADA'].values
        BSAPWOODA = ds_in['BSAPWOODA'].values
        BBARKA    = ds_in['MMEAN_BBARKA_CO'].values
        BSTORAGEA = ( ds_in['MMEAN_BSTORAGE_CO'].values 
            * ds_in['BSAPWOODA'].values
            / (ds_in['BSAPWOODA'].values +
               ds_in['BSAPWOODB'].values)
        )
        # scale by NPLANT (no need to scale by AREA because we only
        # compare within each patch)
        BSTEMA = (BDEADA + BSAPWOODA + BBARKA + BSTORAGEA) * NPLANT
        BLEAF = BLEAF * NPLANT
        
        # loop over patches
        for ipa, paco_start in enumerate(PACO_ID):
            cohort_mask = (
                (np.arange(len(SIZE)) >= PACO_ID[ipa]-1) &
                (np.arange(len(SIZE)) < PACO_ID[ipa]+PACO_N[ipa]-1)
            )
            
            # make sure cohorts are sorted descending with height
            cohort_array = np.arange(len(SIZE))[cohort_mask]
            sort_idx = np.argsort(HITE[cohort_mask])[::-1]
            cohort_array = cohort_array[sort_idx]
            
            hmax_pa = np.amax(HITE[cohort_array])
            h_pa = HITE[cohort_array]
            b_pa = (BLEAF + BSTEMA)[cohort_array]
            
            b_pa_dens    = np.cumsum(b_pa / h_pa) 
            # density of biomass (per m in height) at the top of each cohort
            
            b_pa_hite    = np.zeros_like(b_pa_dens)
            # array to store the effective height for height segement with the same density of biomass
            b_pa_hite[0:-1] = h_pa[0:-1] - h_pa[1::]
            b_pa_hite[-1]   = h_pa[-1] - 0.  # distance to the ground
            
            # total biomass within each trunk is density * height
            b_pa_cum     =  b_pa_dens * b_pa_hite
            # now cumulative over the height
            b_pa_cum     =  np.cumsum(b_pa_cum)
                
        
            # now we loop over carbon size classes
            for isize, size_edge in enumerate(size_list):
                
                ########################
                # Scenario 1
                ########################
                if size_edge == 0.:
                    # the first class
                    # no cohorts are included
                    # set size_hite_edge to tht top most canopy height
                    size_hite_edge[cohort_mask,isize] = hmax_pa
                    continue
                
                # positive size_edge values
                # find the first ico that has a cumulative larger than size_edge
                result = np.argwhere(b_pa_cum >= size_edge)
                
                ########################
                # Scenario 2
                ########################
                if len(result) == 0:
                    # cumulative biomass of the whole patch
                    # is not enough to reach size_edge
                    
                    
                    # set the corresponding hite to be (equivalent to ground-level)
                    size_hite_edge[cohort_mask,isize] = 0.
                    continue
                
                ########################
                # Scenario 3
                ########################
                # we find the first one that is larger than size_edge
                # we need to find the corresponding hite using b_pa_dens[i_first]
                i_first = result[0][0]
                c_diff = b_pa_cum[i_first] - size_edge 
                if (i_first+1) < len(h_pa):
                    size_hite_edge[cohort_mask,isize] = (
                          h_pa[i_first+1]
                        + c_diff / b_pa_dens[i_first]
                    )
                else:
                    size_hite_edge[cohort_mask,isize] = (
                          c_diff / b_pa_dens[i_first]
                    )
            
            # update size_hite_width based on size_hite_edge
            # these values would be NEGATIVE
            size_hite_width[cohort_mask,0:-1] = (
                  size_hite_edge[cohort_mask,1::] 
                - size_hite_edge[cohort_mask,0:-1]
            )
            
            # the lower bound of the last classs should be groud (0)
            size_hite_width[cohort_mask,-1] = (
                            0      
                            - size_hite_edge[cohort_mask,-1]
            )

        
        # finally modify site_hite_edge and site_hite_width
        # so they count from bottom to top (consistent with 'H')
        size_hite_edge = size_hite_edge + size_hite_width
        size_hite_width = -1 * size_hite_width
            
            # since we get size_hite_edge
            # define SIZE the same way as HTOP
            #SIZE[cohort_mask] = np.amax(HITE[cohort_mask]) - HITE[cohort_mask]
    else:
        print(f'Error! Only H, HTOP, CTOP are supported. Yours is {size_type}!')
        
        return -1
    
    
    #=================================
    # End dealing with different size types
    # Should have all info need to calculate size_idx and size_frac
    #=================================
    
    # based on SIZE, size_hite_edge, size_hite_width
    # get size_idx and size_frac
    size_idx = np.zeros_like(HITE,dtype=int)
    size_frac = np.zeros((len(HITE),len(size_list)),dtype=float)
    
    
    for isize, size_edge in enumerate(size_list):
        
        size_bottom = size_hite_edge[:,isize] 
        size_upper  = size_bottom + size_hite_width[:,isize]
        
        # right inclusive will work for all scenarios
        # left inclusive will lead to missing counting for scenario "H"
        cohort_mask = (HITE > size_bottom) & (HITE <= size_upper)
        
            
        # assign size_idx and size_hite
        size_idx[cohort_mask] = isize
        
        cohort_array = np.arange(len(HITE))[cohort_mask]
        for ico in cohort_array:
            size_frac[ico,:] = (
                np.minimum(
                    np.maximum(0.,
                                 HITE[ico]
                               - size_hite_edge[ico,:]
                              )
                    ,size_hite_width[ico,:]
                ) / HITE[ico]
            )
    
    
    # return size_idx and size_frac
    return (size_idx, size_frac)
        

#-------------------------------------------------
# 2.2 extract profile
#-------------------------------------------------
# function to calculate extract vertical profile for aboveground for each pft
def extract_profile_pft(
     ds_in          : 'input xarray dataset that has all cohort-level data'
    ,voi_list       : 'profile variables of interests'
    ,coord_time     : 'a list of time coordinate for xarray'
    ,coord_size     : 'a list of size coordinate for xarray'
    ,coord_pft      : 'a list of pft coordinate for xarray'
    ,coord_hour     : 'a list of hour coordinate for xarray' 
                    = range(24)
    ,coord_mort     : 'a list of coordinate for mortality types' 
                    = ['Age','CSM','BTF','HFM','Cld','Dst']
    ,size_type      : 'type of calculating size classes'
                    = 'H'
    
):
    '''
        Return a data set by time, size, pft, hour, mort type
        
        Use extract_size_info
        
    '''
    
    # first generate size_idx, and size_frac necessary for averaging size groups
    size_idx, size_frac = extract_size_info(ds_in,coord_size,size_type)
    
    
    #########################################################################
    # create and prepare a dataset to store the profiles
    #########################################################################
    ds = xr.Dataset()
    
    # add coordinates
    ds.coords['time'] = coord_time
    ds.coords['size'] = coord_size
    ds.coords['pft']  = coord_pft
    ds.coords['hour'] = coord_hour
    ds.coords['mort'] = coord_mort
    
           
    # first generate key scaling variables for each cohort
    # i.e. AREA, AGB, BA, NPLANT, MMEAN_LAI
    
    AREA      = ds_in['AREA'].values
    PACO_ID   = ds_in['PACO_ID'].values
    PACO_N    = ds_in['PACO_N'].values
    NPLANT    = ds_in['NPLANT'].values
    BA        = ds_in['BA_CO'].values
    MMEAN_LAI = ds_in['MMEAN_LAI_CO'].values
    HITE      = ds_in['HITE'].values # used to distribute wood carbon/water
    
    # get AREA for each cohort
    AREA_co      = np.zeros_like(NPLANT,dtype=float)
    for ipa, paco_start in enumerate(PACO_ID):
        cohort_mask = (
            (np.arange(len(NPLANT)) >= PACO_ID[ipa]-1) &
            (np.arange(len(NPLANT)) < PACO_ID[ipa]+PACO_N[ipa]-1)
        )
        AREA_co[cohort_mask] = AREA[ipa]
    
    # calculate the other scaling variables weighted by AREA_co
    NPLANT_co = NPLANT * AREA_co
    BA_co = BA * NPLANT_co
    MMEAN_LAI_co = MMEAN_LAI * AREA_co
    
    # calculate aboveground sapwood ration
    bsapwood_a_frac_co = (
        ds_in['BSAPWOODA'].values / 
        (ds_in['BSAPWOODA'].values + ds_in['BSAPWOODB'].values)
    )
    
    # now that we have all the weighting factors loop through all variable of interets
    PFT = ds_in.PFT.values # this is also necessary
    #########################################################################
    
    
    #########################################################################
    # create and populate data arrays
    #########################################################################
    for ivar, var in enumerate(voi_list):
        # -------------------------------------------------------------------------------------
        # create data array
        # -------------------------------------------------------------------------------------
        if var[0:5] == 'QMEAN':
            # QMEAN variable need to add hour dimension
            ds[var] = (("time","size","pft","hour")
                      ,np.full((len(coord_time),len(coord_size),len(coord_pft),len(coord_hour)),
                               fill_value=np.nan,dtype=float))
            ds[var].name = var
        elif var in ['MMEAN_MORT_RATE']:
            # special case need to consider mortality
            ds[var] = (("time","size","pft",'mort')
                      ,np.full((len(coord_time),len(coord_size),len(coord_pft),len(coord_mort)),
                               fill_value=np.nan,dtype=float))
            ds[var].name = var
            
        else:
            # not QMEAN variable, only three dimensions
            ds[var] = (("time","size","pft")
                      ,np.full((len(coord_time),len(coord_size),len(coord_pft)),
                               fill_value=np.nan,dtype=float))
            ds[var].name = var
 
        # -------------------------------------------------------------------------------------
    
        # -------------------------------------------------------------------------------------
        # Read data from ds_in
        # -------------------------------------------------------------------------------------
        if var in var_noco:
            var_data = ds_in[var]
        else:
            var_data = ds_in[var+'_CO']
            
        # add Metadata
        ds[var].attrs['Metadata'] = var_data.Metadata
        
        # convert var_data to numpy array
        var_data = var_data.values
        
        # reshape var_data to 2-d array if it is 1-d
        # this can faciliate later operations
        if len(var_data.shape) == 1:
            var_data = np.reshape(var_data,(var_data.shape[0],1))

        dim2_len = var_data.shape[1]
        
        
        # deal with special cases
        if var in ['QMEAN_WOOD_WATER_IM2','QMEAN_WOOD_WATER_INT',
                   'MMEAN_BSTORAGE']:
            # these two variables records total wood water
            # we need to convert them to aboveground values
            
            var_data *= np.reshape(np.repeat(bsapwood_a_frac_co,dim2_len),var_data.shape)
            
        # -------------------------------------------------------------------------------------
        
        
        # -------------------------------------------------------------------------------------
        # loop over size and pft classes to aggregate data
        # for wood-mass-related variables, use size_frac
        # for others (leaf or whole plant related), use size_list
        # -------------------------------------------------------------------------------------
        for i, size_edge in enumerate(coord_size):
            size_mask = (size_idx == i)
            
            for j, pft in enumerate(coord_pft):
                pft_mask = (PFT == pft)
                
                cohort_mask = pft_mask & size_mask
                
                
                if sum(cohort_mask) == 0:
                    # no cohort in this class
                    # no need to do anything
                    # skip the rest of the loop
                    continue
                
                #------------------------------------------------------------
                # if the program executes here it means there are cohorts in 
                # this size-pft class
                #------------------------------------------------------------
                    
                # calculate scaler
                # by default is 1 (identity with raw data)
                scaler = np.ones_like(AREA_co,dtype=float)
                if var in var_area:
                    scaler[cohort_mask] = AREA_co[cohort_mask]
                elif var in var_nplant:
                    scaler[cohort_mask] = NPLANT_co[cohort_mask]
                elif var in var_nplant_norm:
                    scaler[cohort_mask] = NPLANT_co[cohort_mask] / np.sum(NPLANT_co[cohort_mask])
                elif var in var_ba_norm:
                    scaler[cohort_mask] = BA_co[cohort_mask] / np.sum(BA_co[cohort_mask])
                elif var in var_lai_norm:
                    if np.sum(MMEAN_LAI_co[cohort_mask]) == 0.:
                        # in case LAI are all zero (e.g. in non-growing season)
                        scaler[cohort_mask] = MMEAN_LAI_co[cohort_mask] * 0.
                    else:
                        scaler[cohort_mask] = MMEAN_LAI_co[cohort_mask] / np.sum(MMEAN_LAI_co[cohort_mask])

                # scaling the variables
                var_data *= np.reshape(np.repeat(scaler,dim2_len),var_data.shape)
                

                # avoid loop over ico, and use cohort_mask instead
                
                # loop over the second dimension
                for k in np.arange(dim2_len):
                    
                    if var in var_wood:
                        # distribute mass
                        # need to update all index of size dimension
                        
                        data_update_dim = (np.sum(cohort_mask),len(coord_size))
                        
                        coord_size_update = np.arange(len(coord_size))
                        
                        # multiply var_data[cohort_mask,k] with size_frac from with class
                        # using np.repeat and reshape functionality
                        data_update = (
                              size_frac[cohort_mask,:]
                            * np.reshape(
                                np.repeat(
                                     var_data[cohort_mask,k].squeeze()
                                    ,len(coord_size)
                                )
                                , data_update_dim
                              )
                        )
                        
                        
                        # we need to sum up the cohort dimension and get sum for each size class
                        data_update = np.nansum(data_update,axis=0)
                        
                    else:
                        # normal scenario
                        # only need to update size class i
                        coord_size_update = i
                        data_update = np.nansum(var_data[cohort_mask,k],axis=0)
                
                    # update ds[var]
                    # use nan_to_num incase it is the first time to assign the value
                    if dim2_len == 1:
                        # second dimension has only a length of 1
                        # effectively 1-dimensional
                        ds[var][0,coord_size_update,j] =(
                              np.nan_to_num(ds[var][0,coord_size_update,j])
                            + data_update
                        )

                    else:
                        # true 2-dimensional
                        ds[var][0,coord_size_update,j,k] =(
                              np.nan_to_num(ds[var][0,coord_size_update,j,k])
                            + data_update
                        )


                            
        # -------------------------------------------------------------------------------------
    #########################################################################
    
    
    return ds

#-------------------------------------------------



###################################################
# Third, file I/O and actual data extraction
###################################################
data_path = Path(args.data_dir)
# only allow for qmean data
file_list = np.array(sorted(list(data_path.glob(f'{args.prefix}-Q-*.h5'))))


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

# Now we have the file list
# loop over the files and save the output dataset
ds_list = []

# by default extract all possible variables
voi_list = (
      var_area + var_nplant
    + var_nplant_norm + var_ba_norm + var_lai_norm
)

for f in file_list:
    # get year and month to create time stamp
    sub_strs = f.name.split('-') # -Q-year-month-date
    year, month = int(sub_strs[2]),int(sub_strs[3])
    
    ds_in = xr.open_dataset(f)
    
    coord_time = [f'{year:04d}-{month:02d}-01']
    
    # add the profile dataset to array
    ds_list.append(
        extract_profile_pft(
             ds_in
            ,voi_list
            ,coord_time
            ,coord_size
            ,coord_pft
            ,size_type = args.size_type
        )
    )
    

# finally concatenate all data set together
# save it to a .nc file

ds_out = xr.concat(ds_list,dim='time')

output_fn = args.out_dir + args.prefix + f'_profile_{args.size_type}.nc'
ds_out.to_netcdf(output_fn)
###################################################
