# This module contains low level classes and functions 
# to extract ED2 output from hdf5 output files


# import modules here
import os
import sys
import numpy as np
import h5py
import pandas as pd
from calendar import monthrange
from datetime import datetime

#################################################
# object used in this file
class EDTime(object):

    # class attribute

    # name of the time attributes to include
    time_names = ['year','month','day','doy','hour','minute','second']
    null_value = np.nan

    # initialization
    def __init__(self, year=null_value,month=null_value,
                 day=null_value,doy=null_value,
                 hour=null_value,minute=null_value,second=null_value):
        self.data = {}
        self.data['year'] = year
        self.data['month'] = month
        self.data['day'] = day
        self.data['doy'] = doy
        self.data['hour'] = hour
        self.data['minute'] = minute
        self.data['second'] = second



##################################################
# Define some constants

# Place holder

# variable for individual-level output
individual_vars = ['year','DBH','HITE','CROWN_RADIUS','PFT','PATCH_AGE','X_COOR','Y_COOR']

# Here we set arrays to record variables that need different scaling method
# when scaling from cohort level to PFT/size-average/ecosystem level

# first variables that only need to be scaled by patch area
var_scale_a = ['LAI','MMEAN_LAI','NPLANT']

# second variables need area and nplant
var_scale_an = ['AGB','BA','BLEAF',
                'LEAF_WATER','LEAF_WATER_INT','WOOD_WATER','WOOD_WATER_INT',
                'MMEAN_LEAF_WATER','MMEAN_LEAF_WATER_INT',
                'MMEAN_WOOD_WATER','MMEAN_WOOD_WATER_INT',
                'DMEAN_LEAF_WATER','DMEAN_LEAF_WATER_INT',
                'DMEAN_WOOD_WATER','DMEAN_WOOD_WATER_INT',
                'FMEAN_LEAF_WATER','FMEAN_LEAF_WATER_INT',
                'FMEAN_WOOD_WATER','FMEAN_WOOD_WATER_INT',
                'MMEAN_GPP']

# third variables need normalized by area * nplant
var_norm_an = ['LEAF_PSI','DMAX_LEAF_PSI','DMIN_LEAF_PSI',
               'WOOD_PSI','DMAX_WOOD_PSI','DMIN_WOOD_PSI',
               'FMEAN_LEAF_PSI','FMEAN_WOOD_PSI',
               'GPP','TRANSP',
               'DMEAN_LINT_CO2',
               'MMEAN_DMAX_LEAF_PSI','MMEAN_DMIN_LEAF_PSI',
               'MMEAN_DMAX_WOOD_PSI','MMEAN_DMIN_WOOD_PSI',
               'FMEAN_LEAF_TEMP','MMEAN_LEAF_TEMP',
               'MMEAN_TRANSP','MMEAN_LINT_CO2',
               'DDBH_DT','DBA_DT','HITE',
               'MORT_RATE','MMEAN_MORT_RATE']
#------------------------------------------------------

# Here we set arrays to record variables that do not need to followed by _CO
var_noco = ['DBH','PFT','NPLANT','DDBH_DT','DBA_DT','HITE','BLEAF','BDEAD','BROOT']

# Here we set arrays to record variables that need to be summed to get cohort level results
var_cosum = ['MORT_RATE','MMEAN_MORT_RATE']

# default size list to use
dbh_size_list = ('D',[0,10,20,30,50,80])
hite_size_list = ('H',[0,1.5,5,10,20,30])
dbh_size_list_fine = ('D',np.arange(0.,200.+5.,10.))
hite_size_list_fine = ('H',np.arange(0.,50.+1.,5.))
##################################################



##################################################
##################################################
# Low level extracting functions
##################################################
def extract_avg(
     h5in : 'handle for .h5 file'
    ,output_dict : 'dictionary to store output data'
    ,voi_avg : 'variables of interests at ecosystem level'
    ,output_idx : 'index for the output data'):
    '''
        Read ecosystem level average output
    '''
    # we don't need to read additional information
    #AREA    = np.array(h5in['AREA'])
    #PACO_ID = np.array(h5in['PACO_ID'])
    #PACO_N  = np.array(h5in['PACO_N'])
    #NPLANT  = np.array(h5in['NPLANT'])

    ################################################
    for var_name in voi_avg:
        if var_name.split('_')[-1] == 'PY':
            # this is polygon level variable, usually 1-D or 3-D (size-PFT-polygon)
            # just take the sum
            output_dict[var_name][output_idx] = np.nansum(h5in['{:s}'.format(var_name)][:])
        else:
            #
            print('''Only _PY output vars can be processed for now. Please add
                  the format for {:s} in the code.'''.format(var_name))

    return

def extract_polygon(
     h5in : 'handle for .h5 file'
    ,output_dict : 'dictionary to store output data'
    ,voi : 'variables of interests at ecosystem level'
    ):
    '''
        Read ecosystem level average output
    '''

    ################################################
    for var_name in voi:
        if var_name.split('_')[-1] == 'PY':
            # this is polygon level variable, usually 1-D or 3-D (size-PFT-polygon)
            # just take the sum
            output_dict[var_name].append(np.nansum(h5in['{:s}'.format(var_name)][:]))
        else:
            #
            print('''Only _PY output vars can be processed for now. Please add
                  the format for {:s} in the code.'''.format(var_name))

    return
# TODO: use extract_polygon and extract_cohort consistently in high-level functions
def extract_cohort(
     h5in : 'handle for .h5 file'
    ,output_dict : 'dictionary to store output data'
    ,voi : 'variables of interests'):
    '''
        Read ecosystem level average output
    '''
    # we don't need to read additional information

    ################################################
    for var_name in voi:
        if var_name in var_noco or var_name.split('_')[-1] == 'CO':
            # this is cohort level variable 
            if var_name in ['FMEAN_LIGHT_CO','DMEAN_LIGHT_CO','MMEAN_LIGHT_CO']:
                # need to combine ATM_RSHORT_PY and LIGHT_LEVEL_CO
                time_scale = var_name.split('_')[0]
                rshort_max = np.array(h5in[time_scale + '_ATM_RSHORT_PY']).ravel()
                output_data = np.array(h5in[time_scale + '_LIGHT_LEVEL_CO'][:])
                output_dict[var_name] += (output_data * rshort_max).ravel().tolist()
            else:
                output_data = np.array(h5in[var_name][:])
                output_dict[var_name] += output_data.ravel().tolist()

        else:
            #
            print('''Only cohort level output vars can be processed for now. Please add
                  the format for {:s} in the code.'''.format(var_name))

    return

def extract_qmean(
     h5in : 'handle for .h5 file'
    ,output_dict : 'dictionary to store output data'
    ,voi : 'variables of interests'):
    '''
        Read ecosystem level average output
    '''
    # we don't need to read additional information

    ################################################
    for var_name in voi:
        if var_name.split('_')[0] != 'QMEAN':
            print('''Only include QMEAN variables in extract_qmean, yours is {:s}
                  '''.format(var_name))
            return -1

        if var_name.split('_')[-1] == 'PY':
            # this is polygon level variable,
            # ravel the dimension and append to the corresponding column
            output_dict[var_name] += np.ravel(h5in[var_name][:]).tolist()
        elif var_name.split('_')[-1] == 'CO':
            # this is cohort level variable
            # loop over each cohort to append the data

            if var_name == 'QMEAN_LIGHT_CO':
                # need to combine QMEAN_ATM_RSHORT_PY and QMEAN_LIGHT_LEVEL_CO
                rshort_max = np.array(h5in['QMEAN_ATM_RSHORT_PY']).ravel()

                output_data = np.array(h5in['QMEAN_LIGHT_LEVEL_CO'])
                for ico in np.arange(output_data.shape[0]):
                    output_dict[var_name] += (output_data[ico,:] * rshort_max).ravel().tolist()
            else:
                output_data = np.array(h5in[var_name])
                for ico in np.arange(output_data.shape[0]):
                    output_dict[var_name] += output_data[ico,:].ravel().tolist()

    return

def extract_soil(
     h5in : 'handle for .h5 file'
    ,output_dict : 'dictionary to store output data'
    ,voi_avg : 'variables of interests at ecosystem level'
    ,output_idx : 'index for the output data'
    ,soil_layers : 'index for the soil layers to extract, default to be top layer' = [-1]
                ):
    '''
        Read ecosystem level average output
    '''
    # we don't need to read additional information
    #AREA    = np.array(h5in['AREA'])
    #PACO_ID = np.array(h5in['PACO_ID'])
    #PACO_N  = np.array(h5in['PACO_N'])
    #NPLANT  = np.array(h5in['NPLANT'])

    ################################################
    for var_name in voi_avg:
        if var_name.split('_')[-1] == 'PY':
            # this is polygon level variable, usually 1-D or 3-D (size-PFT-polygon)
            # just take the sum
            output_dict[var_name][output_idx] = np.nanmean(
                h5in['{:s}'.format(var_name)][:,soil_layers])
        else:
            #
            print('''Only _PY output vars can be processed for now. Please add
                  the format for {:s} in the code.'''.format(var_name))

    return

def extract_pft(
     h5in : 'handle for .h5 file'
    ,output_dict : 'dictionary to store output data'
    ,voi_avg_pft : 'variables of interests per pft'
    ,pft_list : 'List of PFTs to include'
    ,output_idx : 'index for the output data'):

    AREA    = np.array(h5in['AREA'])
    PACO_ID = np.array(h5in['PACO_ID'])
    PACO_N  = np.array(h5in['PACO_N'])
    NPLANT  = np.array(h5in['NPLANT'])
    PFT     = np.array(h5in['PFT'])

    ###############################################
    # generate arrays of masks for Patch and PFT
    cohort_mask = []

    for ipa in np.arange(len(PACO_ID)):
        cohort_mask.append((np.arange(len(PFT)) >= PACO_ID[ipa]-1) &
                           (np.arange(len(PFT)) < PACO_ID[ipa]+PACO_N[ipa]-1))
    pft_mask = []
    # we also need the total nplant for normalization later
    total_nplant_pft = []
    total_nplant = 0.
    for ipft, pft in enumerate(pft_list):
        pft_mask.append(PFT == pft)
        patch_nplant = 0.
        for ipa in np.arange(len(PACO_ID)):
            patch_nplant += \
                np.sum(NPLANT[cohort_mask[ipa] & pft_mask[ipft]] * AREA[ipa])

        total_nplant_pft.append(patch_nplant)
        total_nplant += patch_nplant

    ################################################

    ################################################
    for var in voi_avg_pft:

        # read the raw data
        if var in var_noco:
            tmp_data = np.array(h5in['{:s}'.format(var)])
        else:
            tmp_data = np.array(h5in['{:s}_CO'.format(var)])

        if var in var_cosum:
            tmp_data = np.sum(tmp_data,axis=1)


        # loop over PFTs
        for ipft,pft in enumerate(pft_list):
            var_name = '{:s}_PFT_{:d}'.format(var,pft)
            if np.sum(pft_mask[ipft]) == 0:
                # this PFT does not exist
                output_dict[var_name][output_idx] = np.nan
                continue
                
            for ipa in np.arange(len(PACO_ID)):

                # consider scaling or normalization
                if var in var_scale_a:
                    # scale by area only
                    data_scaler = AREA[ipa]
                elif var in var_scale_an:
                    # scale by NPLANT and AREA
                    data_scaler = NPLANT[cohort_mask[ipa] & pft_mask[ipft]]\
                                * AREA[ipa]
                elif var in var_norm_an:
                    # normalized by NPLANT * AREA
                    data_scaler = NPLANT[cohort_mask[ipa] & pft_mask[ipft]]\
                                * AREA[ipa] / total_nplant_pft[ipft]
                    
                output_dict[var_name][output_idx] += \
                    np.nansum(tmp_data[cohort_mask[ipa] & pft_mask[ipft]] *
                              data_scaler)

    return

def extract_size(
     h5in           : 'handle for .h5 file'
    ,output_dict    : 'dictionary to store output data'
    ,voi_size       : 'profile variables of interests'
    ,size_list      : 'List of size classes to use' 
    ,output_idx     : 'index for the output data'):

    AREA    = np.array(h5in['AREA'])
    PACO_ID = np.array(h5in['PACO_ID'])
    PACO_N  = np.array(h5in['PACO_N'])
    NPLANT  = np.array(h5in['NPLANT'])
    if (size_list[0] == 'H'):
        # use height to separate size classes
        SIZE     = np.array(h5in['HITE'])
    elif (size_list[0] == 'D'):
        # use DBH to separate size classes
        SIZE     = np.array(h5in['DBH'])
    elif (size_list[0] == 'LAI'):
        # Use cumulative LAI from top of canopy as size classes
        # first read LAI
        LAI = np.array(h5in['LAI_CO'])
        SIZE = np.zeros_like(LAI)
        # loop over each patch to calculate the cumulative LAI for each cohort
        for ipa, paco_start in enumerate(PACO_ID):
            pa_mask = ((np.arange(len(SIZE)) >= PACO_ID[ipa]-1) &
                       (np.arange(len(SIZE)) < PACO_ID[ipa]+PACO_N[ipa]-1)
                      )
            SIZE[pa_mask] = np.cumsum(LAI[pa_mask])
    elif (size_list[0] == 'HTOP'):
        # Use hite difference from top of the canopy as size classes
        # first read HITE
        HITE = np.array(h5in['HITE'])
        SIZE = np.zeros_like(HITE)
        # loop over each patch to calculate the hite difference from the tallest cohort
        for ipa, paco_start in enumerate(PACO_ID):
            pa_mask = ((np.arange(len(SIZE)) >= PACO_ID[ipa]-1) &
                       (np.arange(len(SIZE)) < PACO_ID[ipa]+PACO_N[ipa]-1)
                      )
            SIZE[pa_mask] = (HITE[pa_mask][0] - HITE[pa_mask])

    else:
        print('''Error! Can not recognize size class identifier. Your identifier
              is set as {:s}. Only H or D is accepted'''.format(size_list[0]))
        return

    ###############################################
    # generate arrays of masks for Patch and size classes
    cohort_mask = []

    for ipa in np.arange(len(PACO_ID)):
        cohort_mask.append((np.arange(len(SIZE)) >= PACO_ID[ipa]-1) &
                           (np.arange(len(SIZE)) < PACO_ID[ipa]+PACO_N[ipa]-1))

    size_mask = []
    # we also need the total nplant for normalization later
    total_nplant_size = []
    total_nplant = 0.
    for isize, size_edge in enumerate(size_list[1]):
        if (isize + 1) < len(size_list[1]):
            # not the last one
            data_mask = (SIZE >= size_edge) & (SIZE < size_list[1][isize+1])
        else:
            # last one
            data_mask = (SIZE >= size_edge)

        size_mask.append(data_mask)
        patch_nplant = 0.
        for ipa in np.arange(len(PACO_ID)):
            patch_nplant += \
                np.sum(NPLANT[cohort_mask[ipa] & size_mask[isize]] * AREA[ipa])

        total_nplant_size.append(patch_nplant)
        total_nplant += patch_nplant

    ################################################

    ################################################
    for var in voi_size:
        # read raw data
        if var in var_noco:
            tmp_data = np.array(h5in['{:s}'.format(var)])
        else:
            tmp_data = np.array(h5in['{:s}_CO'.format(var)])

        if var in var_cosum:
            tmp_data = np.sum(tmp_data,axis=1)
#        elif var in ['MMEAN_iWUE']:
#            # we need to derive the iWUE from GPP, LEAF_RESP, LAI and GSW
#            gpp = np.array(h5in['MMEAN_GPP_CO'])
#            leaf_resp = np.array(h5in['MMEAN_LEAF_RESP_CO'])
#            gsw = np.array(h5in['MMEAN_LEAF_GSW_CO']) 
#            lai = np.array(h5in['LAI_CO'])
#            nplant = np.array(h5in['NPLANT'])
#            tmp_data = ( (gpp - leaf_resp) # A_net kgC/pl/yr
#                       * nplant / lai      # convert to kgC/m2leaf/yr
#                       * 1000. / 12. * 1e6 # convert to umolC/m2leaf/yr
#                       / (gsw * 1000. / 18. * 86400. * 365.) # molH2O/m2/yr
#                       )
#            # the resulting value is umolC/molH2O
#            tmp_data[gsw == 0.] = np.nan

        # loop over size classes
        for isize,size_edge in enumerate(size_list[1]):

            # special cases for WOOD_WATER and WOOD_WATER_INT since we want to distribute it evenly
            # across the height profile
            if (size_list[0] == 'HTOP') and \
               (var in ['WOOD_WATER','WOOD_WATER_INT',
                       'FMEAN_WOOD_WATER','FMEAN_WOOD_WATER_INT',
                       'DMEAN_WOOD_WATER','DMEAN_WOOD_WATER_INT',
                       'MMEAN_WOOD_WATER','MMEAN_WOOD_WATER_INT']):

                var_name = '{:s}_{:s}_{:d}'.format(var,size_list[0],isize)

                # these variables need to be weighted by AREA, NPLANT and relative fraction of
                # plant heights that are in the size group
                # loop over patches
                for ipa in np.arange(len(PACO_ID)):
                    hmax = HITE[cohort_mask[ipa]][0]

                    #  not the last one
                    if isize+1 < len(size_list[1]):
                        h_bot = hmax - size_list[1][isize+1]
                    else:
                        # last one
                        h_bot = np.minimum(0.,hmax-size_list[1][isize])

                    if h_bot < 0.:
                        # unrealistic values
                        # just continue to next patch
                        continue


                    # otherwise, calculate hite_scaler
                    # first calculate the total height of this size bin
                    # effectively equal to size_list[1][isize+1] -
                    # size_list[0][isize] but with the control of hmax
                    h_bin = np.maximum(0.,hmax - size_list[1][isize]) - h_bot

                    # hite_scaler equal to the plant height that falls into this size_bin
                    # which equals to max(0,min(h_bin,hite-h_bot))
                    hite_scaler = (
                        np.maximum(0.,np.minimum(h_bin,
                        HITE[cohort_mask[ipa]]-h_bot)) / HITE[cohort_mask[ipa]]
                    )

                    # consider scaling or normalization
                    data_scaler = NPLANT[cohort_mask[ipa]]\
                                * AREA[ipa] * hite_scaler
                
                    output_dict[var_name][output_idx] += \
                        np.nansum(tmp_data[cohort_mask[ipa]] * data_scaler)

                # skip the default processing
                continue

            var_name = '{:s}_{:s}_{:d}'.format(var,size_list[0],isize)

            if np.sum(size_mask[isize]) == 0:
                # No cohorts in this size class
                output_dict[var_name][output_idx] = np.nan
                continue

            # loop over patches
            for ipa in np.arange(len(PACO_ID)):
                # consider scaling or normalization
                if var in var_scale_a:
                    # scale by area only
                    data_scaler = AREA[ipa]
                elif var in var_scale_an:
                    # scale by NPLANT and AREA
                    data_scaler = NPLANT[cohort_mask[ipa] & size_mask[isize]]\
                                * AREA[ipa]
                elif var in var_norm_an:
                    # normalized by NPLANT * AREA
                    data_scaler = NPLANT[cohort_mask[ipa] & size_mask[isize]]\
                                * AREA[ipa] / total_nplant_size[isize]
                
                #print(var_name,
                #      output_idx,
                #      np.nansum(tmp_data[cohort_mask[ipa] & size_mask[isize]] *
                #               data_scaler),
                #      output_dict[var_name][output_idx])
                output_dict[var_name][output_idx] += \
                    np.nansum(tmp_data[cohort_mask[ipa] & size_mask[isize]] *
                              data_scaler)

    return


def extract_pft_size(
     h5in           : 'handle for .h5 file'
    ,output_dict    : 'dictionary to store output data'
    ,voi_pft_size   : 'profile variables of interests per pft'
    ,pft_list       : 'List of PFTs to include'
    ,size_list      : 'List of size classes to use' 
    ,output_idx     : 'index for the output data'):

    AREA    = np.array(h5in['AREA'])
    PACO_ID = np.array(h5in['PACO_ID'])
    PACO_N  = np.array(h5in['PACO_N'])
    NPLANT  = np.array(h5in['NPLANT'])
    PFT     = np.array(h5in['PFT'])

    if (size_list[0] == 'H'):
        # use height to separate size classes
        SIZE     = np.array(h5in['HITE'])
    elif (size_list[0] == 'D'):
        # use DBH to separate size classes
        SIZE     = np.array(h5in['DBH'])
    elif (size_list[0] == 'LAI'):
        # Use cumulative LAI from top of canopy as size classes
        # first read LAI
        LAI = np.array(h5in['LAI_CO'])
        SIZE = np.zeros_like(LAI)
        # loop over each patch to calculate the cumulative LAI for each cohort
        for ipa, paco_start in enumerate(PACO_ID):
            pa_mask = ((np.arange(len(SIZE)) >= PACO_ID[ipa]-1) &
                       (np.arange(len(SIZE)) < PACO_ID[ipa]+PACO_N[ipa]-1)
                      )
            SIZE[pa_mask] = np.cumsum(LAI[pa_mask])

    else:
        print('''Error! Can not recognize size class identifier. Your identifier
              is set as {:s}. Only H or D is accepted'''.format(size_list[0]))
        return

    ###############################################
    # generate arrays of masks for Patch, PFT, and size
    cohort_mask = []

    for ipa in np.arange(len(PACO_ID)):
        cohort_mask.append((np.arange(len(PFT)) >= PACO_ID[ipa]-1) &
                           (np.arange(len(PFT)) < PACO_ID[ipa]+PACO_N[ipa]-1))
    pft_mask = []
    for ipft, pft in enumerate(pft_list):
        pft_mask.append(PFT == pft)

    size_mask = []
    for isize, size_edge in enumerate(size_list[1]):
        if (isize + 1) < len(size_list[1]):
            # not the last one
            data_mask = (SIZE >= size_edge) & (SIZE < size_list[1][isize+1])
        else:
            # last one
            data_mask = (SIZE >= size_edge)

        size_mask.append(data_mask)


    # we also need the total nplant for normalization later
    total_nplant_pft_size = np.zeros((len(pft_list),len(size_list[1])))
    for ipft, pft in enumerate(pft_list):
        for isize, size_edge in enumerate(size_list[1]):
            patch_nplant = 0.
            for ipa in np.arange(len(PACO_ID)):
                patch_nplant += \
                    np.sum(NPLANT[cohort_mask[ipa] & pft_mask[ipft] &
                                  size_mask[isize]] * AREA[ipa])

            total_nplant_pft_size[ipft,isize] = patch_nplant

    ################################################

    ################################################
    for var in voi_pft_size:
        if var in var_noco:
            tmp_data = np.array(h5in['{:s}'.format(var)])
        else:
            tmp_data = np.array(h5in['{:s}_CO'.format(var)])

        if var in var_cosum:
            tmp_data = np.sum(tmp_data,axis=1)
#         elif var in ['MMEAN_iWUE']:
#            # we need to derive the iWUE from GPP, LEAF_RESP, LAI and GSW
#            gpp = np.array(h5in['MMEAN_GPP_CO'])
#            leaf_resp = np.array(h5in['MMEAN_LEAF_RESP_CO'])
#            gsw = np.array(h5in['MMEAN_LEAF_GSW_CO']) 
#            lai = np.array(h5in['LAI_CO'])
#            nplant = np.array(h5in['NPLANT'])
#            tmp_data = ( (gpp - leaf_resp) # A_net kgC/pl/yr
#                       * nplant / lai      # convert to kgC/m2leaf/yr
#                       * 1000. / 12. * 1e6 # convert to umolC/m2leaf/yr
#                       / (gsw * 1000. / 18. * 86400. * 365.) # molH2O/m2/yr
#                       )
#            # the resulting value is umolC/molH2O
#            tmp_data[gsw == 0.] = np.nan

        # loop over PFTs
        for ipft,pft in enumerate(pft_list):
            # loop over sizes
            for isize, size_edge in enumerate(size_list[1]):
                var_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,size_list[0],isize)

                if np.sum(size_mask[isize] & pft_mask[ipft]) == 0:
                    # No cohorts in this pft and size class
                    output_dict[var_name][output_idx] = np.nan
                    continue

                # loop over patches
                for ipa in np.arange(len(PACO_ID)):
                    # consider scaling or normalization
                    if var in var_scale_a:
                        # scale by area only
                        data_scaler = AREA[ipa]
                    elif var in var_scale_an:
                        # scale by NPLANT and AREA
                        data_scaler = NPLANT[cohort_mask[ipa] & pft_mask[ipft]
                                            & size_mask[isize]] * AREA[ipa]
                    elif var in var_norm_an:
                        # normalized by NPLANT and AREA
                        data_scaler = NPLANT[cohort_mask[ipa] & pft_mask[ipft]
                                            & size_mask[isize]] * AREA[ipa] \
                                    / total_nplant_pft_size[ipft,isize]
                    else:
                        print(var)
                        print('ERROR! do not know how to normalize the variable')
            
                    output_dict[var_name][output_idx] += \
                        np.nansum(tmp_data[cohort_mask[ipa] & pft_mask[ipft] &
                                          size_mask[isize]] * data_scaler)

    return


##################################################


##################################################
# TODO: this function uses a new structure to organize processed data
# should be extended to all the previous functions in the future
def extract_pft_size_age(
     h5in           : 'handle for .h5 file'
    ,output_dict    : 'dictionary to store output data'
    ,voi            : 'profile variables of interests per pft'
    ,pft_list       : 'List of PFTs to include'
    ,size_list      : 'List of size classes to use'
    ,time_stamp     : 'An EDTime class that contains all the time information'):

    AREA    = np.array(h5in['AREA'])
    AGE     = np.array(h5in['AGE'])
    PACO_ID = np.array(h5in['PACO_ID'])
    PACO_N  = np.array(h5in['PACO_N'])
    NPLANT  = np.array(h5in['NPLANT'])
    PFT     = np.array(h5in['PFT'])

    if (size_list[0] == 'H'):
        # use height to separate size classes
        SIZE     = np.array(h5in['HITE'])
    elif (size_list[0] == 'D'):
        # use DBH to separate size classes
        SIZE     = np.array(h5in['DBH'])
    elif (size_list[0] == 'LAI'):
        # Use cumulative LAI from top of canopy as size classes
        # first read LAI
        LAI = np.array(h5in['LAI_CO'])
        SIZE = np.zeros_like(LAI)
        # loop over each patch to calculate the cumulative LAI for each cohort
        for ipa, paco_start in enumerate(PACO_ID):
            pa_mask = ((np.arange(len(SIZE)) >= PACO_ID[ipa]-1) &
                       (np.arange(len(SIZE)) < PACO_ID[ipa]+PACO_N[ipa]-1)
                      )
            SIZE[pa_mask] = np.cumsum(LAI[pa_mask])

    else:
        print('''Error! Can not recognize size class identifier. Your identifier
              is set as {:s}. Only H or D is accepted'''.format(size_list[0]))
        return

    ###############################################
    # generate arrays of masks for Patch, PFT, and size
    cohort_mask = []

    for ipa in np.arange(len(PACO_ID)):
        cohort_mask.append((np.arange(len(PFT)) >= PACO_ID[ipa]-1) &
                           (np.arange(len(PFT)) < PACO_ID[ipa]+PACO_N[ipa]-1))
    pft_mask = []
    for ipft, pft in enumerate(pft_list):
        pft_mask.append(PFT == pft)

    size_mask = []
    for isize, size_edge in enumerate(size_list[1]):
        if (isize + 1) < len(size_list[1]):
            # not the last one
            data_mask = (SIZE >= size_edge) & (SIZE < size_list[1][isize+1])
        else:
            # last one
            data_mask = (SIZE >= size_edge)

        size_mask.append(data_mask)


    # we also need the total nplant for normalization later
    total_nplant_pft_size = np.zeros((len(pft_list),len(size_list[1])))
    for ipft, pft in enumerate(pft_list):
        for isize, size_edge in enumerate(size_list[1]):
            patch_nplant = 0.
            for ipa in np.arange(len(PACO_ID)):
                patch_nplant += \
                    np.sum(NPLANT[cohort_mask[ipa] & pft_mask[ipft] &
                                  size_mask[isize]] * AREA[ipa])

            total_nplant_pft_size[ipft,isize] = patch_nplant

    ################################################

    ################################################
    # loop over pft, size and patch
    for ipft,pft in enumerate(pft_list):
        # loop over sizes
        for isize, size_edge in enumerate(size_list[1]):
            # loop over patches
            for ipa in np.arange(len(PACO_ID)):
                # record time, PFT, size, and patch_age information
                for var_name in EDTime.time_names:
                    output_dict[var_name].append(time_stamp.data[var_name])
                output_dict['PFT'].append(pft)
                output_dict['SIZE'].append('{:s}_{:d}'.format(size_list[0],isize))
                output_dict['PATCH_AGE'].append(AGE[ipa])
                output_dict['PATCH_AREA'].append(AREA[ipa])

                # Record variables
                for var_name in voi:
                    # no cohorts in this 
                    if np.sum(size_mask[isize] & pft_mask[ipft] & cohort_mask[ipa]) == 0:
                        output_dict[var_name].append(np.nan)
                        continue
                    # else we have some cohorts
                    # Process the data
                    if var_name in var_noco:
                        tmp_data = np.array(h5in['{:s}'.format(var_name)])
                    else:
                        tmp_data = np.array(h5in['{:s}_CO'.format(var_name)])

                    if var_name in var_cosum:
                        tmp_data = np.sum(tmp_data,axis=1)

                        
                    # since we are include age effect
                    # do not normalize by area
                    # scale by area only
                    # consider scaling or normalization
                    if var_name in var_scale_a:
                        data_scaler = 1.
                    elif var_name in var_scale_an:
                        # scale by NPLANT 
                        data_scaler = NPLANT[
                            cohort_mask[ipa]    &
                            pft_mask[ipft]      &
                            size_mask[isize]]
                    elif var_name in var_norm_an:
                        # normalized by NPLANT
                        data_scaler = (
                            NPLANT[cohort_mask[ipa] &
                                   pft_mask[ipft]   &
                                   size_mask[isize]] 
                          / np.nansum(
                            NPLANT[cohort_mask[ipa] &
                                   pft_mask[ipft]   &
                                   size_mask[isize]] 
                                     )
                                       )
                    else:
                        print(var_name)
                        print('ERROR! do not know how to normalize the variable')
            
                    output_dict[var_name].append(
                        np.nansum(tmp_data[cohort_mask[ipa] & pft_mask[ipft] &
                                          size_mask[isize]] * data_scaler))

    return


##################################################


def extract_individual_for_plot(
     h5in           : 'handle for .h5 file'
    ,output_dict    : 'dictionary to store output data'
    ,pft_list       : 'List of PFTs to include'
    ,year           : 'Year of the plot information'
):
    '''
        Extract individual level information for plotting
        Assume a 1 Ha area to plot (100m by 100m).
        Devide the area into 16 25by25m patches and assign the patches according to patch area
        Only include cohorts >= 1.5 m
        NPLANT < 1 would be scaled up to 1
        include
        DBH, HITE, CROWN_RADIUS, NPLANT, PFT, X_data, Y_data (in meters)
    '''

    plot_x = 100
    plot_y = 100
    plot_size = plot_x * plot_y  
    patch_x = 20 # m
    patch_y = 20 # m
    total_patch_num = plot_size / (patch_x * patch_y)

    # total 25 patches

    # first read data
    AREA    = np.array(h5in['AREA'])
    AGE     = np.array(h5in['AGE'])
    PACO_ID = np.array(h5in['PACO_ID'])
    PACO_N  = np.array(h5in['PACO_N'])
    NPLANT  = np.array(h5in['NPLANT'])
    PFT     = np.array(h5in['PFT'])
    DBH     = np.array(h5in['DBH'])
    HITE    = np.array(h5in['HITE'])
    CROWN_RADIUS    = np.array(h5in['CROWN_AREA_CO']) # m2/m2
    CROWN_RADIUS = (CROWN_RADIUS / NPLANT / np.pi) ** (0.5)  # m

    # Scale NPLANT with AREA
    for ipa in np.arange(len(PACO_ID)):
        # otherwise, extract all the cohorts from this patch
        cohort_mask = (
                        (np.arange(len(PFT)) >= PACO_ID[ipa]-1) &
                        (np.arange(len(PFT)) < PACO_ID[ipa]+PACO_N[ipa]-1))

        NPLANT[cohort_mask] *= AREA[ipa]

    # get the number of patches to plot for each actual patch in the simulations
    plot_patch_nums = (np.around(np.cumsum(AREA) * total_patch_num) - 
                       np.around(np.cumsum(np.concatenate(([0],AREA[0:-1]))) * total_patch_num))

    # get the number of platns to plot for each cohort
    plot_nplant_nums = np.maximum(1.,np.around(NPLANT * plot_size))
    plot_nplant_nums[HITE < 1.0] = 0. # do not plot cohorts smaller thant 1.0m

    # exclude PFTs that are not included
    pft_mask = PFT < 0 # always false
    for ipft, pft in enumerate(pft_list):
        pft_mask |= (PFT == pft)

    plot_nplant_nums[~pft_mask] = 0.  # PFTs not included in the analysis

    # get total number of stems to plot
    total_stems = 0
    for ipa in np.arange(len(PACO_ID)):
        # skip the patch if the plot_patch_nums is zero
        if plot_patch_nums[ipa] == 0:
            continue


        # otherwise, extract all the cohorts from this patch
        cohort_ids = np.arange(len(PFT))[
                            (np.arange(len(PFT)) >= PACO_ID[ipa]-1) &
                            (np.arange(len(PFT)) < PACO_ID[ipa]+PACO_N[ipa]-1)]
        total_stems += int(np.sum(plot_nplant_nums[cohort_ids]))

    # Generate X and Ydata
    # The idea is that a cohort with the same Cohort Number and Same PFT and Same patch number will
    # always has the same X and Y data

    # we use a two layer scheme to generate X and Y
    # X = X_within_patch + Patch_X_start
    # X_within_patch depends on cohort_number and PFT
    # Patch_X_start depends on patch_number

    # we loop over patches

    # generate a randome state class implementation
    rs = np.random.RandomState(seed=0)

    for ipa in np.arange(len(PACO_ID)):
        # skip the patch if the plot_patch_nums is zero
        if plot_patch_nums[ipa] == 0:
            continue


        # otherwise, save the start patch number and end patch number
        if ipa == 0:
            patch_num_start = 0
        else:
            patch_num_start = np.sum(plot_patch_nums[0:ipa])

        patch_num_end = patch_num_start + plot_patch_nums[ipa] - 1

        # calculate the maximum individuals allowed
        ind_max = plot_patch_nums[ipa] * 100
        
        #extract all the cohorts from this patch
        cohort_ids = np.arange(len(PFT))[
                            (np.arange(len(PFT)) >= PACO_ID[ipa]-1) &
                            (np.arange(len(PFT)) < PACO_ID[ipa]+PACO_N[ipa]-1)]

        # loop over cohorts
        current_ind = 0
        for i, ico in enumerate(cohort_ids):
            # first we need to check how many individuals we need to plot for this cohort
            plot_nums = plot_nplant_nums[ico]

            if plot_nums == 0:
                # no need to plot this cohort
                continue

            if current_ind >= ind_max:
                break
            
            # loop over plot_nums
            for ind_id in np.arange(plot_nums):
                # first random chose a patch using the ind_id and ico as a seed
                rs.seed(int(ind_id + 1000 * ico))
                patch_num = int(np.floor(
                    (plot_patch_nums[ipa] * rs.random_sample() + patch_num_start)))

                # random chose X & Y
                rs.seed(int(ind_id + 100000 * ico + PFT[ico] * 100))
                x_data = rs.random_sample() * patch_x + patch_num // (plot_x / patch_x) * patch_x
                y_data = rs.random_sample() * patch_y + patch_num % (plot_x / patch_x) * patch_y

                # save the data
                output_dict['DBH'].append(DBH[ico])
                output_dict['HITE'].append(HITE[ico])
                output_dict['CROWN_RADIUS'].append(CROWN_RADIUS[ico])
                output_dict['PATCH_AGE'].append(AGE[ipa])
                output_dict['PFT'].append(PFT[ico])
                output_dict['X_COOR'].append(x_data)
                output_dict['Y_COOR'].append(y_data)
                output_dict['year'].append(year)

                current_ind += 1
            

def extract_demo_rates_annual(
     data_pf      : 'path and prefix of the ED2 data'
    ,output_dict  : 'output dictionary'
    ,census_yeara   : 'start year' 
    ,census_yearz   : 'end year'
    ,pft_list : 'PFT to inclue'
    ,time_idx : 'index of the array'
    ,size_list : 'size_list' = dbh_size_list
    ):
    '''
       This function will use the BASAL_AREA_SI, BASAL_AERA_GROWTH, BASAL_AREA_MORT
    '''
    # Necessary stamps, time, cohort_id at the last 'census' in the model
    # cohorts that did not survive to the last time point will be given an id
    # of -1
    year_array = np.arange(census_yeara,census_yearz+1)

    # creat a year list and a month list to loop over
    # total months to track
    year_num = len(year_array)
    growth_rate = np.zeros((year_num,len(pft_list),len(size_list[1])))
    mort_rate = np.zeros((year_num,len(pft_list),len(size_list[1])))

    # we also need to get BA_PFT_SIZE
    ba_pft_size_dict = {}
    for var in ['BA']:
        for pft in pft_list:
            for isize, size_edge in enumerate(size_list[1]):
                ba_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(
                    var,pft,size_list[0],isize)] = np.zeros((year_num,))

    #------------------  Loop Over Time   --------------------------------#
    for iyear, year in enumerate(year_array):
        # read data
        data_fn = '{:s}-Y-{:4d}-00-00-000000-g01.h5'.format(
                data_pf,year)

        if not os.path.isfile(data_fn):
            print('{:s} doest not exist!'.format(data_fn))
            # file does not exist
            return -1

        h5in    = h5py.File(data_fn,'r')
        BASAL_AREA_GROWTH = np.array(h5in['BASAL_AREA_GROWTH'])
        BASAL_AREA_MORT = np.array(h5in['BASAL_AREA_MORT'])
        extract_pft_size(h5in,ba_pft_size_dict,
                         ['BA'],pft_list,size_list,iyear)
        h5in.close()

        # get fractional growth and mortality
        for ipft, pft in enumerate(pft_list):
            # loop over model default DBH classes
            for isize, size_edge in enumerate(np.arange(0,100+1,10)):
                # determine the actual size bin
                index_array = np.where(size_list[1] >= size_edge)[0]
                if len(index_array) == 0:
                    output_isize = len(size_list[1]) - 1
                else:
                    output_isize = index_array[0]
                growth_rate[iyear,ipft,output_isize] += (
                    BASAL_AREA_GROWTH[0,isize,pft-1])
                mort_rate[iyear,ipft,output_isize] += (
                    BASAL_AREA_MORT[0,isize,pft-1])
                

    # add the data into output
    for ipft,pft in enumerate(pft_list):
        for isize, size_edge in enumerate(size_list[1]):
            ba_array = ba_pft_size_dict['BA_PFT_{:d}_{:s}_{:d}'.format(
                    pft,size_list[0],isize)]
            growth_array = growth_rate[:,ipft,isize] / ba_array
            growth_array[ba_array <= 0] = np.nan
            
            mort_array = mort_rate[:,ipft,isize] / ba_array
            mort_array[ba_array <= 0] = np.nan


            output_dict['BA_GROWTH_FRAC_PFT_{:d}_{:s}_{:d}'.format(
                        pft,size_list[0],isize)][time_idx] = np.nanmean(
                            growth_array
                )
            output_dict['BA_MORT_FRAC_PFT_{:d}_{:s}_{:d}'.format(
                        pft,size_list[0],isize)][time_idx] = np.nanmean(
                            mort_array
                        )


    return

##########################


def extract_demo_rates_monthly(
     data_pf      : 'path and prefix of the ED2 data'
    ,output_dict  : 'output dictionary'
    ,census_yeara   : 'start year' , census_montha : 'start month'
    ,census_yearz   : 'end year'   , census_monthz : 'end month'
    ,dbh_min        : 'the smallest tree to census [cm]' = 1.
    ,hite_min       : 'the minimum hite of a cohort [m]' = 0.5
    ):
    '''
       This function will mimic the actual tree census in the fields but for
       cohorts in ED2
       (1) It will read the monthly output to find which cohorts have survived the whole census
       period (how to deal with cohort fusion and cohort split? For now ignore the cohorts that have
       split or fused... This would lead to biases)
       (2) It will calculate the growth rates and moratlity during the census rate for each initial
       cohort
    '''
    # Necessary stamps, time, cohort_id at the last 'census' in the model
    # cohorts that did not survive to the last time point will be given an id
    # of -1
    col_list = ['year','final_cohort_id']
    year_array = np.arange(census_yeara,census_yearz+1)

    # creat a year list and a month list to loop over
    year_list = []
    month_list = []
    for iyear, year in enumerate(year_array):
        if year == census_yeara:
            montha = census_montha
        else:
            montha = 1

        if year == census_yearz:
            monthz = census_monthz
        else:
            monthz = 12

        # loop over months
        for imonth, month in enumerate(np.arange(montha,monthz+1)):
            year_list.append(year)
            month_list.append(month)


    # total months to track
    month_num = len(month_list)

    # first we mark the cohorts to track at the start of the census
    # which requires reading the output from the last month because monthly moutput records the
    # results at the end of the month.
    if census_montha == 1:
        first_month = 12
        first_year = census_yeara -1
    else:
        first_month = census_montha - 1
        first_year = census_yeara

    data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
        data_pf,first_year,first_month)

    if not os.path.isfile(data_fn):
        print('{:s} doest not exist!'.format(data_fn))
        print('Make sure you have at least one year simulation after output_yearz {:d}'.format(output_yearz))
        # file does not exist
        return -1

    h5in    = h5py.File(data_fn,'r')
    DBH_init      = np.array(h5in['DBH'])
    DDBH_DT_init      = np.array(h5in['DDBH_DT'])
    PFT_init      = np.array(h5in['PFT'])
    HITE_init      = np.array(h5in['HITE'])
    NPLANT_init   = np.array(h5in['NPLANT'])
    MORT_RATE_init   = np.array(h5in['MMEAN_MORT_RATE_CO'])
    MORT_RATE_init = np.sum(MORT_RATE_init,axis=1) # sum up all kinds of mortality
    AREA           = np.array(h5in['AREA'])
    PACO_ID        = np.array(h5in['PACO_ID'])
    PACO_N         = np.array(h5in['PACO_N'])
    h5in.close()

    # create an array of patch number for each cohort
    PA_NUM_init   = np.zeros_like(NPLANT_init)
    for ico_pa in np.arange(len(PA_NUM_init)):
        PA_NUM_init[ico_pa] = np.where(PACO_ID <= (ico_pa + 1))[0][-1]

    # loop over patches to modify NPLANT with patch area
    # generate arrays of masks for Patch
    for ipa in np.arange(len(PACO_ID)):
        cohort_mask = ((np.arange(len(NPLANT_init)) >= PACO_ID[ipa]-1) &
                           (np.arange(len(NPLANT_init)) < PACO_ID[ipa]+PACO_N[ipa]-1))
        NPLANT_init[cohort_mask] *= AREA[ipa]

    # only track cohorts bigger than dbh_min
    cohort_num = len(DBH_init)
    cohort_flag = np.arange(len(DBH_init))
    cohort_mask = (DBH_init < dbh_min) | (HITE_init < hite_min)
    cohort_flag[cohort_mask] = -1 # these trees are not tracked

    # we need to create space to record the average mortality and growth rates
    cohort_ddbh_dt_avg = np.zeros_like(DBH_init) # total growth, init as 0
    cohort_survive_rate_avg = np.ones_like(DBH_init) # survive ratio (1 - month), init as 1
    cohort_dbh_current = DBH_init.copy() # this records the current DBH


    #------------------  Loop Over Time   --------------------------------#
    for itime, year, month in zip(np.arange(len(year_list)),year_list,month_list):
        # read data
        data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
                data_pf,year,month)

        if not os.path.isfile(data_fn):
            print('{:s} doest not exist!'.format(data_fn))
            print('Make sure you have at least one year simulation after output_yearz  {:d}'.format(census_yearz))
            # file does not exist
            return -1

        h5in    = h5py.File(data_fn,'r')
        DBH      = np.array(h5in['DBH'])
        DDBH_DT      = np.array(h5in['DDBH_DT'])
        MORT_RATE   = np.array(h5in['MMEAN_MORT_RATE_CO'])
        MORT_RATE = np.sum(MORT_RATE,axis=1) # sum up all kinds of mortality
        PFT      = np.array(h5in['PFT'])
        HITE     = np.array(h5in['HITE'])
        BA       = np.array(h5in['BA_CO'])
        NPLANT   = np.array(h5in['NPLANT'])
        AREA           = np.array(h5in['AREA'])
        PACO_ID        = np.array(h5in['PACO_ID'])
        PACO_N         = np.array(h5in['PACO_N'])
        h5in.close()

        # create an array of patch number for each cohort
        PA_NUM   = np.zeros_like(NPLANT)
        for ico_pa in np.arange(len(PA_NUM)):
            PA_NUM[ico_pa] = np.where(PACO_ID <= (ico_pa + 1))[0][-1]

        # loop over patches to modify NPLANT with patch area
        # generate arrays of masks for Patch
        for ipa in np.arange(len(PACO_ID)):
            cohort_mask = ((np.arange(len(NPLANT)) >= PACO_ID[ipa]-1) &
                               (np.arange(len(NPLANT)) < PACO_ID[ipa]+PACO_N[ipa]-1))
            NPLANT[cohort_mask] *= AREA[ipa]

        # loop over cohort to check whether we can find the match
        # flag the cohorts without match as id = -1 (not tracking)
        for ico, dbh_start in enumerate(cohort_dbh_current):
            if cohort_flag[ico] == -1:
                # skip
                continue

            # find the matching cohort
            dbh_mask = (np.absolute(
                            (dbh_start + DDBH_DT/12.)
                            / DBH - 1.)) < 1e-8
            pft_mask = (PFT == PFT_init[ico])
            pa_mask  = (PA_NUM == PA_NUM_init[ico])

            if (np.sum(dbh_mask & pft_mask & pa_mask) >= 1):
                # find the exact match or multiple matches
                # use the first one
                ico_match = \
                    np.arange(len(DBH))[dbh_mask&pft_mask&pa_mask].tolist()[0]

                #update cohort information
                cohort_ddbh_dt_avg[ico] += DDBH_DT[ico_match]/12.
                cohort_survive_rate_avg[ico] *= np.exp(-MORT_RATE[ico_match]/12.)
                cohort_dbh_current[ico] += DDBH_DT[ico_match]/12.
            else:
                # no exact match
                # For now, just flag this cohort as not tracked
                # This can reduce the number of cohorts to track and create biases
                # but will free us from the cohort fusion and split stuff...
                cohort_flag[ico] = -1


        
    # now we have gone through the whole census
    # normalize growth and mortality
    cohort_ddbh_dt_avg /= (float(month_num) / 12.)  # cm/year
    cohort_mort_rate_avg = np.log(cohort_survive_rate_avg) / (- month_num / 12.) 

    # attach the information to output_dict
    # loop over cohort
    for ico, dbh_init in enumerate(DBH_init):
        if cohort_flag[ico] == -1:
            continue

        output_dict['census_yeara'].append(census_yeara)
        output_dict['census_montha'].append(census_montha)
        output_dict['census_yearz'].append(census_yearz)
        output_dict['census_monthz'].append(census_monthz)
        output_dict['DBH_init'].append(DBH_init[ico])
        output_dict['NPLANT_init'].append(NPLANT_init[ico])
        output_dict['PFT_init'].append(PFT_init[ico])
        output_dict['HITE_init'].append(HITE_init[ico])
        output_dict['DDBH_DT_avg'].append(cohort_ddbh_dt_avg[ico])
        output_dict['MORT_RATE_avg'].append(cohort_mort_rate_avg[ico])
    
    return

##########################











##################################################
## extract a series of daily output
###################################################

##################################################
## extract a series of monthly output
###################################################



##################################################
## extract a series of annual ouput
##################################################



##################################################
##################################################

##################################################
## Take virtual tree cores from ED2 Monthly output
##################################################

