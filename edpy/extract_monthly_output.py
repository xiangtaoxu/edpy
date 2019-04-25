# High-level functions for users

# import modules
import extract_utils as exut
from extract_utils import dbh_size_list, hite_size_list
from extract_utils import dbh_size_list_fine, hite_size_list_fine
from calendar import monthrange
from datetime import datetime
import h5py
import pandas as pd
import numpy as np
import os
import sys

def extract_ed2_monthly(
     data_pf : 'path and prefix of the ED2 data, output file name would be automatically generated'
    ,out_dir : 'output directory'
    ,out_pf  : 'output prefix'
    ,output_yeara   : 'start year',output_montha  : 'start month'
    ,output_yearz   : 'end year' ,output_monthz  : 'end month'  
    ,voi            : 'variable of interests (ecosystem average)' = ['AGB','LAI','BA','NPLANT']
    ,pft_list       : 'List of PFTs to include for pft VOIs' = []
 ):
    '''
        This function aims to extract monthly scale ecosystem diagnostics for plots.
        The default variables to include is
        Polygon level
        Meterological drivers
        GPP, NPP, NEE, ET


        PFT-level data
        AGB, BA, LAI, NPLANT

        # size-level data
        AGB, BA, LAI, NPLANT

        Census time scale
        _demography
        Demography data
        _pft_size
        PFT-size level data
        _individual_plot
        Individual-level data
    '''

    if (len(pft_list) == 0):
        print('No PFT is specified for PFT VOIs')
        return

    # Necessary time stamps
    # time is the normalized year, used for plotting...
    col_list = ['year','month','time']


    # first some polygon level data
    # using extract_avg
    output_met_vars = ['MMEAN_ATM_TEMP_PY','MMEAN_PCPG_PY','MMEAN_ATM_RSHORT_PY',
                       'MMEAN_ATM_VPDEF_PY','MMEAN_ATM_CO2_PY']
    output_flux_vars = ['MMEAN_GPP_PY','MMEAN_NPP_PY','MMEAN_CARBON_AC_PY',
                        'MMEAN_TRANSP_PY','MMEAN_VAPOR_AC_PY',
                        'MMEAN_SENSIBLE_AC_PY']

    # TODO: add extract soil variables
    #output_soil_vars = ['MMEAN_SOIL_MSTPOT_PY'] # need to be added later

    # second some pft level data
    output_pft_vars = ['AGB','MMEAN_LAI','BA','NPLANT','MMEAN_GPP','MMEAN_NPP']
    output_pft_vars = np.sort(list(set(output_pft_vars) | set(voi)))

    # third some size level data
    output_size_vars = ['AGB','MMEAN_LAI','BA','NPLANT','DDBH_DT','MMEAN_MORT_RATE']


    #------------------  Loop Over Years  ---------------------------------#
    year_array = np.arange(output_yeara,output_yearz+1)
    for iyear, year in enumerate(year_array):
        #------------------  Loop Over Years  -----------------------------#
        # Define/Clear the dict that stores output data
        # we do this so that the memory does not explode

        # first monthly output with PFT data
        output_monthly_dict ={}
        for var in col_list + output_met_vars + output_flux_vars:
            output_monthly_dict[var] = np.zeros((12,))

        # PFT-specific vars
        for var in output_pft_vars:
            for pft in pft_list:
                output_monthly_dict['{:s}_PFT_{:d}'.format(var,pft)] = np.zeros((12,))

        # size-specific vars
        for var in output_size_vars:
            for isize, size_edge in enumerate(dbh_size_list[1]):
                output_monthly_dict['{:s}_{:s}_{:d}'.format(var,dbh_size_list[0],isize)] = np.zeros((12,))
            for isize, size_edge in enumerate(hite_size_list[1]):
                output_monthly_dict['{:s}_{:s}_{:d}'.format(var,hite_size_list[0],isize)] = np.zeros((12,))


        # loop over month
        montha, monthz = 1, 12
        if year == output_yeara:
            montha = output_montha
        if year == output_yearz:
            monthz = output_monthz
        month_array = np.arange(montha,monthz+1)


        for month in month_array:
            # append time stamps
            output_monthly_dict['year'][month-1]  = year
            output_monthly_dict['month'][month-1] = month
            output_monthly_dict['time'][month-1] = year + (month-0.5) / 12.

            # read data
            data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
                data_pf,year,month)

            if not os.path.isfile(data_fn):
                print('{:s} doest not exist!'.format(data_fn))
                # file does not exist
                # Just ignore the file
                continue

            h5in    = h5py.File(data_fn,'r')

            # first extract avg vars
            voi_extract = output_met_vars + output_flux_vars
            exut.extract_avg(h5in,output_monthly_dict,voi_extract,month-1)

            # second extract PFT vars
            voi_extract = output_pft_vars
            exut.extract_pft(h5in,output_monthly_dict,
                        voi_extract,pft_list,month-1)

            # third extract size vars
            voi_extract = output_size_vars
            exut.extract_size(h5in,output_monthly_dict,
                         voi_extract,dbh_size_list,month-1)
            exut.extract_size(h5in,output_monthly_dict,
                         voi_extract,hite_size_list,month-1)
            h5in.close()

        # get rid of the zero entries
        nonzero_mask = ~((output_monthly_dict['year'] == 0) & 
                         (output_monthly_dict['month'] == 0))
        for var in output_monthly_dict.keys():
            output_monthly_dict[var] = output_monthly_dict[var][nonzero_mask]
        # save the extracted data to a dictionary
        csv_monthly_df = pd.DataFrame(data = output_monthly_dict)
        csv_monthly_fn = out_dir + out_pf + 'monthly.csv'
        # if it is the first year overwrite
        # otherwise append
        if iyear == 0:
            csv_monthly_df.to_csv(csv_monthly_fn,index=False,mode='w',header=True)
        else:
            csv_monthly_df.to_csv(csv_monthly_fn,index=False,mode='a',header=False)

        del csv_monthly_df


    # census time scale
    # for each census we will extract
    # pft_size vars (at the end)
    output_pft_size_vars = ['AGB','MMEAN_LAI','BA','NPLANT']

    # individual vars for plot (at the end)
    output_individual_dict = {}
    
    # demographic rates...
    output_demography_vars = ['census_yeara','census_montha','census_yearz','census_monthz',
                              'DBH_init','NPLANT_init','PFT_init','HITE_init','DDBH_DT_avg',
                              'MORT_RATE_avg']
   

    # determine cenus frequency

    census_freq = 5 # census every 5 years...
    
    if (output_yearz - output_yeara) < census_freq:
        census_freq = output_yearz - output_yeara


    # first generate the census years
    # we work backword since we only track the survived cohorts....
    census_years = np.arange(output_yearz,output_yeara+census_freq,-census_freq)
    # include a maximum of 5 censuses
    if len(census_years) > 5:
        census_years = np.array([year_array[0] for year_array in np.array_split(census_years,5) ])

    # loop over the census years
    for icensus, census_end in enumerate(census_years):
        # first for this year and month, we need to extract 
        # (1) pft_size and (2) individual for plot
        output_pft_size_dict ={}
        for var in col_list:
            output_pft_size_dict[var] = np.zeros((1,))

        # size-specific vars
        for var in output_pft_size_vars:
            for pft in pft_list:
                for isize, size_edge in enumerate(dbh_size_list[1]):
                    output_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(
                        var,pft,dbh_size_list[0],isize)] = np.zeros((1,))
                for isize, size_edge in enumerate(hite_size_list[1]):
                    output_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(
                        var,pft,hite_size_list[0],isize)] = np.zeros((1,))

        # append time stamps
        output_pft_size_dict['year'][0]  = census_end
        output_pft_size_dict['month'][0] = output_monthz
        output_pft_size_dict['time'][0] = census_end + (output_monthz-0.5) / 12.

        # get individual file
        output_individual_dict ={}
        for var in individual_vars:
            output_individual_dict[var] = []

        # get demography file
        output_demography_dict = {}
        for var in output_demography_vars:
            output_demography_dict[var] = []

        census_yeara = census_end - census_freq
        census_yearz = census_end
        census_montha = output_monthz
        census_monthz = output_monthz
        exut.extract_demo_rates_monthly(data_pf,output_demography_dict,
                                  census_yeara,census_montha,
                                  census_yearz,census_monthz)

        
        # save the demography files
        csv_demography_df = pd.DataFrame(data = output_demography_dict)
        csv_demography_fn = out_dir + out_pf + 'monthly_demography.csv'
        # if it is the first year overwrite
        # otherwise append
        if icensus == 0:
            csv_demography_df.to_csv(csv_demography_fn,index=False,mode='w',header=True)
        else:
            csv_demography_df.to_csv(csv_demography_fn,index=False,mode='a',header=False)

        del csv_demography_df

        # read data
        data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
            data_pf,census_end,output_monthz)

        if not os.path.isfile(data_fn):
            print('{:s} doest not exist!'.format(data_fn))
            # file does not exist
            # Just ignore the file
            continue

        h5in    = h5py.File(data_fn,'r')
        voi_extract = output_pft_size_vars
        exut.extract_pft_size(h5in,output_pft_size_dict,
                         voi_extract,pft_list,dbh_size_list,0)
        exut.extract_pft_size(h5in,output_pft_size_dict,
                         voi_extract,pft_list,hite_size_list,0)

        # extract individual vars
        exut.extract_individual_for_plot(h5in,output_individual_dict,
                                        pft_list,census_end)


        h5in.close()

        # done save the file
        # save the extracted data to a dictionary
        csv_pft_size_df = pd.DataFrame(data = output_pft_size_dict)
        csv_pft_size_fn = out_dir + out_pf + 'monthly_pft_size.csv'
        # if it is the first year overwrite
        # otherwise append
        if icensus == 0:
            csv_pft_size_df.to_csv(csv_pft_size_fn,index=False,mode='w',header=True)
        else:
            csv_pft_size_df.to_csv(csv_pft_size_fn,index=False,mode='a',header=False)

        del csv_pft_size_df

        # save the individual files
        csv_individual_df = pd.DataFrame(data = output_individual_dict)
        csv_individual_fn = out_dir + out_pf + 'monthly_individual.csv'
        # if it is the first year overwrite
        # otherwise append
        if icensus == 0:
            csv_individual_df.to_csv(csv_individual_fn,index=False,mode='w',header=True)
        else:
            csv_individual_df.to_csv(csv_individual_fn,index=False,mode='a',header=False)

        del csv_individual_df



    return


