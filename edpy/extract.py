# High-level functions for users

# import modules
from . import extract_utils as exut
from .extract_utils import dbh_size_list, hite_size_list
from .extract_utils import dbh_size_list_fine, hite_size_list_fine
from .extract_utils import individual_vars
from calendar import monthrange
from datetime import datetime
import h5py
import pandas as pd
import numpy as np
import os
import sys

def extract_annual(
     data_pf : 'path and prefix of the ED2 data, output file name would be automatically generated'
    ,out_dir : 'output directory'
    ,out_pf  : 'output prefix'
    ,output_yeara   : 'start year' 
    ,output_yearz   : 'end year'   
    ,pft_list       : 'List of PFTs to include for pft VOIs' = []
    ,voi  : 'Variable of interest to extract' = ['AGB','LAI','BA','NPLANT'] 
 ):
    '''
        This function aims to extract annual scale ecosystem diagnostics for plots.

        # For all NPLANT extractions exclude grass PFT

        # size-PFT-level data (every year)
        AGB, BA, LAI, NPLANT


        # separate the whole period into N (N<=10) period
        # for each census 
        extract PFT-size level data (_pft_size) 
        extract individual level data for plot (_individual_for_plot)
        
    '''
    if (len(pft_list) == 0):
        print('No PFT is specified for PFT VOIs')
        return

    # Necessary time stamps
    col_list = ['year']
    year_array = np.arange(output_yeara,output_yearz+1)
    year_num = len(year_array)


    # Define/Clear the dict that stores output data for annual ecosystem level data

    # some pft level variables
    output_annual_dict = {}
    for var in col_list:
        output_annual_dict[var] = []#np.zeros((year_num,))

    # PFT-size-specific vars
    for var in voi:
        for pft in pft_list:
            for isize, size_edge in enumerate(dbh_size_list[1]):
                output_annual_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,dbh_size_list[0],isize)] = \
                           []# np.zeros((year_num,))
            for isize, size_edge in enumerate(hite_size_list[1]):
                output_annual_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,hite_size_list[0],isize)] = \
                           []# np.zeros((year_num,))

    #------------------  Loop Over Years  ---------------------------------#
    for iyear, year in enumerate(year_array):
        #------------------  Loop Over Years  -----------------------------#
        # append time stamps
        output_annual_dict['year']  += [year]

        # read data
        data_fn = '{:s}-Y-{:4d}-00-00-000000-g01.h5'.format(
            data_pf,year)

        if not os.path.isfile(data_fn):
            print('{:s} doest not exist!'.format(data_fn))
            # file does not exist
            # Just ignore the file
            continue

        h5in    = h5py.File(data_fn,'r')

        # extract PFT size vars
        exut.extract_pft_size(h5in,output_annual_dict,
                         voi,pft_list,dbh_size_list)
        exut.extract_pft_size(h5in,output_annual_dict,
                         voi,pft_list,hite_size_list)

        h5in.close()

    # save the data
    # pft_level
    csv_df = pd.DataFrame(data = output_annual_dict)
    csv_fn = out_dir + out_pf + 'annual.csv'
    csv_df.to_csv(csv_fn,index=False,mode='w',header=True,float_format='%g')
    del csv_df

    # pft and size, & indiviudal level extractions
    # first we need to generate year_array for size output
    output_pft_size_dict = {}
    year_step = np.maximum(1,(np.amax(year_array) - np.amin(year_array)+1) // 5)
    if year_step > 5:
        year_step = year_step // 5 * 5

    year_num = np.minimum(5,len(year_array)) 

    size_year_array = np.arange(year_array[-1]-(year_num-1)*year_step,year_array[-1]+1,year_step)

    # don't include output demography

#    output_demography_vars = ['yeara','yearz']
#
#    for pft in pft_list:
#        for isize, size_edge in enumerate(dbh_size_list[1]):
#            output_demography_vars.append(
#                'BA_GROWTH_FRAC_PFT_{:d}_{:s}_{:d}'.format(pft,dbh_size_list[0],isize)
#            )
#            output_demography_vars.append(
#                'BA_MORT_FRAC_PFT_{:d}_{:s}_{:d}'.format(pft,dbh_size_list[0],isize)
#            )
#    output_demography_dict = {}
#    for var in output_demography_vars:
#        output_demography_dict[var] = np.zeros((len(size_year_array),))

    # PFT and Size vars
    for var in col_list:
        output_pft_size_dict[var] = [] # np.zeros((len(size_year_array),))

    for var in voi:
        for pft in pft_list:
            for isize, size_edge in enumerate(dbh_size_list_fine[1]):
                output_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,dbh_size_list_fine[0],isize)] = \
                       []# np.zeros((len(size_year_array),))

            for isize, size_edge in enumerate(hite_size_list_fine[1]):
                output_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,hite_size_list_fine[0],isize)] = \
                       []# np.zeros((len(size_year_array),))


    # we also need to extract individual information to plot
    output_individual_dict = {}
    for var in individual_vars:
        output_individual_dict[var] = []



    #------------------  Loop Over Years  ---------------------------------#
    for iyear, year in enumerate(size_year_array):
        #------------------  Loop Over Years  -----------------------------#
        # append time stamps
        output_pft_size_dict['year']  += [year]

        # read data
        data_fn = '{:s}-Y-{:4d}-00-00-000000-g01.h5'.format(
            data_pf,year)

        if not os.path.isfile(data_fn):
            print('{:s} doest not exist!'.format(data_fn))
            # file does not exist
            # Just ignore the file
            continue

        h5in    = h5py.File(data_fn,'r')

        # extract individual vars
        exut.extract_individual_for_plot(h5in,output_individual_dict,
                                        pft_list,year)

        # extract PFT-SIZE vars
        exut.extract_pft_size(h5in,output_pft_size_dict,
                         voi,pft_list,dbh_size_list_fine)

        exut.extract_pft_size(h5in,output_pft_size_dict,
                         voi,pft_list,hite_size_list_fine)

        h5in.close()

        # extract demography data
#        yeara = year - np.minimum(year_step,5)+1
#        yearz = year
#        output_demography_dict['yeara'][iyear] = yeara
#        output_demography_dict['yearz'][iyear] = yearz
#        extract_demo_rates_annual(
#            data_pf,output_demography_dict,
#            yeara,yearz,pft_list,iyear)




    # save the data
    # pft_size_level
    csv_df = pd.DataFrame(data = output_pft_size_dict)
    csv_fn = out_dir + out_pf + 'annual_pft_size.csv'
    csv_df.to_csv(csv_fn,index=False,mode='w',header=True,float_format='%g')
    del csv_df

#    # demography
#    csv_df = pd.DataFrame(data = output_demography_dict)
#    csv_fn = out_dir + out_pf + 'annual_demography.csv'
#    csv_df.to_csv(csv_fn,index=False,mode='w',header=True,float_format='%g')
#    del csv_df

    # individual_level
    csv_df = pd.DataFrame(data = output_individual_dict)
    csv_fn = out_dir + out_pf + 'annual_individual_plot.csv'
    csv_df.to_csv(csv_fn,index=False,mode='w',header=True,float_format='%g')
    del csv_df

    # pft-size-age extraction for the last year
    output_pft_size_age_dict = {}
    # only include year
    col_list = exut.EDTime.time_names + ['PFT','SIZE','PATCH_AGE','PATCH_AREA'] + voi
    for var in col_list:
        output_pft_size_age_dict[var] = []

    year = year_array[-1]
    cur_time = exut.EDTime(year=year)
    
    # read data
    data_fn = '{:s}-Y-{:4d}-00-00-000000-g01.h5'.format(
        data_pf,year)

    if not os.path.isfile(data_fn):
        print('{:s} doest not exist!'.format(data_fn))
        print('Extract PFT-SIZE-AGE FAILED!!')
        # file does not exist
        return -1 

    h5in    = h5py.File(data_fn,'r')
    exut.extract_pft_size_age(h5in,output_pft_size_age_dict,
                         voi,pft_list,hite_size_list_fine,cur_time)
    exut.extract_pft_size_age(h5in,output_pft_size_age_dict,
                         voi,pft_list,dbh_size_list_fine,cur_time)
    h5in.close()

    csv_df = pd.DataFrame(data = output_pft_size_age_dict)
    csv_fn = out_dir + out_pf + 'annual_pft_size_age.csv'
    csv_df.to_csv(csv_fn,index=False,mode='w',header=True,float_format='%g')
    del csv_df




    return


##################################################
##################################################


def extract_monthly(
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
    # using extract_polygon
    output_met_vars = ['MMEAN_ATM_TEMP_PY','MMEAN_PCPG_PY','MMEAN_ATM_RSHORT_PY',
                       'MMEAN_ATM_VPDEF_PY','MMEAN_ATM_CO2_PY']
    output_flux_vars = ['MMEAN_GPP_PY','MMEAN_NPP_PY','MMEAN_LEAF_RESP_PY',
                        'MMEAN_STEM_RESP_PY','MMEAN_ROOT_RESP_PY','MMEAN_CARBON_AC_PY',
                        'MMEAN_TRANSP_PY','MMEAN_VAPOR_AC_PY','MMEAN_SENSIBLE_AC_PY']

    # TODO: add extract soil variables
    #output_soil_vars = ['MMEAN_SOIL_MSTPOT_PY'] # need to be added later

    # second some pft level data
    output_pft_vars = ['AGB','MMEAN_LAI','BA','NPLANT','MMEAN_GPP']
    output_pft_vars = np.sort(list(set(output_pft_vars) | set(voi)))

    # third some size level data
    output_size_vars = ['AGB','MMEAN_LAI','BA','NPLANT','MMEAN_GPP']
    output_size_vars = np.sort(list(set(output_size_vars) | set(voi)))


    #------------------  Loop Over Years  ---------------------------------#
    year_array = np.arange(output_yeara,output_yearz+1)
    for iyear, year in enumerate(year_array):
        #------------------  Loop Over Years  -----------------------------#
        # Define/Clear the dict that stores output data
        # we do this so that the memory does not explode

        # first monthly output with PFT data
        output_monthly_dict ={}
        for var in col_list + output_met_vars + output_flux_vars:
            output_monthly_dict[var] = [] #np.zeros((12,))

        # PFT-specific vars
        for var in output_pft_vars:
            for pft in pft_list:
                output_monthly_dict['{:s}_PFT_{:d}'.format(var,pft)] = [] #np.zeros((12,))

        # size-specific vars
        for var in output_size_vars:
            for isize, size_edge in enumerate(dbh_size_list[1]):
                output_monthly_dict['{:s}_{:s}_{:d}'.format(var,dbh_size_list[0],isize)] = [] #np.zeros((12,))
            for isize, size_edge in enumerate(hite_size_list[1]):
                output_monthly_dict['{:s}_{:s}_{:d}'.format(var,hite_size_list[0],isize)] = [] #np.zeros((12,))


        # loop over month
        montha, monthz = 1, 12
        if year == output_yeara:
            montha = output_montha
        if year == output_yearz:
            monthz = output_monthz
        month_array = np.arange(montha,monthz+1)


        for month in month_array:
            # append time stamps
            output_monthly_dict['year']  += [year]
            output_monthly_dict['month'] += [month]
            output_monthly_dict['time']  += [(year + (month-0.5) / 12.)]

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
            exut.extract_polygon(h5in,output_monthly_dict,voi_extract)

            # second extract PFT vars
            voi_extract = output_pft_vars
            exut.extract_pft(h5in,output_monthly_dict,
                        voi_extract,pft_list)

            # third extract size vars
            voi_extract = output_size_vars
            exut.extract_size(h5in,output_monthly_dict,
                         voi_extract,dbh_size_list)
            exut.extract_size(h5in,output_monthly_dict,
                         voi_extract,hite_size_list)
            h5in.close()

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
    output_pft_size_vars = ['AGB','LAI','BA','NPLANT']

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
            output_pft_size_dict[var] = []

        # size-specific vars
        for var in output_pft_size_vars:
            for pft in pft_list:
                for isize, size_edge in enumerate(dbh_size_list_fine[1]):
                    output_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(
                        var,pft,dbh_size_list[0],isize)] = []
                for isize, size_edge in enumerate(hite_size_list_fine[1]):
                    output_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(
                        var,pft,hite_size_list[0],isize)] = []

        # append time stamps
        output_pft_size_dict['year']  += [census_end]
        output_pft_size_dict['month'] += [output_monthz]
        output_pft_size_dict['time']  += [census_end + (output_monthz-0.5) / 12.]

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
                         voi_extract,pft_list,dbh_size_list_fine)
        exut.extract_pft_size(h5in,output_pft_size_dict,
                         voi_extract,pft_list,hite_size_list_fine)

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

################################################################
# %% Extract Qmean output, which contains average diurnal cycles
################################################################
def extract_monthly_diurnal(
     data_pf : 'path and prefix of the ED2 data, output file name would be automatically generated'
    ,out_dir : 'output directory'
    ,out_pf  : 'output prefix'
    ,output_yeara   : 'start year',output_montha  : 'start month'
    ,output_yearz   : 'end year' ,output_monthz  : 'end month'  
    ,voi            : 'variable of interests (ecosystem average)' = ['AGB','LAI','BA','NPLANT']
    ,include_cohort      : 'whether include cohort level variable' = False
 ):
    '''
        This function aims to extract month average diurnal cycles

        # for ecosystem level we are interested in the diurnal
        # (1) meteorology, (2) flux, (3) canopy state variables

        # for cohort level we are interested in the diurnal
        # (1) leaf state, (2) carbon/water flux, (3) leaf micro-environment
    '''

    # Necessary time stamps
    # time is the normalized year, used for plotting...
    col_list = ['year','month','hour']


    # first some polygon level data
    # using extract_polygon
    polygon_qmean_vars = ['QMEAN_ATM_TEMP_PY','QMEAN_ATM_VPDEF_PY','QMEAN_PCPG_PY',
                    'QMEAN_ATM_RSHORT_PY','QMEAN_ATM_PAR_PY','QMEAN_ATM_PAR_DIFF_PY',
                    'QMEAN_CAN_TEMP_PY','QMEAN_CAN_VPDEF_PY','QMEAN_CAN_CO2_PY',
                    'QMEAN_GPP_PY','QMEAN_NPP_PY','QMEAN_PLRESP_PY','QMEAN_NEP_PY',
                    'QMEAN_LEAF_RESP_PY','QMEAN_STEM_RESP_PY','QMEAN_ROOT_RESP_PY',
                    'QMEAN_WATER_SUPPLY_PY','QMEAN_TRANSP_PY']

    # for now do not add size-pft specific aggregates


    # second cohort level data
    cohort_vars = ['DBH','PFT','HITE','NPLANT','LAI_CO']
    cohort_qmean_vars = [
                   'QMEAN_GPP_CO','QMEAN_NPP_CO','QMEAN_PLRESP_CO',
                   'QMEAN_LEAF_RESP_CO','QMEAN_STEM_RESP_CO','QMEAN_ROOT_RESP_CO',
                   'QMEAN_LEAF_TEMP_CO','QMEAN_LEAF_VPDEF_CO',
                   'QMEAN_LEAF_GSW_CO','QMEAN_LINT_CO2_CO','QMEAN_A_NET_CO',
                   'QMEAN_WATER_SUPPLY_CO','QMEAN_TRANSP_CO','QMEAN_LEAF_PSI_CO',
                   'QMEAN_WOOD_PSI_CO','QMEAN_WFLUX_WL_CO',
                   'QMEAN_PAR_L_CO','QMEAN_LIGHT_LEVEL_CO','QMEAN_LIGHT_CO']

    # we save into two files, one for polygon and one for cohort

    #------------------  Loop Over Years  ---------------------------------#
    year_array = np.arange(output_yeara,output_yearz+1)
    for iyear, year in enumerate(year_array):
        #------------------  Loop Over Years  -----------------------------#
        # Define/Clear the dict that stores output data
        # we do this so that the memory does not explode

        # first polygon level
        output_polygon_dict={}
        for var in col_list + polygon_qmean_vars:
            # we initiate it as an empty array since we do not know how long it would be for
            # cohort-level extractions
            output_polygon_dict[var] = []

        # second cohort level
        output_cohort_dict={}
        for var in col_list + cohort_vars + cohort_qmean_vars:
            output_cohort_dict[var] = []

        # loop over month
        montha, monthz = 1, 12
        if year == output_yeara:
            montha = output_montha
        if year == output_yearz:
            monthz = output_monthz
        month_array = np.arange(montha,monthz+1)


        for month in month_array:

            # read data
            data_fn = '{:s}-Q-{:4d}-{:02d}-00-000000-g01.h5'.format(
                data_pf,year,month)

            if not os.path.isfile(data_fn):
                print('{:s} doest not exist!'.format(data_fn))
                # file does not exist
                # Just ignore the file
                continue

            h5in    = h5py.File(data_fn,'r')

            # 0. get qmean_num
            qmean_num = np.array(h5in['QMEAN_GPP_PY']).shape[1]
            # first extract polygon vars
            exut.extract_qmean(h5in,output_polygon_dict,polygon_qmean_vars)
            # append time stamps
            output_polygon_dict['year'] += ([year] * qmean_num)
            output_polygon_dict['month'] += ([month] * qmean_num)
            output_polygon_dict['hour'] += np.arange(1,qmean_num+1).tolist()

            # second extract cohort vars
            # first get number of cohort
            cohort_num = len(h5in['DBH'][:])
            exut.extract_qmean(h5in,output_cohort_dict,cohort_qmean_vars)
            # append time stamps and cohort_vars
            output_cohort_dict['year'] += ([year] * qmean_num * cohort_num)
            output_cohort_dict['month'] += ([month] * qmean_num * cohort_num)
            output_cohort_dict['hour'] += (np.arange(1,qmean_num+1).tolist() * cohort_num)
            for var in cohort_vars:
                output_cohort_dict[var] += (np.repeat(h5in[var][:],qmean_num).tolist())

            h5in.close()

        # save the extracted data to a dictionary
        output_df = pd.DataFrame(data = output_polygon_dict)
        output_fn = out_dir + out_pf + 'qmean_polygon.csv'
        # if it is the first year overwrite
        # otherwise append
        if iyear == 0:
            output_df.to_csv(output_fn,index=False,mode='w',header=True)
        else:
            output_df.to_csv(output_fn,index=False,mode='a',header=False)

        del output_df

        output_df = pd.DataFrame(data = output_cohort_dict)
        output_fn = out_dir + out_pf + 'qmean_cohort.csv'
        # if it is the first year overwrite
        # otherwise append
        if iyear == 0:
            output_df.to_csv(output_fn,index=False,mode='w',header=True)
        else:
            output_df.to_csv(output_fn,index=False,mode='a',header=False)

        del output_df


    return

################################################################
# %% Extract Fmean output
################################################################
def extract_fast(
     data_pf : 'path and prefix of the ED2 data, output file name would be automatically generated'
    ,out_dir : 'output directory'
    ,out_pf  : 'output prefix'
    ,output_yeara   : 'start year',output_montha  : 'start month',output_daya: 'start day'
    ,output_yearz   : 'end year' ,output_monthz  : 'end month', output_dayz: 'end day'
    ,output_time_list : 'a list of strings consisting time of day'
    ,voi            : 'variable of interests (ecosystem average)' = ['AGB','LAI','BA','NPLANT']
    ,pft_list       : 'List of PFTs to include for pft VOIs' = []
    ,include_cohort      : 'whether include cohort level variable' = False
 ):
    '''
        This function aims to extract month average diurnal cycles

        # for ecosystem level we are interested in the diurnal
        # (1) meteorology, (2) flux, (3) canopy state variables

        # for cohort level we are interested in the diurnal
        # (1) leaf state, (2) carbon/water flux, (3) leaf micro-environment
    '''

    # Necessary time stamps
    # time is the normalized year, used for plotting...
    col_list = ['year','month','day','doy','hour','minute','second']
    n_daytime = len(output_time_list)


    # first some polygon level data
    # using extract_polygon
    polygon_fmean_vars = [
        'FMEAN_ATM_TEMP_PY','FMEAN_ATM_VPDEF_PY','FMEAN_ATM_RSHORT_PY',
        'FMEAN_CAN_TEMP_PY','FMEAN_CAN_VPDEF_PY','FMEAN_CAN_CO2_PY',
        'FMEAN_GPP_PY','FMEAN_NPP_PY','FMEAN_PLRESP_PY','FMEAN_NEP_PY',
        'FMEAN_LEAF_RESP_PY','FMEAN_STEM_RESP_PY','FMEAN_ROOT_RESP_PY',
        'FMEAN_TRANSP_PY']

    # for now do not add size-pft specific aggregates


    # second cohort level data
    cohort_vars = ['DBH','PFT','HITE','NPLANT','LAI_CO']
    cohort_fmean_vars = [
                   'FMEAN_GPP_CO','FMEAN_NPP_CO','FMEAN_PLRESP_CO',
                   'FMEAN_LEAF_RESP_CO','FMEAN_STEM_RESP_CO','FMEAN_ROOT_RESP_CO',
                   'FMEAN_LEAF_TEMP_CO','FMEAN_LEAF_VPDEF_CO','FMEAN_LEAF_GSW_CO',
                   'FMEAN_LINT_CO2_CO','FMEAN_A_NET_CO',
                   'FMEAN_TRANSP_CO','FMEAN_WFLUX_WL_CO',
                   'FMEAN_LEAF_PSI_CO','FMEAN_WOOD_PSI_CO',
                   'FMEAN_PAR_L_CO']

    # we save into two files, one for polygon and one for cohort
    first_write = True

    #------------------  Loop Over Years  ---------------------------------#
    year_array = np.arange(output_yeara,output_yearz+1)
    for iyear, year in enumerate(year_array):
        #------------------  Loop Over Years  -----------------------------#
        # loop over month
        montha, monthz = 1, 12
        if year == output_yeara:
            montha = output_montha
        if year == output_yearz:
            monthz = output_monthz
        month_array = np.arange(montha,monthz+1)


        for month in month_array:
            # Define/Clear the dict that stores output data
            # we do this so that the memory does not explode

            # first polygon level
            output_polygon_dict={}
            for var in col_list + polygon_fmean_vars:
                # we initiate it as an empty array since we do not know how long it would be for
                # cohort-level extractions
                output_polygon_dict[var] = []

            # second cohort level
            output_cohort_dict={}
            for var in col_list + cohort_vars + cohort_fmean_vars:
                output_cohort_dict[var] = []


            # loop over day
            daya, dayz = monthrange(year,month)
            daya = 1

            if (year == output_yeara) and (month == output_montha):
                daya = output_daya

            if (year == output_yearz) and (month == output_monthz):
                dayz = output_dayz

            day_array = np.arange(daya,dayz+1)

            for day in day_array:
                # Get day of year
                doy = datetime(year,month,day).timetuple().tm_yday

                # loop over time of day
                for itime, time_str in enumerate(output_time_list):

                    data_fn = '{:s}-I-{:4d}-{:02d}-{:02d}-{:s}-g01.h5'.format(
                        data_pf,year,month,day,time_str)

                    if not os.path.isfile(data_fn):
                        print('{:s} doest not exist!'.format(data_fn))
                        # file does not exist
                        # Just ignore the file
                        continue

                    h5in    = h5py.File(data_fn,'r')

                    # first extract polygon vars
                    exut.extract_polygon(h5in,output_polygon_dict,polygon_fmean_vars)
                    # append time stamps
                    output_polygon_dict['year']   += [year]
                    output_polygon_dict['month']  += [month]
                    output_polygon_dict['day']    += [day]
                    output_polygon_dict['doy']    += [doy]
                    output_polygon_dict['hour']   += [int(time_str[0:2])]
                    output_polygon_dict['minute'] += [int(time_str[2:4])]
                    output_polygon_dict['second'] += [int(time_str[4:6])]

                    # second extract cohort vars
                    # first get number of cohort
                    cohort_num = len(h5in['DBH'][:])
                    exut.extract_cohort(h5in,output_cohort_dict,cohort_vars + cohort_fmean_vars)
                    # append time stamps 
                    output_cohort_dict['year']   += ([year] * cohort_num)
                    output_cohort_dict['month']  += ([month] * cohort_num)
                    output_cohort_dict['day']    += ([day] * cohort_num)
                    output_cohort_dict['doy']    += ([doy] * cohort_num)
                    output_cohort_dict['hour']   += ([int(time_str[0:2])] * cohort_num)
                    output_cohort_dict['minute'] += ([int(time_str[2:4])] * cohort_num)
                    output_cohort_dict['second'] += ([int(time_str[4:6])] * cohort_num)
                    h5in.close()

            # save the extracted data to a dictionary
            output_df = pd.DataFrame(data = output_polygon_dict)
            output_fn = out_dir + out_pf + 'fmean_polygon.csv'
            # if it is the first year overwrite
            # otherwise append
            if first_write:
                output_df.to_csv(output_fn,index=False,mode='w',header=True)
            else:
                output_df.to_csv(output_fn,index=False,mode='a',header=False)

            del output_df

            output_df = pd.DataFrame(data = output_cohort_dict)
            output_fn = out_dir + out_pf + 'fmean_cohort.csv'
            # if it is the first year overwrite
            # otherwise append
            if first_write:
                output_df.to_csv(output_fn,index=False,mode='w',header=True)
            else:
                output_df.to_csv(output_fn,index=False,mode='a',header=False)

            del output_df

            # by the time now, we have surely finished the first write so set first_write to False
            first_write = False


    return


def extract_treering(
     data_pf : 'path and prefix of the ED2 data'
    ,out_dir : 'output director'
    ,out_pf : 'output prefix'
    ,treering_yeara : 'year of the earliest treering to track'
    ,treering_yearz : 'year of the latest treering to track (or the year to tree coring)'
    ,last_month_of_year : 'the last month for a growth year' = 12
    ,pft_list : 'List of PFTs to include' = []
    ,dbh_min        : 'the smallest tree to core [cm]' = 10.
    ,hite_min       : 'the minimum hite of a cohort [m]' = 0.5
    ,voi_add     : 'additional variable of interests' = ['LINT_CO2']
    , include_all : 'whether track all the trees in history or just the trees survive at last census' = True
    ):
    '''
       This function will mimic the actual tree coring in the fields but for
       cohorts in ED2
       (1) It will read the monthly output at the end of simulations
       (2) It determins trees to track based on some size requirement
       (3) For each cohort to track/core, extract the growth history as long as
       possible. By default, the function will track DBH growth, Basal area
       growth, DBH, Height, PFT, NPLANT, and inter-cellular CO2 (daytime)...

       (4) write the final data into a csv file. The function appends the
       processed info to the output file for every year to avoid execessive
       memory usage
    '''

    first_write = True

    if last_month_of_year == 12:
        first_month_of_year = 1
    else:
        first_month_of_year = last_month_of_year + 1

    # Necessary stamps
    # growth_end_year: year in which growth accounting starts
    # cohort_flag: 1 -> survived in treering_yearz with dbh > dbh_min
    #              0 -> tracked but did not survived in treering_yearz
    #             -1 -> not tracked in the tree coring algorithm or it has reached hite_min
    # cohort_id: {year}_{cohort_#} -> year means the year of the last tree ring record in our
    # simulation, cohort_# is the cohort_# in that year

    col_list = ['growth_end_year','cohort_flag','cohort_id',
                'DBH','DDBH_DT','BA','DBA_DT','H','DH_DT',
                'PFT','NPLANT','PA_NUM']
    if 'LINT_CO2' in voi_add:
        col_list.append('LINT_CO2')

    # create an output dictionary
    output_dict = {}
    for col_name in col_list:
        output_dict[col_name] = []

    year_array = np.arange(treering_yearz,treering_yeara-1,-1)

    # create a backward-counting year list and a month list to loop over
    year_list = []
    month_list = []
    for iyear, year in enumerate(year_array):
        if year == treering_yearz:
            # last year
            monthz = last_month_of_year
        else:
            monthz = 12

        if year == treering_yeara:
            montha = last_month_of_year
        else:
            montha = 1

        # loop over months backward
        for imonth, month in enumerate(np.arange(monthz,montha-1,-1)):
            year_list.append(year)
            month_list.append(month)

    # total months to track
    month_num = len(month_list)

    # A brief description of the coring algorithm is listed as follows:
    # 1. Loop over months backward (year_list and month_list)
    # 2. If it is the last month_of_year, update cohorts_id (cohorts to track), save the tree ring
    # information of the past year
    # 3. If it is not the last month_of_year, match the cohorts that are tracked, save growth and
    # other information into a temporary structure

    #------------------  Loop Over Time   --------------------------------#
    for itime, year, month in zip(np.arange(len(year_list)),year_list,month_list):

        # -------------------------------------------------------------
        # read data
        data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
                data_pf,year,month)

        if not os.path.isfile(data_fn):
            print('{:s} doest not exist!'.format(data_fn))
            # file does not exist
            return -1

        h5in    = h5py.File(data_fn,'r')
        DBH      = np.array(h5in['DBH'])
        DDBH_DT  = np.array(h5in['DDBH_DT'])
        PFT      = np.array(h5in['PFT'])
        HITE     = np.array(h5in['HITE'])
        BA       = np.array(h5in['BA_CO'])
        NPLANT   = np.array(h5in['NPLANT'])

        if 'LINT_CO2' in voi_add:
            # if track intercellular CO2
            LINT_CO2 = np.array(h5in['MMEAN_LINT_CO2_CO'])  # ppm
            GPP      = np.array(h5in['MMEAN_GPP_CO'])  # kgC/pl/yr

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

        
        #-----------------------------------------------------------

        
        #-----------------------------------------------------------
        # Initialization of temporary structures
        #-----------------------------------------------------------
        if itime == 0:
            # the very first month
            # no previous information
            # create new structures
            cur_cohort_flag = np.zeros_like(DBH)
            cur_cohort_id = np.empty_like(DBH,dtype=object)

            # loop over cohort to update the flag
            for ico in np.arange(len(DBH)):
                if (len(pft_list) == 0 or PFT[ico] in pft_list):
                    # if this pft is tracked
                    if DBH[ico] >= dbh_min:
                        # larger than dbh_min
                        cur_cohort_flag[ico] = 1
                        cur_cohort_id[ico] = '{:04d}_{:04d}'.format(year,ico+1)
                    else:
                        cur_cohort_flag[ico] = 0
                        cur_cohort_id[ico] = '{:04d}_{:04d}'.format(year,ico+1)
                else:
                    cur_cohort_flag[ico] = -1
                    cur_cohort_id[ico] = '{:04d}_{:04d}'.format(year,0)

            # create structures to temporarily store output information
            # All information is recorded at the END of the month
            cur_cohort_pft = PFT.copy()
            cur_cohort_pa = PA_NUM.copy()
            cur_cohort_dbh = np.zeros((len(cur_cohort_flag),13))
            cur_cohort_ddbh_dt = np.zeros((len(cur_cohort_flag),13))
            cur_cohort_ba = np.zeros((len(cur_cohort_flag),13))
            cur_cohort_hite = np.zeros((len(cur_cohort_flag),13))
            cur_cohort_nplant = np.zeros((len(cur_cohort_flag),13))

            # initial conditions

            cur_cohort_dbh[:,-1] = DBH
            cur_cohort_ddbh_dt[:,-1] = DDBH_DT
            cur_cohort_ba[:,-1] = BA
            cur_cohort_hite[:,-1] = HITE
            cur_cohort_nplant[:,-1] = NPLANT

            if 'LINT_CO2' in voi_add:
                cur_cohort_gpp = np.zeros((len(cur_cohort_flag),13))
                cur_cohort_lint_co2 = np.zeros((len(cur_cohort_flag),13))

                cur_cohort_gpp[:,-1] = GPP
                cur_cohort_lint_co2[:,-1] = LINT_CO2

        #-----------------------------------------------------------
        #-----------------------------------------------------------

        
        #-----------------------------------------------------------
        # Match cohort and fill the temporary structures
        #-----------------------------------------------------------
        if itime > 0:
            # always need to do this except for the first month
            if month == 12:
                cur_last_month = 1
            else:
                cur_last_month = month + 1


            # if cur_last_month is the last_month_of_year
            # actual data is stored as the last element

            if cur_last_month == last_month_of_year:
                cur_last_idx = -1
            else:
                cur_last_idx = cur_last_month - 1


            for ico, dbh_end in enumerate(cur_cohort_dbh[:,cur_last_idx]):
                if cur_cohort_flag[ico] == -1:
                    # skip
                    continue

                # find the matching cohort
                cur_ddbh_dt = cur_cohort_ddbh_dt[ico,cur_last_idx]/12.
                dbh_mask = np.absolute(
                                (dbh_end - cur_ddbh_dt) / DBH - 1.) < 1e-8
                pft_mask = (PFT == cur_cohort_pft[ico])

                # patch number does not necessarily match
                pa_mask  = (PA_NUM == cur_cohort_pa[ico])

                if (np.sum(dbh_mask & pft_mask & pa_mask) >= 1):
                    # find the exact match or there are two cohorts matching with this one due to
                    # cohort fusion

                    # find the one with the closes nplant with the ico
                    ico_match_array = \
                        np.arange(len(DBH))[dbh_mask & pft_mask & pa_mask]

                    ico_match_nplant = NPLANT[dbh_mask & pft_mask & pa_mask]
                    ico_match = ico_match_array[
                        np.argmin(
                            np.absolute(ico_match_nplant -
                                        cur_cohort_nplant[ico,cur_last_idx]))]
                    # use the first one
                    #update cohort information
                    cur_cohort_dbh[ico,month-1] = DBH[ico_match]
                    cur_cohort_ddbh_dt[ico,month-1] = DDBH_DT[ico_match]
                    cur_cohort_ba[ico,month-1] = BA[ico_match]
                    cur_cohort_hite[ico,month-1] = HITE[ico_match]
                    cur_cohort_nplant[ico,month-1] = NPLANT[ico_match]
                    
                    if 'LINT_CO2' in voi_add:
                        cur_cohort_gpp[ico,month-1] = GPP[ico_match]
                        cur_cohort_lint_co2[ico,month-1] = LINT_CO2[ico_match]

                elif (np.sum(pft_mask) > 1 and cur_cohort_hite[ico,cur_last_idx] > 2.):
                    # the no match is due to cohort split/fusion
                    # find the cohort with the smallest difference
                    ico_match_array = \
                        np.arange(len(DBH))[pft_mask]

                    ico_match_dbh = DBH[pft_mask]

                    ico_match = ico_match_array[
                        np.argmin(np.absolute(
                            (dbh_end - cur_ddbh_dt) / ico_match_dbh - 1.))]

                    cur_cohort_dbh[ico,month-1] = DBH[ico_match]
                    cur_cohort_ddbh_dt[ico,month-1] = DDBH_DT[ico_match]
                    cur_cohort_ba[ico,month-1] = BA[ico_match]
                    cur_cohort_hite[ico,month-1] = HITE[ico_match]
                    cur_cohort_nplant[ico,month-1] = NPLANT[ico_match]
                    
                    if 'LINT_CO2' in voi_add:
                        cur_cohort_gpp[ico,month-1] = GPP[ico_match]
                        cur_cohort_lint_co2[ico,month-1] = LINT_CO2[ico_match]

                else:
                    # truely no match or has reached the seedling stage
                    # set flag to -1
                    cur_cohort_flag[ico] = -1

        #-----------------------------------------------------------
        #-----------------------------------------------------------

        
        #-----------------------------------------------------------
        # Process and save result every year
        #-----------------------------------------------------------
        if month == last_month_of_year and itime > 0:
            # we have count one year and this is not the first month
            # in this case, we need to save the annual metrics into output_dict
            # and update temporary structures
            # previous years
            # save the previous information
            for ico, cohort_id in enumerate(cur_cohort_id):
                if cur_cohort_flag[ico] == -1:
                    # not tracked or has already reached hite_min
                    continue
                else:
                    # growth_end_year is always in the next year
                    output_dict['growth_end_year'].append(year+1)
                    output_dict['cohort_flag'].append(cur_cohort_flag[ico])
                    output_dict['cohort_id'].append(cur_cohort_id[ico])
                    output_dict['PFT'].append(cur_cohort_pft[ico])
                    output_dict['PA_NUM'].append(cur_cohort_pa[ico])

                    # DBH at the end of the growth year
                    output_dict['DBH'].append(cur_cohort_dbh[ico,-1])
                    output_dict['DDBH_DT'].append(
                        (   cur_cohort_dbh[ico,-1]
                        -   cur_cohort_dbh[ico,last_month_of_year-1]
                        ))
                    output_dict['BA'].append(cur_cohort_ba[ico,-1])
                    output_dict['DBA_DT'].append(
                        (   cur_cohort_ba[ico,-1]
                        -   cur_cohort_ba[ico,last_month_of_year-1]
                        ))
                    output_dict['H'].append(cur_cohort_hite[ico,-1])
                    output_dict['DH_DT'].append(
                        (   cur_cohort_hite[ico,-1]
                        -   cur_cohort_hite[ico,last_month_of_year-1]
                        ))
                    output_dict['NPLANT'].append(cur_cohort_nplant[ico,-1])

                    if 'LINT_CO2' in voi_add:
                        month_idx = np.arange(0,13).tolist().remove(last_month_of_year-1)

                        if np.nanmean(cur_cohort_gpp[ico,month_idx]) == 0:
                            # no GPP at all
                            output_dict['LINT_CO2'].append(np.nanmean(
                                cur_cohort_lint_co2[ico,month_idx]
                            ))
                        else:
                            output_dict['LINT_CO2'].append(
                                np.nansum(cur_cohort_lint_co2[ico,month_idx]
                                            *cur_cohort_gpp[ico,month_idx]) /
                                np.nansum(cur_cohort_gpp[ico,month_idx])
                                                        )

            # save the old cohort_id, cohort_flag
            prev_cohort_id = cur_cohort_id.copy()
            prev_cohort_flag = cur_cohort_flag.copy()

            # create new ones
            cur_cohort_id = np.empty_like(DBH,dtype=object)
            cur_cohort_flag = np.zeros_like(DBH)

            # match the old ones
            for ico, dbh in enumerate(DBH):
                # find whehter we can find the same cohort in the last year
                cohort_mask = (
                    (np.absolute(cur_cohort_dbh[:,last_month_of_year-1] / DBH[ico] - 1.) < 1e-8) & # Same DBH
                    (np.absolute(cur_cohort_ddbh_dt[:,last_month_of_year-1] - DDBH_DT[ico]) < 1e-8) & # Same Growth
                    (np.absolute(cur_cohort_nplant[:,last_month_of_year-1] / NPLANT[ico] - 1.) < 0.1) & # Same Nplant
                    (cur_cohort_pft == PFT[ico])  # Same PFT
                )
                patch_mask = (cur_cohort_pa == PA_NUM[ico])  # Same patch

                # patch fusion can lead to changes in patch_number for the same cohort
                # in this case cohort_mask == 0 but we still need to account for that

                if np.nansum(cohort_mask & patch_mask) == 1:
                    ico_match = np.arange(len(prev_cohort_id))[cohort_mask][0]
                    # there must only exist one cohort like this
                    cur_cohort_id[ico] = prev_cohort_id[ico_match]
                    cur_cohort_flag[ico] = prev_cohort_flag[ico_match]

                elif np.nansum(cohort_mask) == 1:
                    ico_match = np.arange(len(prev_cohort_id))[cohort_mask][0]
                    # there must only exist one cohort like this
                    cur_cohort_id[ico] = prev_cohort_id[ico_match]
                    cur_cohort_flag[ico] = prev_cohort_flag[ico_match]
                    print('Unique match with patch fusion!')

                elif np.nansum(cohort_mask) > 1:
                    # TODO: modify the code to account for cohort split
                    # there are multiple cohorts that are the same as the prev_cohort
                    print('Mutliple Match!')
                    # only copy the first one
                    ico_match = np.arange(len(prev_cohort_id))[cohort_mask][0]
                    # there must only exist one cohort like this
                    cur_cohort_id[ico] = prev_cohort_id[ico_match]
                    cur_cohort_flag[ico] = prev_cohort_flag[ico_match]

                    # ignore the rest

                elif np.nansum(cohort_mask) == 0:
                    # patch does not match but there are matching cohorts
                    # this can due to patch fusion

                    # a new cohort
                    if (len(pft_list) == 0 or PFT[ico] in pft_list):
                        # if this pft is tracked
                        if DBH[ico] >= dbh_min:
                            # larger than dbh_min
                            cur_cohort_flag[ico] = 1
                            cur_cohort_id[ico] = '{:04d}_{:04d}'.format(year,ico+1)
                        else:
                            cur_cohort_flag[ico] = 0
                            cur_cohort_id[ico] = '{:04d}_{:04d}'.format(year,ico+1)

                    else:
                        cur_cohort_flag[ico] = -1
                        cur_cohort_id[ico] = '{:04d}_{:04d}'.format(year,0)

            # create structures to temporarily store output information
            # All information is recorded at the END of the month
            cur_cohort_pft = PFT.copy()
            cur_cohort_pa = PA_NUM.copy()
            cur_cohort_dbh = np.zeros((len(cur_cohort_flag),13))
            cur_cohort_ddbh_dt = np.zeros((len(cur_cohort_flag),13))
            cur_cohort_ba = np.zeros((len(cur_cohort_flag),13))
            cur_cohort_hite = np.zeros((len(cur_cohort_flag),13))
            cur_cohort_nplant = np.zeros((len(cur_cohort_flag),13))

            # initial conditions
            cur_cohort_dbh[:,-1] = DBH
            cur_cohort_ddbh_dt[:,-1] = DDBH_DT
            cur_cohort_ba[:,-1] = BA
            cur_cohort_hite[:,-1] = HITE
            cur_cohort_nplant[:,-1] = NPLANT

            if 'LINT_CO2' in voi_add:
                cur_cohort_gpp = np.zeros((len(cur_cohort_flag),13))
                cur_cohort_lint_co2 = np.zeros((len(cur_cohort_flag),13))

                cur_cohort_gpp[:,-1] = GPP
                cur_cohort_lint_co2[:,-1] = LINT_CO2



        #-----------------------------------------------------------
        #-----------------------------------------------------------


        # save the data when necessary
        if (month == last_month_of_year and 
            (len(output_dict['cohort_flag']) > 500 or itime == (len(month_list)-1))
           ):
            # when there are more than 500 entries or it is the last month to check
            csv_df = pd.DataFrame(data = output_dict)[col_list]
            csv_fn = out_dir + out_pf + 'treering.csv'

            if first_write:
                csv_df.to_csv(csv_fn,index=False,mode='w',header=True)
                first_write = False
            else:
                csv_df.to_csv(csv_fn,index=False,mode='a',header=False)

            del csv_df

            # empty output_dict as well
            del output_dict
            output_dict = {}
            for col_name in col_list:
                output_dict[col_name] = []

    
#    # Find out how many cohorts to core
#    # read the first month in growth year in the year after output_yearz
#    data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
#        data_pf,output_yearz,first_month_of_year)
#
#    if not os.path.isfile(data_fn):
#        print('{:s} doest not exist!'.format(data_fn))
#        # file does not exist
#        return -1
#
#    h5in    = h5py.File(data_fn,'r')
#    DBH_final      = np.array(h5in['DBH'])
#    DDBH_DT_final  = np.array(h5in['DDBH_DT'])
#    PFT_final      = np.array(h5in['PFT'])
#    HITE_final     = np.array(h5in['HITE'])
#    BA_final       = np.array(h5in['BA_CO'])
#    NPLANT_final   = np.array(h5in['NPLANT'])
#    AREA           = np.array(h5in['AREA'])
#    PACO_ID        = np.array(h5in['PACO_ID'])
#    PACO_N         = np.array(h5in['PACO_N'])
#    h5in.close()
#
#    # create an array of patch number for each cohort
#    PA_NUM_final   = np.zeros_like(NPLANT_final)
#    for ico_pa in np.arange(len(PA_NUM_final)):
#        PA_NUM_final[ico_pa] = np.where(PACO_ID <= (ico_pa + 1))[0][-1]
#
#    # loop over patches to modify NPLANT with patch area
#    # generate arrays of masks for Patch
#    for ipa in np.arange(len(PACO_ID)):
#        cohort_mask = ((np.arange(len(NPLANT_final)) >= PACO_ID[ipa]-1) &
#                           (np.arange(len(NPLANT_final)) < PACO_ID[ipa]+PACO_N[ipa]-1))
#        NPLANT_final[cohort_mask] *= AREA[ipa]
#
#    # for the survivors only track cohorts bigger than dbh_min
#    cohort_num = len(DBH_final)
#    cohort_id_final = np.arange(len(DBH_final))
#    cohort_mask = DBH_final < dbh_min
#    cohort_id_final[cohort_mask] = -1 # these trees are not recorded as surviving trees
#
##    DBH_final = DBH_final[cohort_mask]
##    DDBH_DT_final = DDBH_DT_final[cohort_mask]
##    PFT_final = PFT_final[cohort_mask]
##    HITE_final = HITE_final[cohort_mask]
##    BA_final = BA_final[cohort_mask]
##    NPLANT_final = NPLANT_final[cohort_mask]
##    PA_NUM_final = PA_NUM_final[cohort_mask]
#
#    
#    #------------------  Loop Over Years   --------------------------------#
#    for iyear, year in enumerate(year_array):
#        # allocate space to output structure
#        print(year)
#        # start of the tracking, initiate the growth year end status
#        if iyear == 0:
#            DBH_end = DBH_final.copy()
#            DDBH_DT_end = DDBH_DT_final.copy()
#            PFT_end = PFT_final.copy()
#            HITE_end = HITE_final.copy()
#            NPLANT_end = NPLANT_final.copy()
#            PA_NUM_end = PA_NUM_final.copy()
#            cohort_id_end = cohort_id_final.copy()
#        else:
#            # in this case, we need to read the monthly output from the last
#            # year and update the cohort_id_end based on cur_ variables from
#            # last year
#
#            # read the first month in growth year in the year after output_yearz
#            data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
#                data_pf,year+1,first_month_of_year)
#
#            if not os.path.isfile(data_fn):
#                print('{:s} doest not exist!'.format(data_fn))
#                print('Make sure you have at least one year simulation after output_yearz {:d}'.format(output_yearz))
#                # file does not exist
#                return -1
#
#            h5in    = h5py.File(data_fn,'r')
#            DBH_end      = np.array(h5in['DBH'])
#            DDBH_DT_end      = np.array(h5in['DDBH_DT'])
#            PFT_end      = np.array(h5in['PFT'])
#            HITE_end     = np.array(h5in['HITE'])
#            BA_end       = np.array(h5in['BA_CO'])
#            NPLANT_end   = np.array(h5in['NPLANT'])
#            AREA           = np.array(h5in['AREA'])
#            PACO_ID        = np.array(h5in['PACO_ID'])
#            PACO_N         = np.array(h5in['PACO_N'])
#            h5in.close()
#
#            # create an array of patch number for each cohort
#            PA_NUM_end   = np.zeros_like(NPLANT_end)
#            for ico_pa in np.arange(len(PA_NUM_end)):
#                PA_NUM_end[ico_pa] = np.where(PACO_ID <= (ico_pa + 1))[0][-1]
#
#            # loop over patches to modify NPLANT with patch area
#            # generate arrays of masks for Patch
#            for ipa in np.arange(len(PACO_ID)):
#                cohort_mask = ((np.arange(len(NPLANT_end)) >= PACO_ID[ipa]-1) &
#                                   (np.arange(len(NPLANT_end)) < PACO_ID[ipa]+PACO_N[ipa]-1))
#                NPLANT_end[cohort_mask] *= AREA[ipa]
#
#            # for the survivors only track cohorts bigger than dbh_min
#            # now we need to loop over each cur_ variable to update
#            # cohort_id_end
#            cohort_id_end = np.ones_like(DBH_end) * -1
#            for cur_ico in np.arange(len(cur_DBH)):
#                if cur_cohort_id[cur_ico] == -1 or cur_DBH[cur_ico] == 0:
#                    # not survivor or has reached seedling
#                    continue
#
#                # find the cohort that has the same DBH, PFT, PA_NUM, DDBH_DT
#                cohort_mask = ((DBH_end == cur_DBH[cur_ico]) &
#                               (PFT_end == cur_PFT[cur_ico]) &
#                               (PA_NUM_end == cur_PA_NUM[cur_ico]) &
#                               (DDBH_DT == cur_DDBH_DT[cur_ico]))
#                if (np.sum(cohort_mask) != 1):
#                    print('Error! Wrong matching cohorts at the start of the year')
#                    print(np.sum(cohort_mask))
#                
#                cohort_id_end[cohort_mask] = cur_cohort_id[cur_ico]
#
#
#        # if we are only interested in the surviving cohorts, modify _end
#        # variables using cohort_id
#        if not include_all:
#            cohort_mask = cohort_id_end >= 0
#            DBH_end = DBH_end[cohort_mask]
#            DDBH_DT_end = DDBH_DT_end[cohort_mask]
#            PFT_end = PFT_end[cohort_mask]
#            HITE_end = HITE_end[cohort_mask]
#            NPLANT_end = NPLANT_end[cohort_mask]
#            PA_NUM_end = PA_NUM_end[cohort_mask]
#            cohort_id_end = cohort_id_end[cohort_mask]
#
#
#
#
#        # after initate _end variables, we update the cur_ variables
#        print(year,np.sum(DBH_end > 0.))
#        cur_DBH = DBH_end.copy()
#        cur_DDBH_DT = DDBH_DT_end.copy()
#        cur_DDBH_annual = np.zeros_like(cur_DDBH_DT) # record annual DBH growth
#        cur_PFT = PFT_end.copy()
#        cur_HITE = HITE_end.copy()
#        cur_NPLANT = NPLANT_end.copy()
#        cur_PA_NUM = PA_NUM_end.copy()
#        cur_cohort_id = cohort_id_end.copy()
#        LINT_CO2_avg = np.zeros((len(DBH_end),12))
#        GPP_avg = np.zeros_like(LINT_CO2_avg) # used to scale LINT_CO2
#
#        cohort_num = len(cur_DBH)
#
#        # create dictoionary to store the data in this year
#        output_dict = {}
#        for var in col_list + voi:
#            output_dict[var] = np.zeros((cohort_num,))
#
#
#        # loop over each month
#        # first loop over the next year if necessary
#        month_array = np.arange(first_month_of_year-1,0,-1)
#        month_idx = 0
#        for imonth, month in enumerate(month_array):
#            data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
#                      data_pf,year+1,month)
#            if not os.path.isfile(data_fn):
#                print('Error! {:s} doest not exist!'.format(data_fn))
#                print('Exiting the program....')
#                # file does not exist
#                return -1
#
#            # read the data
#            h5in    = h5py.File(data_fn,'r')
#            DBH      = np.array(h5in['DBH'])
#            HITE     = np.array(h5in['HITE'])
#            DDBH_DT  = np.array(h5in['DDBH_DT'])  # cm/year
#            LINT_CO2 = np.array(h5in['MMEAN_LINT_CO2_CO'])  # ppm
#            GPP      = np.array(h5in['MMEAN_GPP_CO'])  # kgC/pl/yr
#            PFT      = np.array(h5in['PFT'])
#            PACO_ID  = np.array(h5in['PACO_ID'])
#            PACO_N   = np.array(h5in['PACO_N'])
#            h5in.close()
#
#            # create an array of patch number for each cohort
#            PA_NUM   = np.zeros_like(DBH)
#            for ico_pa in np.arange(len(PA_NUM)):
#                PA_NUM[ico_pa] = np.where(PACO_ID <= (ico_pa + 1))[0][-1]
#
#            # loop over cohorts
#            for ico in np.arange(cohort_num):
#                #print('ico ',ico)
#                # skip this cohort if its current DBH is <= 0. -> seedling
#                # stage reached....
#                if cur_DBH[ico] <= 0 :
#                    continue
#
#                # The same cohort should comply with the following criterions
#                # (1) PFT = cur_PFT
#                # (2) DBH + DDBH/12 = cur_DBH
#                # (3) PA_NUM = cur_PA_NUM  or  if no matching PA_NUM and
#                # cur_HITE > hite_min, that's probably because new patch formation. In the
#                # second case just use the first cohort that match (1) & (2)
#                dbh_mask = (np.absolute((DBH + cur_DDBH_DT[ico]/12.) /
#                                        cur_DBH[ico] - 1.)) < 1e-8
#                pft_mask = (PFT == cur_PFT[ico])
#                pa_mask  = (PA_NUM == cur_PA_NUM[ico])
#
#                if (np.sum(dbh_mask & pft_mask & pa_mask) == 1):
#                    # find the exact match
#                    ico_match = \
#                        np.arange(len(dbh_mask))[dbh_mask&pft_mask&pa_mask].tolist()[0]
#                    #print('exact match')
#                elif (cur_HITE[ico] > hite_min * 1.5 and np.sum(pft_mask) > 0):
#                    # no exact match but the current hite is taller than
#                    # hite_min
#
#                    # this is likely due to disturbance creating new patch or
#                    # cohort fusion/cohort split
#
#                    # in this case ignore dbh_mask and pa_mask
#                    # find the cohort with the smallest difference with cur_DBH
#                    # - DDBH_DT/12
#
#                    pft_ico_array = np.arange(len(dbh_mask))[pft_mask]
#                    
#                    ico_match = pft_ico_array[
#                        np.argmin(
#                          np.absolute(DBH[pft_mask] + cur_DDBH_DT[ico]/12. -
#                                      cur_DBH[ico]))]
#                    #print('ico {:d} disturbance or fusion'.format(ico))
#                    #print('sum dbh_mask&pft_mask: {:d}'.format(np.sum(dbh_mask&pft_mask)))
#                    #print('sum dbh_mask&pft_mask&pa_mask: {:d}'.format(np.sum(dbh_mask&pft_mask&pa_mask)))
#                    #print('DBH difference array is:')
#                    #print(np.absolute((DBH[pft_mask]+cur_DDBH_DT[ico]/12.)/cur_DBH[ico] - 1.))
#                else:
#                    # can't find any possibility of matching
#                    # we have reached seedling status
#                    # set cur_DBH to 0.
#                    cur_DBH[ico] = 0.
#                    cur_HITE[ico] = 0.
#                    print('ico {:d} reach seedling'.format(ico))
#                    continue
#
#                # reaching this place means we have found a matching cohort
#                # update cur_ values
#                LINT_CO2_avg[ico,month_idx] = LINT_CO2[ico_match]
#                GPP_avg[ico,month_idx] = GPP[ico_match]
#                cur_DBH[ico] = DBH[ico_match]
#                # before modifying cur_DDBH_DT, we need to add it to
#                # cur_DDBH_annual
#                cur_DDBH_annual[ico] += cur_DDBH_DT[ico]
#                cur_DDBH_DT[ico] = DDBH_DT[ico_match]
#                cur_PFT[ico] = PFT[ico_match]
#                cur_HITE[ico] = HITE[ico_match]
#                cur_PA_NUM[ico] = PA_NUM[ico_match]
#            # increase month_idx
#            month_idx += 1
#
#        # repeat the tracking for the current year
#        month_array = np.arange(12,first_month_of_year-1,-1)
#        for imonth, month in enumerate(month_array):
#            data_fn = '{:s}-E-{:4d}-{:02d}-00-000000-g01.h5'.format(
#                      data_pf,year,month)
#            if not os.path.isfile(data_fn):
#                print('Error! {:s} doest not exist!'.format(data_fn))
#                print('Exiting the program....')
#                # file does not exist
#                return -1
#
#            # read the data
#            h5in    = h5py.File(data_fn,'r')
#            DBH      = np.array(h5in['DBH'])
#            HITE     = np.array(h5in['HITE'])
#            DDBH_DT  = np.array(h5in['DDBH_DT'])  # cm/year
#            LINT_CO2 = np.array(h5in['MMEAN_LINT_CO2_CO'])  # ppm
#            GPP      = np.array(h5in['MMEAN_GPP_CO'])  # kgC/pl/yr
#            PFT      = np.array(h5in['PFT'])
#            PACO_ID  = np.array(h5in['PACO_ID'])
#            PACO_N   = np.array(h5in['PACO_N'])
#            h5in.close()
#
#            # create an array of patch number for each cohort
#            PA_NUM   = np.zeros_like(DBH)
#            for ico_pa in np.arange(len(PA_NUM)):
#                PA_NUM[ico_pa] = np.where(PACO_ID <= (ico_pa + 1))[0][-1]
#
#            # loop over cohorts
#            for ico in np.arange(cohort_num):
##                print('ico ',ico)
#                # skip this cohort if its current DBH is <= 0. -> seedling
#                # stage reached....
#                if cur_DBH[ico] <= 0 :
#                    continue
#
#                # The same cohort should comply with the following criterions
#                # (1) PFT = cur_PFT
#                # (2) DBH + DDBH/12 = cur_DBH
#                # (3) PA_NUM = cur_PA_NUM  or  if no matching PA_NUM and
#                # cur_HITE > hite_min, that's probably because new patch formation. In the
#                # second case just use the first cohort that match (1) & (2)
#                dbh_mask = (np.absolute((DBH + cur_DDBH_DT[ico]/12.) /
#                                        cur_DBH[ico] - 1.)) < 1e-8
#                pft_mask = (PFT == cur_PFT[ico])
#                pa_mask  = (PA_NUM == cur_PA_NUM[ico])
#
#                if (np.sum(dbh_mask & pft_mask & pa_mask) == 1):
#                    # find the exact match
#                    ico_match = \
#                        np.arange(len(dbh_mask))[dbh_mask&pft_mask&pa_mask].tolist()[0]
##                    print('exact match')
#                elif (cur_HITE[ico] > hite_min * 1.5 and np.sum(pft_mask) > 0):
#                    # no exact match but the current hite is taller than
#                    # hite_min
#
#                    # this is likely due to disturbance creating new patch or
#                    # cohort fusion
#
#                    # in this case ignore dbh_mask and pa_mask
#                    # find the cohort with the smallest difference with cur_DBH
#                    # - DDBH_DT/12
#
#                    pft_ico_array = np.arange(len(dbh_mask))[pft_mask]
#                    
#                    ico_match = pft_ico_array[
#                        np.argmin(
#                          np.absolute(DBH[pft_mask] + cur_DDBH_DT[ico]/12. -
#                                      cur_DBH[ico]))]
#                    #print('ico {:d} disturbance or fusion'.format(ico))
#                    #print('sum dbh_mask&pft_mask: {:d}'.format(np.sum(dbh_mask&pft_mask)))
#                    #print('sum dbh_mask&pft_mask&pa_mask: {:d}'.format(np.sum(dbh_mask&pft_mask&pa_mask)))
#                    #print('DBH difference array is:')
#                    #print(np.absolute((DBH[pft_mask]+cur_DDBH_DT[ico]/12.)/cur_DBH[ico] - 1.))
#                else:
#                    # can't find any possibility of matching
#                    # we have reached seedling status
#                    # set cur_DBH to 0.
#                    cur_DBH[ico] = 0.
#                    cur_HITE[ico] = 0.
#                    print('ico {:d} reach seedling'.format(ico))
#                    continue
#
#                # reaching this place means we have found a matching cohort
#                # update cur_ values
#                LINT_CO2_avg[ico,month_idx] = LINT_CO2[ico_match]
#                GPP_avg[ico,month_idx] = GPP[ico_match]
#                cur_DBH[ico] = DBH[ico_match]
#                # before modifying cur_DDBH_DT, we need to add it to
#                # cur_DDBH_annual
#                cur_DDBH_annual[ico] += cur_DDBH_DT[ico]
#                cur_DDBH_DT[ico] = DDBH_DT[ico_match]
#                cur_PFT[ico] = PFT[ico_match]
#                cur_HITE[ico] = HITE[ico_match]
#                cur_PA_NUM[ico] = PA_NUM[ico_match]
#            # increase month_idx
#            month_idx += 1
#
#        # at this time point we have found the matching cohort one year
#        # before. Record the results and update _end variables
#        for ico in np.arange(cohort_num):
#            output_idx = ico
#
#            # we first check whether the logged tree ring is reasonable...
#            if (cur_DBH[ico] <= 0.):
#                # no need to record
#                if cur_DBH[ico] > 0.:
#                    print('ico {:d} wrong tree core'.format(ico))
#                    print('cur_DBH: {:f}\tDBH_end: {:f}'.format(cur_DBH[ico],DBH_end[ico]))
#                # this check is necessary for incorrect hgt_min input
#                cur_DBH[ico] = 0.
#                cur_HITE[ico] = 0.
#                continue
#            elif (DBH_end[ico] < cur_DBH[ico]):
#                # start DBH larger than the ending DBH
#                # this is likely caused by cohort split
#                # we still allow this 
#                print('ico {:d} start DBH larger than ending DBH'.format(ico))
#                print('cur_DBH: {:f}\tDBH_end: {:f}\tgrowth: {:f}'.format(
#                    cur_DBH[ico],DBH_end[ico],cur_DDBH_annual[ico]/12.))
#
#                
#            output_dict['year'][output_idx] = year
#            output_dict['final_cohort_id'][output_idx] = cur_cohort_id[ico]
#            if 'DDBH_DT' in voi:
#                #print(ico,cur_DBH[ico],DBH_end[ico])
#                output_dict['DDBH_DT'][output_idx] = cur_DDBH_annual[ico]/12.
#            if 'DBA_DT' in voi:
#                output_dict['DBA_DT'][output_idx] = np.pi/4. * \
#                        ((cur_DBH[ico] + cur_DDBH_annual[ico]/12.) ** 2. - cur_DBH[ico] ** 2.)
#            if 'DBH' in voi:
#                output_dict['DBH'][output_idx] = cur_DBH[ico]
#            if 'HITE' in voi:
#                output_dict['HITE'][output_idx] = cur_HITE[ico]
#            if 'PFT' in voi:
#                output_dict['PFT'][output_idx] = cur_PFT[ico]
#            if 'NPLANT' in voi:
#                output_dict['NPLANT'][output_idx] = cur_NPLANT[ico]
#            if 'LINT_CO2' in voi:
#                if np.sum(GPP_avg[ico,:]) == 0.:
#                    # no GPP recorded this year
#                    output_dict['LINT_CO2'][output_idx] = \
#                            np.nanmean(LINT_CO2_avg[ico,:])
#                else:
#                    # GPP is positive
#                    output_dict['LINT_CO2'][output_idx] = \
#                            (np.sum(LINT_CO2_avg[ico,:] * GPP_avg[ico,:]) /
#                             np.sum(GPP_avg[ico,:]))
#
#        #DBH_end  = cur_DBH.copy()
#        #DDBH_DT_end = cur_DDBH_DT.copy()
#        #HITE_end = cur_HITE.copy()
#        #PFT_end  = cur_PFT.copy()
#        #PA_NUM_end = cur_PA_NUM.copy()
#        #LINT_CO2_avg[:] = 0.
#        #GPP_avg[:] = 0.
#
#
#        # get rid of the zero entries
#        nonzero_mask = ~((output_dict['year'] == 0))
#
#        if np.sum(nonzero_mask > 0):
#            # has data to write
#            for var in output_dict.keys():
#                output_dict[var] = output_dict[var][nonzero_mask]
#            # save the extracted data to a dictionary
#            csv_df = pd.DataFrame(data = output_dict)
#            # if it is the first year overwrite
#            # otherwise append
#            if iyear == 0:
#                csv_df.to_csv(csv_fn,index=False,mode='w',header=True)
#            else:
#                csv_df.to_csv(csv_fn,index=False,mode='a',header=False)
#
#            del csv_df

    print('='*80)
    print('Finished all tree cores!\nExiting the fuction successfully')
    return

#
