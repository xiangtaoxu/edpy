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

def extract_ed2_annual(
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
        output_annual_dict[var] = np.zeros((year_num,))

    # PFT-size-specific vars
    for var in voi:
        for pft in pft_list:
            for isize, size_edge in enumerate(dbh_size_list[1]):
                output_annual_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,dbh_size_list[0],isize)] = \
                            np.zeros((year_num,))
            for isize, size_edge in enumerate(hite_size_list[1]):
                output_annual_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,hite_size_list[0],isize)] = \
                            np.zeros((year_num,))

    #------------------  Loop Over Years  ---------------------------------#
    for iyear, year in enumerate(year_array):
        #------------------  Loop Over Years  -----------------------------#
        # append time stamps
        output_annual_dict['year'][iyear]  = year

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
                         voi,pft_list,dbh_size_list,iyear)
        exut.extract_pft_size(h5in,output_annual_dict,
                         voi,pft_list,hite_size_list,iyear)

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
        output_pft_size_dict[var] = np.zeros((len(size_year_array),))

    for var in voi:
        for pft in pft_list:
            for isize, size_edge in enumerate(dbh_size_list_fine[1]):
                output_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,dbh_size_list_fine[0],isize)] = \
                        np.zeros((len(size_year_array),))

            for isize, size_edge in enumerate(hite_size_list_fine[1]):
                output_pft_size_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,hite_size_list_fine[0],isize)] = \
                        np.zeros((len(size_year_array),))


    # we also need to extract individual information to plot
    output_individual_dict = {}
    for var in individual_vars:
        output_individual_dict[var] = []



    #------------------  Loop Over Years  ---------------------------------#
    for iyear, year in enumerate(size_year_array):
        #------------------  Loop Over Years  -----------------------------#
        # append time stamps
        output_pft_size_dict['year'][iyear]  = year

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
                         voi,pft_list,dbh_size_list_fine,iyear)

        exut.extract_pft_size(h5in,output_pft_size_dict,
                         voi,pft_list,hite_size_list_fine,iyear)

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


