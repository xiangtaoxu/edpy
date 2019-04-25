# High-level functions for users

# import modules
import extract_utils as exut
from calendar import monthrange
from datetime import datetime
import h5py
import pandas as pd
import numpy as np
import os
import sys

def extract_ed2_fast(
     data_pf : 'path and prefix of the ED2 data'
    ,csv_fn  : 'name for output csv file'
    ,output_yeara   : 'start year' , output_montha : 'start month' ,output_daya : 'start day'
    ,output_yearz   : 'end year'   , output_monthz : 'end month'   ,output_dayz : 'end day'
    ,output_time_list : 'a list of strings consisting of time of day'
    ,voi_avg            : 'variable of interests (ecosystem average)' = ['LAI_PY']
    ,voi_avg_pft        : 'variable of interests (ecosystem average)' = ['LAI']
    ,voi_size           : 'variable of interests (size profile)' = []
    ,voi_pft_size       : 'variable of interests (size profile)' = []
    ,pft_list           : 'List of PFTs to include for pft VOIs' = []
    ,size_list          : 'size list used to calculate profile' = ('H',np.arange(5,25,5))
 ):

    '''
        Extract fast-time scale output, including ecosystem average vars and profile
        vars. The program appends the processed info into the output file every
        year to avoid excessive memory usage.
    '''

    #------------------  Sanity Check     ---------------------------------#
    if ( len(voi_avg) + len(voi_avg_pft)
       + len(voi_size) + len(voi_pft_size)) == 0:
        print('No variable of interest is specificed!')
        return
    elif (len(pft_list) > 0 and len(pft_list) == 0):
        print('No PFT is specified for PFT VOIs')
        return
    elif ((len(voi_size) + len(voi_pft_size)) > 0) and (len(size_list[1]) == 0):
        print('No size class is specified for SIZE VOIs')
        return


    # Necessary time stamps
    col_list = ['year','month','day','doy','hour','minute','second']
    n_daytime = len(output_time_list)

    #------------------  Loop Over Years  ---------------------------------#
    year_array = np.arange(output_yeara,output_yearz+1)
    for iyear, year in enumerate(year_array):
        #------------------  Loop Over Years  -----------------------------#
        # Define/Clear the dict that stores output data
        output_dict = {}
        for var in col_list + voi_avg:
            output_dict[var] = np.zeros((366*n_daytime,))

        # PFT vars
        for var in voi_avg_pft:
            for pft in pft_list:
                output_dict['{:s}_PFT_{:d}'.format(var,pft)] = np.zeros((366*n_daytime,))

        # Size vars
        for var in voi_size:
            for isize, size_edge in enumerate(size_list[1]):
                output_dict['{:s}_{:s}_{:d}'.format(var,size_list[0],isize)] = np.zeros((366*n_daytime,))

        # size PFT vars
        for var in voi_pft_size:
            for pft in pft_list:
                for isize, size_edge in enumerate(size_list[1]):
                    output_dict['{:s}_PFT_{:d}_{:s}_{:d}'.format(var,pft,size_list[0],isize)] = \
                            np.zeros((366*n_daytime,))


        # loop over month
        montha, monthz = 1, 12
        if year == output_yeara:
            montha = output_montha
        if year == output_yearz:
            monthz = output_monthz
        month_array = np.arange(montha,monthz+1)


        for month in month_array:
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

                    output_idx = (doy-1) * n_daytime + itime

                    # append time stamps
                    output_dict['year'][output_idx]  = year
                    output_dict['month'][output_idx] = month
                    output_dict['day'][output_idx]   = day
                    output_dict['doy'][output_idx]   = doy
                    output_dict['hour'][output_idx]  = int(time_str[0:2])
                    output_dict['minute'][output_idx]  = int(time_str[2:4])
                    output_dict['second'][output_idx]  = int(time_str[4:6])


                    # read data
                    data_fn = '{:s}-I-{:4d}-{:02d}-{:02d}-{:s}-g01.h5'.format(
                        data_pf,year,month,day,time_str)

                    if not os.path.isfile(data_fn):
                        print('{:s} doest not exist!'.format(data_fn))
                        # file does not exist
                        # Just ignore the file
                        continue

                    h5in    = h5py.File(data_fn,'r')

                    # first extract avg vars
                    if (len(voi_avg) > 0):
                        exut.extract_avg(h5in,output_dict,voi_avg,output_idx)

                    # then extract PFT vars
                    if (len(voi_avg_pft) > 0):
                        exut.extract_pft(h5in,output_dict,
                                         voi_avg_pft,pft_list,output_idx)

                    # then extract SIZE vars
                    if (len(voi_size) > 0):
                        exut.extract_size(h5in,output_dict,
                                     voi_size,size_list,output_idx)

                    # then extract PFT-SIZE vars
                    if (len(voi_pft_size) > 0):
                        exut.extract_pft_size(h5in,output_dict,
                                        voi_pft_size,pft_list,size_list,output_idx)

                    h5in.close()

        # get rid of the zero entries
        nonzero_mask = ~((output_dict['year'] == 0) & 
                     (output_dict['month'] == 0) &
                     (output_dict['day'] == 0))
        for var in output_dict.keys():
            output_dict[var] = output_dict[var][nonzero_mask]
        # save the extracted data to a dictionary
        csv_df = pd.DataFrame(data = output_dict)
        # if it is the first year overwrite
        # otherwise append
        if iyear == 0:
            csv_df.to_csv(csv_fn,index=False,mode='w',header=True)
        else:
            csv_df.to_csv(csv_fn,index=False,mode='a',header=False)

        del csv_df

    return

##################################################
##################################################




