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

def extract_treering_ed2_monthly(
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
                    (np.absolute(cur_cohort_nplant[:,last_month_of_year-1] / NPLANT[ico] - 1.) < 1e-8) & # Same Nplant
                    (cur_cohort_pft == PFT[ico]) & # Same PFT
                    (cur_cohort_pa == PA_NUM[ico])  # Same patch
                )

                if np.nansum(cohort_mask) == 1:
                    ico_match = np.arange(len(prev_cohort_id))[cohort_mask][0]
                    # there must only exist one cohort like this
                    cur_cohort_id[ico] = prev_cohort_id[ico_match]
                    cur_cohort_flag[ico] = prev_cohort_flag[ico_match]

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

##########################
