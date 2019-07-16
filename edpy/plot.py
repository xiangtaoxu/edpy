# High-level plotting functions

# import modules
from . import plot_utils as plut
from .plot_utils import PFT_COLORS, voi_unit_dict
from .extract_utils import dbh_size_list,hite_size_list
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') # allow ploting without X11
import matplotlib.pyplot as plt
from fpdf import FPDF

#############################
def plot_annual(
     data_dir : 'path of the csv data'
    ,data_pf  : 'prefix of the csv data'
    ,out_dir  : 'output directory'
    ,pft_list : 'List of PFTs to include for pft VOIs'
    ,pft_names: 'Name of the pfts'
    ,include_census : 'whether to include census level measurements' = False

):
    '''
        Plot ED2 annual outputs and save results to a pdf
    '''

    # first generate the final pdf name
    pdf_name = out_dir + data_pf + 'annual_diagnostics.pdf'

    # create a pdf class
    pdf = FPDF('P','in','Letter') # Portrait, inch, Letter

    # now we plot figures
    # by default we assume letter size

    # first timeseries
    # by default we have 4 variables to plot
    # we will plot all of them in 1 page
    # each panel has 5'' wide and ~ 1 inch in height (too condensed?)
    # if there are more, we split them into different pages
    voi_pft = ['AGB','LAI','BA10','NPLANT10']
    voi_size = ['AGB','LAI','BA','NPLANT']
    time_unit = 'year'

    # read the csv
    df_fn = data_dir + data_pf + 'annual.csv'
    data_df = pd.read_csv(df_fn)
    time_data = np.unique(data_df['year'].values.astype(int))

    split_num = int(np.ceil(float(len(voi_pft)) / 5.))

    page_num = 1
    # loop over split_num
    for isp in np.arange(split_num):
        # get # of vars in this category
        var_num = min(5,(len(voi_pft) - isp * 5))
        figsize=(pdf.w * 0.9,pdf.h*0.9)
        fig,axes = plt.subplots(var_num,1,figsize=figsize,sharex=True)
        for i, ax in enumerate(axes):
            if i == var_num - 1:
                # last one
                xlabel_on = True
            else:
                xlabel_on = False
            voi_plot = voi_pft[isp*5+i]

            # for PFT
            # first prepare data
            if voi_plot in ['BA','BA10','NPLANT','NPLANT10']:
                # do not include grasses
                pft_mask = (np.array(pft_list) != 1)
            else:
                pft_mask = (np.array(pft_list) > 0)

            pft_list_plot = np.array(pft_list)[pft_mask]
            pft_names_plot = np.array(pft_names)[pft_mask]
            pft_colors_plot = np.array(PFT_COLORS[np.array(pft_list_plot).astype(int)])

            # prepare data
            # aggregate data to PFT average
            var_data = np.zeros((len(pft_list_plot),
                                 len(time_data)))

            if voi_plot in ['NPLANT','NPLANT10']:
                var_scaler = 1.e4
            else:
                var_scaler = 1.

            if voi_plot in ['BA10','NPLANT10']:
                voi_extract = voi_plot[0:-2]
            else:
                voi_extract = voi_plot

            for ipft, pft in enumerate(pft_list_plot):
                for isize, size_edge in enumerate(dbh_size_list[1]):
                    col_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(
                        voi_extract,pft,dbh_size_list[0],isize)

                    if ((voi_plot in ['BA10','NPLANT10']) and
                        isize == 0
                       ):
                        # ignore cohorts with DBH < 10cm
                        continue
            
                    temp_data = data_df[col_name].values * var_scaler
                    temp_data[np.isnan(temp_data)] = 0.
                    var_data[ipft,:] += temp_data



            plut.area_timeseries(ax,time_data,time_unit,
                            var_data,voi_plot,voi_unit_dict[voi_plot],
                            var_labels=pft_names_plot,
                            xlabel_on=xlabel_on,
                            color_list=pft_colors_plot)

            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,labels,fontsize=7,loc='upper left',
                          ncol=np.maximum(1,len(pft_list)//3),
                          frameon=False)
       
        # save the page
        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
        page_num += 1
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)



        # add the figure into pdf
        pdf.add_page()
        pdf.ln()
        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
              w=figsize[0],h=figsize[1])

        
        #
        # dbh size distributions
        size_names = []
        for isize, size_edge in enumerate(dbh_size_list[1]):
            if size_edge == dbh_size_list[1][-1]:
                # last one
                size_names.append('{:d} < D'.format(size_edge))
            else:
                size_names.append('{:d} < D <= {:d}'.format(size_edge,
                                                           dbh_size_list[1][isize+1]))
        size_colors_plot = np.array(PFT_COLORS[1:len(size_names)+1])

        fig,axes = plt.subplots(var_num,1,figsize=figsize,sharex=True)
        for i, ax in enumerate(axes):
            if i == var_num - 1:
                # last one
                xlabel_on = True
            else:
                xlabel_on = False
            voi_plot = voi_size[isp*5+i]


            # first prepare data
            if voi_plot in ['BA','BA10','NPLANT','NPLANT10']:
                # do not include grasses
                pft_mask = (np.array(pft_list) != 1)
            else:
                pft_mask = (np.array(pft_list) > 0)

            pft_list_plot = np.array(pft_list)[pft_mask]
            #pft_names_plot = np.array(pft_names)[pft_mask]
            #pft_colors_plot = np.array(PFT_COLORS[0:len(pft_list)])[pft_mask]

            # prepare data
            # aggregate data to PFT average
            var_data = np.zeros((len(dbh_size_list[1]),
                                 len(time_data)))

            if voi_plot in ['NPLANT','NPLANT10']:
                var_scaler = 1.e4
            else:
                var_scaler = 1.

            if voi_plot in ['BA10','NPLANT10']:
                voi_extract = voi_plot[0:-2]
            else:
                voi_extract = voi_plot


            for ipft, pft in enumerate(pft_list_plot):
                for isize, size_edge in enumerate(dbh_size_list[1]):
                    col_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(
                        voi_extract,pft,dbh_size_list[0],isize)

                    temp_data = data_df[col_name].values * var_scaler
                    temp_data[np.isnan(temp_data)] = 0.
                    var_data[isize,:] += temp_data



            plut.area_timeseries(ax,time_data,time_unit,
                            var_data,voi_plot,voi_unit_dict[voi_plot],
                            var_labels=size_names,
                            xlabel_on=xlabel_on,
                            color_list=size_colors_plot)

            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,labels,fontsize=7,loc='upper left',
                          ncol=np.maximum(1,len(pft_list)//3),
                          frameon=False)
       
        # save the page
        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
        page_num += 1
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)

        # add the figure into pdf
        pdf.add_page()
        pdf.ln()
        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
              w=figsize[0],h=figsize[1])

        # height size distributions
        size_names = []
        for isize, size_edge in enumerate(hite_size_list[1]):
            if size_edge == hite_size_list[1][-1]:
                # last one
                size_names.append('{:4.2f} < H'.format(size_edge))
            else:
                size_names.append('{:4.2f} < H <= {:4.2f}'.format(
                    size_edge,hite_size_list[1][isize+1]))

        size_colors_plot = np.array(PFT_COLORS[1:len(size_names)+1])

        fig,axes = plt.subplots(var_num,1,figsize=figsize,sharex=True)
        for i, ax in enumerate(axes):
            if i == var_num - 1:
                # last one
                xlabel_on = True
            else:
                xlabel_on = False
            voi_plot = voi_size[isp*5+i]

            # first prepare data
            if voi_plot in ['BA','BA10','NPLANT','NPLANT10']:
                # do not include grasses
                pft_mask = (np.array(pft_list) != 1)
            else:
                pft_mask = (np.array(pft_list) > 0)

            pft_list_plot = np.array(pft_list)[pft_mask]
            #pft_names_plot = np.array(pft_names)[pft_mask]
            #pft_colors_plot = np.array(PFT_COLORS[0:len(pft_list)])[pft_mask]

            # prepare data
            # aggregate data to PFT average
            var_data = np.zeros((len(hite_size_list[1]),
                                 len(time_data)))

            if voi_plot in ['NPLANT','NPLANT10']:
                var_scaler = 1.e4
            else:
                var_scaler = 1.

            if voi_plot in ['BA10','NPLANT10']:
                voi_extract = voi_plot[0:-2]
            else:
                voi_extract = voi_plot


            for ipft, pft in enumerate(pft_list_plot):
                for isize, size_edge in enumerate(hite_size_list[1]):
                    col_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(
                        voi_extract,pft,hite_size_list[0],isize)

                    temp_data = data_df[col_name].values * var_scaler
                    temp_data[np.isnan(temp_data)] = 0.
                    var_data[isize,:] += temp_data



            plut.area_timeseries(ax,time_data,time_unit,
                            var_data,voi_plot,voi_unit_dict[voi_plot],
                            var_labels=size_names,
                            xlabel_on=xlabel_on,
                            color_list=size_colors_plot)

            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,labels,fontsize=7,loc='upper left',
                          ncol=np.maximum(1,len(pft_list)//3),
                          frameon=False)
       
        # save the page
        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
        page_num += 1
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)

        # add the figure into pdf
        pdf.add_page()
        pdf.ln()
        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
              w=figsize[0],h=figsize[1])

    



    # plot pft size distributions for three patches (young mid old)
    voi_size = ['AGB','LAI','BA','NPLANT']

    # read csv for pft-size-age plots
    df_fn = data_dir + data_pf + 'annual_pft_size_age.csv'
    csv_df = pd.read_csv(df_fn)

    #patches to plot
    patch_age = np.sort(np.unique(csv_df['PATCH_AGE'].values))
    if len(patch_age) > 4:
        # more than 4 patches
        patch_age_split = np.array_split(patch_age,4)
        patch_age_to_plot = []
        for i in np.arange(4):
            patch_age_to_plot.append(patch_age_split[i][-1])
    else:
        patch_age_to_plot = patch_age

    patch_num = len(patch_age_to_plot)

    row_num = np.maximum(4,patch_num)

    # create a subset of dataframe to plot
    # first loop over vars
    for ivar, var_name in enumerate(voi_size):
        # create a figure
        figsize = (pdf.w * 0.9,pdf.h * 0.9 * row_num / 4.)
        fig,axes = plt.subplots(row_num,2,figsize=figsize,sharey=True)
        for i, row in enumerate(axes):
            if i >= patch_num:
                # turn off the axis
                for j, ax in enumerate(row):
                    ax.axis('off')

                # skip the rest of the plotting
                continue

            # get subset of data
            patch_mask = csv_df['PATCH_AGE'].values == patch_age_to_plot[i]
            plot_df = csv_df.loc[patch_mask,:]

            for j, ax in enumerate(row):
                if j == 0:
                    ylabel_on = True
                    size_list = ('D',np.arange(0.,200.+1.,10.))
                elif j == 1:
                    ylabel_on = False
                    size_list = ('H',np.arange(0.,50.+1.,5.))

                plut.bar_size_pft(ax,size_list,plot_df,var_name,voi_unit_dict[var_name],
                            pft_list,pft_names,ylabel_on=ylabel_on)

                if i == 0 and j == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles,labels,fontsize=7,loc='upper left',
                              ncol=np.maximum(1,len(pft_list)//3),
                                frameon=False)

                # plot the PATCH AGE and AREA
                if j == 0:
                    ax.set_title('PATCH AGE = {:4.2f} yrs, AREA = {:4.2f}'.format(
                        plot_df['PATCH_AGE'].values[0],
                        plot_df['PATCH_AREA'].values[0]))
        
        # save the page
        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
        page_num += 1
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)

        # add the figure into pdf
        pdf.add_page()
        pdf.ln()
        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
                w=figsize[0],h=figsize[1])


    if include_census:


        # read the csv for individual plots
        individual_fn = data_dir + data_pf + 'annual_individual_plot.csv'
        individual_df = pd.read_csv(individual_fn)

        pft_size_fn = data_dir + data_pf + 'annual_pft_size.csv'
        pft_size_df = pd.read_csv(pft_size_fn)

        #years to plot
        pdf.add_page()
        pdf.ln()
        year_to_plot = np.unique(individual_df['year'].values)
        year_num = len(year_to_plot)
        for iyear, year in enumerate(year_to_plot):
            figsize=(pdf.w / 2. * 0.9  ,pdf.h / 5. * 0.9)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
            sub_individual_df = individual_df[individual_df['year'].values == year]
            sub_pft_size_df = pft_size_df[pft_size_df['year'].values == year]
            plut.forest_2d_profile(ax,
                                   sub_individual_df,
                                   sub_pft_size_df,
                                   pft_list,pft_names)
            ax.set_title('{:d}'.format(year))
            if iyear == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,labels,fontsize=7,loc='upper center',
                          ncol=np.maximum(1,len(pft_list)//2),
                          frameon=False)


            fig.tight_layout()
            fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}_y{:d}.png'.format(page_num,iyear)
            plt.savefig(fig_fn,dpi=300)
            plt.close(fig)
            # add the figure into pdf
            pdf.image(fig_fn,
                      x=pdf.w*(0.05+iyear%2 * 0.9/2.),
                      y=pdf.h*(0.05+iyear//2 * 0.9/5.),
                      w=figsize[0],h=figsize[1])

        page_num += 1

    # save the pdf
    pdf.output(name=pdf_name,dest='F')



    return


# plot monthly output for a simulation and save results to a pdf
def plot_monthly(
     data_dir : 'path of the csv data'
    ,data_pf  : 'prefix of the csv data'
    ,out_dir  : 'output directory'
    ,pft_list : 'List of PFTs to include for pft VOIs'
    ,pft_names: 'Name of the pfts'
    ,include_census : 'whether to include census level measurements' = False

):
    # first generate the final pdf name
    pdf_name = out_dir + data_pf + 'monthly_diagnostics.pdf'

    # create a pdf class
    pdf = FPDF('P','in','Letter') # Portrait, inch, Letter

    # now we plot figures
    # by default we assume letter size


    # for now only plot census time scale
    page_num = 0

    # first timeseries
    # plot the average seasonality of the simulated forests
    df_fn = data_dir + data_pf + 'monthly.csv'
    data_df = pd.read_csv(df_fn)

    month_range = np.arange(1,13)

    # by default we have seasonality of 
    # MMEAN_ATM_TEMP, MMEAN_PCPG, MMEAN_ATM_RSHORT, MMEAN_ATM_VPDEF (4 atmospheric forcing)
    # MMEAN_GPP, MMEAN_NPP, MMEAN_LEAF_RESP, MMEAN_STEM_RESP, MMEAN_ROOT_RESP,
    # MMEAN_NEE, MMEAN_ET, MMEAN_SH (8 ecosystem fluxes)
    # in total 12 panels
    # 4 by 3 panels
    panel_layout = (4,3)
    figsize=(pdf.w * 0.9,pdf.h*0.9)
    voi_plots = ['MMEAN_ATM_TEMP_PY','MMEAN_PCPG_PY','MMEAN_ATM_RSHORT_PY',
                'MMEAN_ATM_VPDEF_PY','MMEAN_GPP_PY','MMEAN_NPP_PY',
                 'MMEAN_LEAF_RESP_PY','MMEAN_STEM_RESP_PY','MMEAN_ROOT_RESP_PY',
                 'MMEAN_CARBON_AC_PY','MMEAN_VAPOR_AC_PY','MMEAN_SENSIBLE_AC_PY']

    # prepare data
    data_plots = []
    for iv, voi in enumerate(voi_plots):
        voi_data = data_df[voi].values
        if voi == 'MMEAN_CARBON_AC_PY':
            # CARBON_AC_PY is in umol/m2/s
            voi_data *= ( 1e-6 * 12 / 1000 * 86400 * 365) # convert to kgC/m2/yr
        elif voi == 'MMEAN_PCPG_PY':
            # mm/s
            voi_data *= (86400 * 30.) # convert to mm/month
        elif voi == 'MMEAN_ATM_TEMP_PY':
            voi_data -= 273.15 # convert from K to degC
        elif voi == 'MMEAN_VAPOR_AC_PY':
            voi_data *= (-1 * 86400 * 30. ) # convert to mm/month
        elif voi == 'MMEAN_TRANSP_PY':
            voi_data *= (86400 * 30.) # convert to mm/month
        elif voi == 'MMEAN_SENSIBLE_AC_PY':
            voi_data *= (-1.)



        month_data = np.zeros((len(month_range),),dtype=np.float)
        # loop over month to get the average
        for imonth, month in enumerate(month_range):
            data_mask = (data_df['month'].values.astype(int) == month)
            month_data[imonth] = np.nanmean(voi_data[data_mask])

        data_plots.append(month_data)

    voi_labels = ['Tair [degC]','P [mm/month]','Rshort [W/m2]','VPD [Pa]',
                  'GPP [kgC/m2/yr]','NPP [kgC/m2/yr]','Rleaf [kgC/m2/yr]',
                  'Rstem [kgC/m2/yr]','Rroot [kgC/m2/yr]','NEE [kgC/m2/yr]',
                  'ET [mm/month]','SH [W/m2]']

    fig,axes = plt.subplots(panel_layout[0],panel_layout[1],figsize=figsize,sharex=True)
    for i, ax in enumerate(axes.ravel()):

        ax.plot(month_range,data_plots[i],'k--o',lw=2)
        ax.set_xlim(0,13)
        ax.set_ylabel(voi_labels[i])
    
    # save the page
    fig.tight_layout()
    fig_fn = out_dir + data_pf + 'month_diagnostics_p{:d}.png'.format(page_num)
    page_num += 1
    plt.savefig(fig_fn,dpi=300)
    plt.close(fig)



    # add the figure into pdf
    pdf.add_page()
    pdf.ln()
    pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
            w=figsize[0],h=figsize[1])


    # MMEAN_LAI by PFT, MMEAN_GPP by PFT, MMEAN_NPPDAILY by PFT (3)
    # MMEAN_LAI by DBH, MMEAN_GPP by DBH, MMEAN_NPPDAILY by DBH (3)
    # MMEAN_LAI by HITE, MMEAN_GPP by HITE, MMEAN_NPPDAILY by HITE (3)
    # 3 figures each one has 3 panels


#    # we will plot all of them in 1 page
#    # each panel has 5'' wide and ~ 1 inch in height (too condensed?)
#    # if there are more, we split them into different pages
    time_unit='month'
    voi_plots = ['MMEAN_LAI','MMEAN_GPP']
    voi_names = ['LAI','GPP',]
    voi_units = ['m2/m2','kgC/m2/yr']

    # there are 3 by 3 panels
    figsize=(pdf.w * 0.9,pdf.h*0.9)
    fig,axes = plt.subplots(3,2,figsize=figsize,sharex=True)
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            # each column has the same voi
            # each row shows different separation

            voi_plot = voi_plots[j]
            xlabel_on = True

            if i == 0:
                # separate by PFT
                pft_list_plot = np.array(pft_list)
                pft_names_plot = np.array(pft_names)
                pft_colors_plot = np.array(PFT_COLORS[np.array(pft_list_plot).astype(int)])

                # prepare data
                # aggregate data to PFT average
                var_data = np.zeros((len(pft_list_plot),
                                        len(month_range)))

                for ipft, pft in enumerate(pft_list_plot):
                    col_name = '{:s}_PFT_{:d}'.format(
                        voi_plot,pft)
                    for imonth, month in enumerate(month_range):
                        data_mask = (data_df['month'].values.astype(int) == month)
            
                        var_data[ipft,imonth] = np.nanmean(data_df[col_name].values[data_mask])



                plut.area_timeseries(ax,month_range,time_unit,
                                var_data,voi_names[j],voi_units[j],
                                var_labels=pft_names_plot,
                                xlabel_on=xlabel_on,
                                color_list=pft_colors_plot)
            elif i == 1:
                # separate by DBH
                size_names = []
                for isize, size_edge in enumerate(dbh_size_list[1]):
                    if size_edge == dbh_size_list[1][-1]:
                        # last one
                        size_names.append('{:d} < D'.format(size_edge))
                    else:
                        size_names.append('{:d} < D <= {:d}'.format(size_edge,
                                                                dbh_size_list[1][isize+1]))
                size_colors_plot = np.array(PFT_COLORS[1:len(size_names)+1])

                # prepare data
                var_data = np.zeros((len(dbh_size_list[1]),
                                    len(month_range)))

                for isize, size_edge in enumerate(dbh_size_list[1]):
                    col_name = '{:s}_{:s}_{:d}'.format(
                        voi_plot,dbh_size_list[0],isize)
                    for imonth, month in enumerate(month_range):
                        data_mask = (data_df['month'].values.astype(int) == month)
                        var_data[isize,imonth] = np.nanmean(data_df[col_name].values[data_mask])


                plut.area_timeseries(ax,month_range,time_unit,
                                var_data,voi_names[j],voi_units[j],
                                var_labels=size_names,
                                xlabel_on=xlabel_on,
                                color_list=size_colors_plot)
            elif i == 2:
                # separate by height
                size_names = []
                for isize, size_edge in enumerate(hite_size_list[1]):
                    if size_edge == hite_size_list[1][-1]:
                        # last one
                        size_names.append('{:4.2f} < H'.format(size_edge))
                    else:
                        size_names.append('{:4.2f} < H <= {:4.2f}'.format(size_edge,
                                                                hite_size_list[1][isize+1]))
                size_colors_plot = np.array(PFT_COLORS[1:len(size_names)+1])

                # prepare data
                var_data = np.zeros((len(hite_size_list[1]),
                                    len(month_range)))

                for isize, size_edge in enumerate(hite_size_list[1]):
                    col_name = '{:s}_{:s}_{:d}'.format(
                        voi_plot,hite_size_list[0],isize)
                    for imonth, month in enumerate(month_range):
                        data_mask = (data_df['month'].values.astype(int) == month)
                        var_data[isize,imonth] = np.nanmean(data_df[col_name].values[data_mask])


                plut.area_timeseries(ax,month_range,time_unit,
                                var_data,voi_names[j],voi_units[j],
                                var_labels=size_names,
                                xlabel_on=xlabel_on,
                                color_list=size_colors_plot)



            if j == 0:
                # include legends if it is the first column
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,labels,fontsize=7,loc='upper left',
                            ncol=np.maximum(1,len(pft_list)//3),
                            frameon=False)
        
    # save the page
    fig.tight_layout()
    fig_fn = out_dir + data_pf + 'monthly_diagnostics_p{:d}.png'.format(page_num)
    page_num += 1
    plt.savefig(fig_fn,dpi=300)
    plt.close(fig)



    # add the figure into pdf
    pdf.add_page()
    pdf.ln()
    pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
            w=figsize[0],h=figsize[1])



#    # plot size distributions
#    voi_size = ['AGB','BA']
#
#    # read csv for size plots
#    # read the csv for individual plots
#    df_fn = data_dir + data_pf + 'monthly_pft_size.csv'
#    csv_df = pd.read_csv(df_fn)
#
#    #years to plot
#    year_to_plot = np.unique(csv_df['year'].values).astype(int)
#    year_num = len(year_to_plot)
#
#    # first loop over vars
#    for ivar, var_name in enumerate(voi_size):
#        # create a figure
#        figsize = (pdf.w * 0.9,pdf.h * 0.9)
#        fig,axes = plt.subplots(year_num,2,figsize=figsize,sharey=True)
#        for i, row in enumerate(axes):
#            # get subset of data
#            for j, ax in enumerate(row):
#                if j == 0:
#                    ylabel_on = True
#                    size_list = ('D',np.arange(0.,200.+1.,10.))
#                elif j == 1:
#                    ylabel_on = False
#                    size_list = ('H',np.arange(0.,50.+1.,5.))
#
#                bar_size(ax,size_list,csv_df,i,var_name,voi_unit_dict[var_name],
#                         pft_list,pft_names,ylabel_on=ylabel_on)
#
#                if i == 0 and j == 0:
#                    handles, labels = ax.get_legend_handles_labels()
#                    ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
#                              frameon=False)
#
#                # plot the year
#                if j == 0:
#                    ax.set_title('{:d}'.format(year_to_plot[i]))
#       
#        # save the page
#        fig.tight_layout()
#        fig_fn = out_dir + data_pf + 'monthly_diagnostics_p{:d}.png'.format(page_num)
#        page_num += 1
#        plt.savefig(fig_fn,dpi=300)
#        plt.close(fig)
#
#        # add the figure into pdf
#        pdf.add_page()
#        pdf.ln()
#        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
#              w=figsize[0],h=figsize[1])
#
#
#


    page_num += 1
    pdf.add_page()
    pdf.ln()

    # read the csv for individual plots
    df_fn = data_dir + data_pf + 'monthly_individual.csv'
    individual_df = pd.read_csv(df_fn)

    df_fn = data_dir + data_pf + 'monthly_pft_size.csv'
    pft_size_df = pd.read_csv(df_fn)

    #years to plot
    year_to_plot = np.unique(individual_df['year'].values).astype(int)
    year_num = len(year_to_plot)
    for iyear, year in enumerate(year_to_plot):
        figsize=(pdf.w / 2. * 0.9  ,pdf.h / 5. * 0.9)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        sub_individual_df = individual_df[individual_df['year'].values == year]
        sub_pft_size_df = pft_size_df[pft_size_df['year'].values == year]
        plut.forest_2d_profile(ax,
                               sub_individual_df,
                               pft_size_df,
                               pft_list,pft_names)
        ax.set_title('{:d}'.format(year))
        if iyear == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,labels,fontsize=7,loc='upper center',ncol=len(pft_list)//2,
                      frameon=False)


        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'monthly_diagnostics_p{:d}_y{:d}.png'.format(page_num,iyear)
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)
        # add the figure into pdf
        pdf.image(fig_fn,
                  x=pdf.w*(0.05+iyear%2 * 0.9/2.),
                  y=pdf.h*(0.05+iyear//2 * 0.9/5.),
                  w=figsize[0],h=figsize[1])


    # finally growth-size and mortality-size

    # read the csv for demography
    df_fn = data_dir + data_pf + 'monthly_demography.csv'
    demography_df = pd.read_csv(df_fn)

    year_to_plot = np.unique(demography_df['census_yeara'].values).astype(int)
    year_num = len(year_to_plot)
    print(year_to_plot)
    print(year_num)

    # growth
    page_num += 1
    pdf.add_page()
    pdf.ln()
    for iyear, year in enumerate(year_to_plot):
        figsize=(pdf.w / 2. * 0.9  ,pdf.h / 5. * 0.9)

        # first plot relationship against dbh
        x_name = 'DBH_init'
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        sub_df = demography_df[demography_df['census_yeara'].values == year]
        # plot growth
        for ipft, pft in enumerate(pft_list):
            pft_mask = sub_df['PFT_init'].values == pft
            if np.sum(pft_mask) == 0:
                continue

            # prepare data
            x_data = sub_df[x_name].values[pft_mask]
            y_data = sub_df['DDBH_DT_avg'].values[pft_mask]
            scale_data = sub_df['NPLANT_init'].values[pft_mask]
            scale_data = np.log10(scale_data / np.nanmin(scale_data)) # relative abundance
            scale_data = 15. + np.minimum(20.,scale_data * 5.)
            color_data = PFT_COLORS[pft]

            ax.scatter(x_data,y_data,s=scale_data,c=color_data)

        ax.set_xlabel(x_name)
        ax.set_ylabel('DBH_DT [cm]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((1.,150.))
        ax.set_ylim((1e-2,2.5))

        if iyear == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
                        frameon=False)

        ax.set_title('{:d}'.format(year))


        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'monthly_diagnostics_p{:d}_y{:d}_dbh.png'.format(page_num,iyear)
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)
        # add the figure into pdf
        pdf.image(fig_fn,
                  x=pdf.w*(0.05),
                  y=pdf.h*(0.05+iyear * 0.9 / 5.),
                  w=figsize[0],h=figsize[1])


        # relationship against hite
        x_name = 'HITE_init'
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        sub_df = demography_df[demography_df['census_yeara'].values == year]
        # plot growth
        for ipft, pft in enumerate(pft_list):
            pft_mask = sub_df['PFT_init'].values == pft
            if np.sum(pft_mask) == 0:
                continue

            # prepare data
            x_data = sub_df[x_name].values[pft_mask]
            y_data = sub_df['DDBH_DT_avg'].values[pft_mask]
            scale_data = sub_df['NPLANT_init'].values[pft_mask]
            scale_data = np.log10(scale_data / np.nanmin(scale_data)) # relative abundance
            scale_data = 15. + np.minimum(20.,scale_data * 5.)
            color_data = PFT_COLORS[pft]

            ax.scatter(x_data,y_data,s=scale_data,c=color_data)

        ax.set_xlabel(x_name)
        ax.set_ylabel('DBH_DT [cm]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((1.,70))
        ax.set_ylim((1e-2,2.5))

        if iyear == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
                        frameon=False)

        ax.set_title('{:d}'.format(year))


        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'monthly_diagnostics_p{:d}_y{:d}_hite.png'.format(page_num,iyear)
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)
        # add the figure into pdf
        pdf.image(fig_fn,
                  x=pdf.w*(0.05+0.9/2.),
                  y=pdf.h*(0.05+iyear * 0.9 / 5.),
                  w=figsize[0],h=figsize[1])


    # mortality
    page_num += 1
    pdf.add_page()
    pdf.ln()
    for iyear, year in enumerate(year_to_plot):
        figsize=(pdf.w / 2. * 0.9  ,pdf.h / 5. * 0.9)

        # first plot relationship against dbh
        x_name = 'DBH_init'
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        sub_df = demography_df[demography_df['census_yeara'].values == year]
        # plot growth
        for ipft, pft in enumerate(pft_list):
            pft_mask = sub_df['PFT_init'].values == pft
            if np.sum(pft_mask) == 0:
                continue

            # prepare data
            x_data = sub_df[x_name].values[pft_mask]
            y_data = sub_df['MORT_RATE_avg'].values[pft_mask]
            scale_data = sub_df['NPLANT_init'].values[pft_mask]
            scale_data = np.log10(scale_data / np.nanmin(scale_data)) # relative abundance
            scale_data = 15. + np.minimum(20.,scale_data * 5.)
            color_data = PFT_COLORS[pft]

            ax.scatter(x_data,y_data,s=scale_data,c=color_data)

        ax.set_xlabel(x_name)
        ax.set_ylabel('Mort per yr')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((1.,150))
        ax.set_ylim((0.002,1.00))

        if iyear == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
                        frameon=False)

        ax.set_title('{:d}'.format(year))


        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'monthly_diagnostics_p{:d}_y{:d}_dbh.png'.format(page_num,iyear)
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)
        # add the figure into pdf
        pdf.image(fig_fn,
                  x=pdf.w*(0.05),
                  y=pdf.h*(0.05+iyear * 0.9 / 5.),
                  w=figsize[0],h=figsize[1])


        # relationship against hite
        x_name = 'HITE_init'
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        sub_df = demography_df[demography_df['census_yeara'].values == year]
        # plot growth
        for ipft, pft in enumerate(pft_list):
            pft_mask = sub_df['PFT_init'].values == pft
            if np.sum(pft_mask) == 0:
                continue

            # prepare data
            x_data = sub_df[x_name].values[pft_mask]
            y_data = sub_df['MORT_RATE_avg'].values[pft_mask] 
            scale_data = sub_df['NPLANT_init'].values[pft_mask]
            scale_data = np.log10(scale_data / np.nanmin(scale_data)) # relative abundance
            scale_data = 15. + np.minimum(20.,scale_data * 5.)
            color_data = PFT_COLORS[pft]

            ax.scatter(x_data,y_data,s=scale_data,c=color_data)

        ax.set_xlabel(x_name)
        ax.set_ylabel('Mort per year')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((1.,70))
        ax.set_ylim((0.002,1.00))

        if iyear == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
                        frameon=False)

        ax.set_title('{:d}'.format(year))


        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'monthly_diagnostics_p{:d}_y{:d}_hite.png'.format(page_num,iyear)
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)
        # add the figure into pdf
        pdf.image(fig_fn,
                  x=pdf.w*(0.05+0.9/2.),
                  y=pdf.h*(0.05+iyear * 0.9 / 5.),
                  w=figsize[0],h=figsize[1])

        

#    # first timeseries
#    # by default we have 5 variables to plot
#    # we will plot all of them in 1 page
#    # each panel has 5'' wide and ~ 1 inch in height (too condensed?)
#    # if there are more, we split them into different pages
#    voi_pft = ['AGB','LAI','BA10','NPLANT10']
#    voi_size = ['AGB','LAI','BA','NPLANT']
#    time_unit = 'year'
#
#    # read the csv
#    df_fn = data_dir + data_pf + 'annual.csv'
#    data_df = pd.read_csv(df_fn)
#    time_data = np.unique(data_df['year'].values.astype(int))
#
#    split_num = int(np.ceil(float(len(voi_pft)) / 5.))
#
#    page_num = 1
#    # loop over split_num
#    for isp in np.arange(split_num):
#        # get # of vars in this category
#        var_num = min(5,(len(voi_pft) - isp * 5))
#        figsize=(pdf.w * 0.9,pdf.h*0.9)
#        fig,axes = plt.subplots(var_num,1,figsize=figsize,sharex=True)
#        for i, ax in enumerate(axes):
#            if i == var_num - 1:
#                # last one
#                xlabel_on = True
#            else:
#                xlabel_on = False
#            voi_plot = voi_pft[isp*5+i]
#
#            # for PFT
#            # first prepare data
#            if voi_plot in ['BA','BA10','NPLANT','NPLANT10']:
#                # do not include grasses
#                pft_mask = (np.array(pft_list) != 1)
#            else:
#                pft_mask = (np.array(pft_list) > 0)
#
#            pft_list_plot = np.array(pft_list)[pft_mask]
#            pft_names_plot = np.array(pft_names)[pft_mask]
#            pft_colors_plot = np.array(PFT_COLORS[0:len(pft_list)])[pft_mask]
#
#            # prepare data
#            # aggregate data to PFT average
#            var_data = np.zeros((len(pft_list_plot),
#                                 len(time_data)))
#
#            if voi_plot in ['NPLANT','NPLANT10']:
#                var_scaler = 1.e4
#            else:
#                var_scaler = 1.
#
#            if voi_plot in ['BA10','NPLANT10']:
#                voi_extract = voi_plot[0:-2]
#            else:
#                voi_extract = voi_plot
#
#            for ipft, pft in enumerate(pft_list_plot):
#                for isize, size_edge in enumerate(dbh_size_list[1]):
#                    col_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(
#                        voi_extract,pft,dbh_size_list[0],isize)
#
#                    if ((voi_plot in ['BA10','NPLANT10']) and
#                        isize == 0
#                       ):
#                        # ignore cohorts with DBH < 10cm
#                        continue
#            
#                    temp_data = data_df[col_name].values * var_scaler
#                    temp_data[np.isnan(temp_data)] = 0.
#                    var_data[ipft,:] += temp_data
#
#
#
#            area_timeseries(ax,time_data,time_unit,
#                            var_data,voi_plot,voi_unit_dict[voi_plot],
#                            var_labels=pft_names_plot,
#                            xlabel_on=xlabel_on,
#                            color_list=pft_colors_plot)
#
#            if i == 0:
#                handles, labels = ax.get_legend_handles_labels()
#                ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
#                          frameon=False)
#       
#        # save the page
#        fig.tight_layout()
#        fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
#        page_num += 1
#        plt.savefig(fig_fn,dpi=300)
#        plt.close(fig)
#
#
#
#        # add the figure into pdf
#        pdf.add_page()
#        pdf.ln()
#        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
#              w=figsize[0],h=figsize[1])
#
#        
#        #
#        # dbh size distributions
#        size_names = []
#        for isize, size_edge in enumerate(dbh_size_list[1]):
#            if size_edge == dbh_size_list[1][-1]:
#                # last one
#                size_names.append('{:d} < D'.format(size_edge))
#            else:
#                size_names.append('{:d} < D <= {:d}'.format(size_edge,
#                                                           dbh_size_list[1][isize+1]))
#        size_colors_plot = np.array(PFT_COLORS[0:len(size_names)])
#
#        fig,axes = plt.subplots(var_num,1,figsize=figsize,sharex=True)
#        for i, ax in enumerate(axes):
#            if i == var_num - 1:
#                # last one
#                xlabel_on = True
#            else:
#                xlabel_on = False
#            voi_plot = voi_size[isp*5+i]
#
#
#            # first prepare data
#            if voi_plot in ['BA','BA10','NPLANT','NPLANT10']:
#                # do not include grasses
#                pft_mask = (np.array(pft_list) != 1)
#            else:
#                pft_mask = (np.array(pft_list) > 0)
#
#            pft_list_plot = np.array(pft_list)[pft_mask]
#            pft_names_plot = np.array(pft_names)[pft_mask]
#            pft_colors_plot = np.array(PFT_COLORS[0:len(pft_list)])[pft_mask]
#
#            # prepare data
#            # aggregate data to PFT average
#            var_data = np.zeros((len(dbh_size_list[1]),
#                                 len(time_data)))
#
#            if voi_plot in ['NPLANT','NPLANT10']:
#                var_scaler = 1.e4
#            else:
#                var_scaler = 1.
#
#            if voi_plot in ['BA10','NPLANT10']:
#                voi_extract = voi_plot[0:-2]
#            else:
#                voi_extract = voi_plot
#
#
#            for ipft, pft in enumerate(pft_list_plot):
#                for isize, size_edge in enumerate(dbh_size_list[1]):
#                    col_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(
#                        voi_extract,pft,dbh_size_list[0],isize)
#
#                    temp_data = data_df[col_name].values * var_scaler
#                    temp_data[np.isnan(temp_data)] = 0.
#                    var_data[isize,:] += temp_data
#
#
#
#            area_timeseries(ax,time_data,time_unit,
#                            var_data,voi_plot,voi_unit_dict[voi_plot],
#                            var_labels=size_names,
#                            xlabel_on=xlabel_on,
#                            color_list=size_colors_plot)
#
#            if i == 0:
#                handles, labels = ax.get_legend_handles_labels()
#                ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
#                          frameon=False)
#       
#        # save the page
#        fig.tight_layout()
#        fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
#        page_num += 1
#        plt.savefig(fig_fn,dpi=300)
#        plt.close(fig)
#
#        # add the figure into pdf
#        pdf.add_page()
#        pdf.ln()
#        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
#              w=figsize[0],h=figsize[1])
#
#        # height size distributions
#        size_names = []
#        for isize, size_edge in enumerate(hite_size_list[1]):
#            if size_edge == hite_size_list[1][-1]:
#                # last one
#                size_names.append('{:4.2f} < H'.format(size_edge))
#            else:
#                size_names.append('{:4.2f} < H <= {:4.2f}'.format(
#                    size_edge,hite_size_list[1][isize+1]))
#
#        size_colors_plot = np.array(PFT_COLORS[0:len(size_names)])
#
#        fig,axes = plt.subplots(var_num,1,figsize=figsize,sharex=True)
#        for i, ax in enumerate(axes):
#            if i == var_num - 1:
#                # last one
#                xlabel_on = True
#            else:
#                xlabel_on = False
#            voi_plot = voi_size[isp*5+i]
#
#            # first prepare data
#            if voi_plot in ['BA','BA10','NPLANT','NPLANT10']:
#                # do not include grasses
#                pft_mask = (np.array(pft_list) != 1)
#            else:
#                pft_mask = (np.array(pft_list) > 0)
#
#            pft_list_plot = np.array(pft_list)[pft_mask]
#            pft_names_plot = np.array(pft_names)[pft_mask]
#            pft_colors_plot = np.array(PFT_COLORS[0:len(pft_list)])[pft_mask]
#
#            # prepare data
#            # aggregate data to PFT average
#            var_data = np.zeros((len(hite_size_list[1]),
#                                 len(time_data)))
#
#            if voi_plot in ['NPLANT','NPLANT10']:
#                var_scaler = 1.e4
#            else:
#                var_scaler = 1.
#
#            if voi_plot in ['BA10','NPLANT10']:
#                voi_extract = voi_plot[0:-2]
#            else:
#                voi_extract = voi_plot
#
#
#            for ipft, pft in enumerate(pft_list_plot):
#                for isize, size_edge in enumerate(hite_size_list[1]):
#                    col_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(
#                        voi_extract,pft,hite_size_list[0],isize)
#
#                    temp_data = data_df[col_name].values * var_scaler
#                    temp_data[np.isnan(temp_data)] = 0.
#                    var_data[isize,:] += temp_data
#
#
#
#            area_timeseries(ax,time_data,time_unit,
#                            var_data,voi_plot,voi_unit_dict[voi_plot],
#                            var_labels=size_names,
#                            xlabel_on=xlabel_on,
#                            color_list=size_colors_plot)
#
#            if i == 0:
#                handles, labels = ax.get_legend_handles_labels()
#                ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
#                          frameon=False)
#       
#        # save the page
#        fig.tight_layout()
#        fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
#        page_num += 1
#        plt.savefig(fig_fn,dpi=300)
#        plt.close(fig)
#
#        # add the figure into pdf
#        pdf.add_page()
#        pdf.ln()
#        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
#              w=figsize[0],h=figsize[1])
#
#    
#    # plot demography rates, changes with PFT and Size
#    # read  csv
#    df_fn = data_dir + data_pf + 'annual_demography.csv'
#    csv_df = pd.read_csv(df_fn)
#    #years to plot # only plot 3 years, first, middle, last
#    yeara_plot = np.unique(csv_df['yeara'].values)
#    yearz_plot = np.unique(csv_df['yearz'].values)
#    year_num = len(yeara_plot)
#    if year_num > 2:
#        yeara_plot = yeara_plot[[0,year_num//2,year_num-1]]
#        yearz_plot = yearz_plot[[0,year_num//2,year_num-1]]
#
#    year_num = len(yeara_plot)
#
#    figsize = (pdf.w * 0.9,pdf.h * 0.9)
#    fig,axes = plt.subplots(year_num,2,figsize=figsize,sharey=True,sharex=True)
#    for i, row in enumerate(axes):
#        # year to plot
#        yeara = yeara_plot[i]
#        yearz = yearz_plot[i]
#        # get subset of data
#        for j, ax in enumerate(row):
#            if j == 0:
#                ylabel='BA GROWTH [%]'
#                y_var = 'BA_GROWTH_FRAC' 
#            elif j == 1:
#                ylabel='BA Mortality [%]'
#                y_var = 'BA_MORT_FRAC' 
#
#            # first prepare data
#            sub_df = csv_df[(csv_df['yeara'] == yeara) & (csv_df['yearz'] == yearz)]
#            demo_pft = pft_list[:]
#            demo_pft_names = pft_names[:]
#            if type(demo_pft).__module__ != np.__name__:
#                demo_pft = np.array(demo_pft)
#
#            if type(demo_pft_names).__module__ != np.__name__:
#                demo_pft_names = np.array(demo_pft_names)
#
#            demo_pft_names = demo_pft_names[demo_pft != 1] # exclude grasses
#            demo_pft = demo_pft[demo_pft != 1] # exclude grasses
#            
#            demo_data = np.zeros((len(dbh_size_list[1]),len(demo_pft)))
#
#            for ipft, pft in enumerate(demo_pft):
#                for isize, size_edge in enumerate(dbh_size_list[1]):
#                    var_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(
#                        y_var,pft,dbh_size_list[0],isize)
#                    demo_data[isize,ipft] = sub_df[var_name].values[0] * 100.
#
#            x_data = np.zeros((len(dbh_size_list[1]),))
#            for isize, size_edge in enumerate(dbh_size_list[1]):
#                if size_edge == dbh_size_list[1][-1]:
#                    # last one
#                    x_data[isize] = (dbh_size_list[1][isize] + 
#                                     (dbh_size_list[1][isize] - dbh_size_list[1][isize-1])
#                                     / 2)
#                else:
#                    x_data[isize] = (dbh_size_list[1][isize]
#                                    +dbh_size_list[1][isize+1]) / 2
#
#            # plot the data
#            for ipft, pft in enumerate(demo_pft):
#                ax.plot(x_data,demo_data[:,ipft],lw=2,marker='o',color=PFT_COLORS[pft-1],
#                        label=demo_pft_names[ipft])
#
#            # deal with axes and title
#            ax.set_xticks(dbh_size_list[1])
#            ax.set_xlabel('DBH [cm]',fontsize=8)
#            ax.set_ylabel(ylabel,fontsize=8)
#            ax.set_title('{:d}-{:d}'.format(yeara,yearz),fontsize=8)
#
#            # legend
#            if i == 0 and j == 0:
#                handles, labels = ax.get_legend_handles_labels()
#                ax.legend(handles,labels,fontsize=7,loc='upper right',ncol=len(pft_list)//3,
#                          frameon=False)
#    # save the page
#    fig.tight_layout()
#    fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
#    page_num += 1
#    plt.savefig(fig_fn,dpi=300)
#    plt.close(fig)
#
#    # add the figure into pdf
#    pdf.add_page()
#    pdf.ln()
#    pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
#          w=figsize[0],h=figsize[1])




    # save the pdf
    pdf.output(name=pdf_name,dest='F')



    return


# plot monthly diurnal output for a simulation and save results to a pdf
def plot_monthly_diurnal(
     data_dir : 'path of the csv data'
    ,data_pf  : 'prefix of the csv data'
    ,out_dir  : 'output directory'
):
    '''
        For now, only include monthly average diurnal cycle at polygon level
    '''
    # first generate the final pdf name
    pdf_name = out_dir + data_pf + 'monthly_diurnal_diagnostics.pdf'

    # create a pdf class
    pdf = FPDF('P','in','Letter') # Portrait, inch, Letter

    # now we plot figures
    # by default we assume letter size


    # for now only plot census time scale
    page_num = 0

    # first timeseries
    # plot the average seasonality of the simulated forests
    df_fn = data_dir + data_pf + 'qmean_polygon.csv'
    data_df = pd.read_csv(df_fn)

    month_range = np.arange(1,13)
    tod_range = np.unique(data_df['hour'].values) # time of day

    # each page has 2 x 6 panels, each panel showing the average of one month
    page_vars = [
                ['QMEAN_ATM_TEMP_PY','QMEAN_CAN_TEMP_PY'],
                ['QMEAN_ATM_VPDEF_PY','QMEAN_CAN_VPDEF_PY'],
                ['QMEAN_ATM_RSHORT_PY','QMEAN_ATM_PAR_PY'],
                ['QMEAN_GPP_PY','QMEAN_NPP_PY','QMEAN_PLRESP_PY','QMEAN_NEP_PY'],
                ['QMEAN_LEAF_RESP_PY','QMEAN_STEM_RESP_PY','QMEAN_ROOT_RESP_PY'],
                ['QMEAN_TRANSP_PY']
                ]
    page_units = ['degC','kPa','W/m2','kgC/m2/yr','kgC/m2/yr','mm/yr']

    panel_layout = (6,2)
    color_list=['k','r','g','b']

    for ipage, voi_plots in enumerate(page_vars):

        figsize=(pdf.w * 0.9,pdf.h*0.9)
        # generate var names for legend
        var_names = [var_name[6:len(var_name)-3] for var_name in voi_plots]

        # prepare data
        data_plots = []
        for iv, voi in enumerate(voi_plots):
            voi_data = np.zeros((len(month_range),len(tod_range),),dtype=float)
            for imonth,month in enumerate(month_range):
                for itime,time in enumerate(tod_range):
                    data_mask = (
                        (data_df['month'].values == month) & 
                        (data_df['hour'].values == time))
                    voi_data[imonth,itime] = np.nanmean(data_df[voi].values[data_mask])
            # convert unit
            if page_units[ipage] == 'degC':
                voi_data -= 273.15
            elif page_units[ipage] == 'kPa':
                voi_data /= 1000.
            elif page_units[ipage] == 'mm/yr':
                voi_data *= (86400. * 365.)
        
            data_plots.append(voi_data)

        fig,axes = plt.subplots(panel_layout[0],panel_layout[1],figsize=figsize,
                                sharex=True,sharey=True)
        for i, ax in enumerate(axes.ravel()):
            for iv, voi in enumerate(voi_plots):
                ax.plot(tod_range,data_plots[iv][i,:],
                        c=color_list[iv],ls='-',lw=2,
                        label=var_names[iv])

            ax.set_ylabel(page_units[ipage])
            ax.set_title('Month {:d}'.format(month_range[i]))
            # set legend
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,labels,fontsize=7,loc='upper left',
                            ncol=np.maximum(1,len(handles)//3),
                            frameon=False)

        # save the page
        fig.tight_layout()
        fig_fn = out_dir + data_pf + 'qmean_diagnostics_p{:d}.png'.format(page_num)
        page_num += 1
        plt.savefig(fig_fn,dpi=300)
        plt.close(fig)



        # add the figure into pdf
        pdf.add_page()
        pdf.ln()
        pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
                w=figsize[0],h=figsize[1])


    # save the pdf
    pdf.output(name=pdf_name,dest='F')



    return
