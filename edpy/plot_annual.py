# High-level plotting functions

# import modules
import plot_utils as plut
from plot_utils import PFT_COLORS, voi_unit_dict
from extract_utils import dbh_size_list,hite_size_list
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
        size_colors_plot = np.array(PFT_COLORS[0:len(size_names)])

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

        size_colors_plot = np.array(PFT_COLORS[0:len(size_names)])

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

#    # add the figure into pdf
#    pdf.add_page()
#    pdf.ln()
#    pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
#          w=figsize[0],h=figsize[1])


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

    # create a subset of dataframe to plot
    # first loop over vars
    for ivar, var_name in enumerate(voi_size):
        # create a figure
        figsize = (pdf.w * 0.9,pdf.h * 0.9 * patch_num / 4.)
        fig,axes = plt.subplots(patch_num,2,figsize=figsize,sharey=True)
        for i, row in enumerate(axes):
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
#        # plot size distributions
#        voi_size = ['AGB','LAI','BA','NPLANT']
#
#        # read csv for size plots
#        # read the csv for individual plots
#        df_fn = data_dir + data_pf + 'annual_pft_size.csv'
#        csv_df = pd.read_csv(df_fn)
#
#        #years to plot
#        year_to_plot = np.unique(csv_df['year'].values)
#        year_num = len(year_to_plot)
#
#        # first loop over vars
#        for ivar, var_name in enumerate(voi_size):
#            # create a figure
#            figsize = (pdf.w * 0.9,pdf.h * 0.9)
#            fig,axes = plt.subplots(year_num,2,figsize=figsize,sharey=True)
#            for i, row in enumerate(axes):
#                # get subset of data
#                for j, ax in enumerate(row):
#                    if j == 0:
#                        ylabel_on = True
#                        size_list = ('D',np.arange(0.,200.+1.,10.))
#                    elif j == 1:
#                        ylabel_on = False
#                        size_list = ('H',np.arange(0.,50.+1.,5.))
#
#                    bar_size(ax,size_list,csv_df,i,var_name,voi_unit_dict[var_name],
#                             pft_list,pft_names,ylabel_on=ylabel_on)
#
#                    if i == 0 and j == 0:
#                        handles, labels = ax.get_legend_handles_labels()
#                        ax.legend(handles,labels,fontsize=7,loc='upper left',ncol=len(pft_list)//3,
#                                  frameon=False)
#
#                    # plot the year
#                    if j == 0:
#                        ax.set_title('{:d}'.format(year_to_plot[i]))
#           
#            # save the page
#            fig.tight_layout()
#            fig_fn = out_dir + data_pf + 'annual_diagnostics_p{:d}.png'.format(page_num)
#            page_num += 1
#            plt.savefig(fig_fn,dpi=300)
#            plt.close(fig)
#
#            # add the figure into pdf
#            pdf.add_page()
#            pdf.ln()
#            pdf.image(fig_fn,x=pdf.w*0.05,y=pdf.h*0.05,
#                  w=figsize[0],h=figsize[1])



        # read the csv for individual plots
        df_fn = data_dir + data_pf + 'annual_individual_plot.csv'
        individual_df = pd.read_csv(df_fn)

        #years to plot
        pdf.add_page()
        pdf.ln()
        year_to_plot = np.unique(individual_df['year'].values)
        year_num = len(year_to_plot)
        for iyear, year in enumerate(year_to_plot):
            figsize=(pdf.w / 2. * 0.9  ,pdf.h / 5. * 0.9)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1,1,1)
            sub_df = individual_df[individual_df['year'].values == year]
            plut.forest_2d_profile(ax,sub_df,pft_list,pft_names)
            ax.set_title('{:d}'.format(year))
            if iyear == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,labels,fontsize=7,loc='upper left',
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

