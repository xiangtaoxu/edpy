# This file contains function for low level plotting

# import modules here
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg') # allow ploting without X11
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection

from .extract_utils import dbh_size_list,hite_size_list


#########################################
# Several util functions for usual ED2 result plots
##########################################

# some constants
# don't use black since black is reserved to represent stem
# the first one is served as a place holder since the index starts from 0 but PFT numbers start from
# 1
PFT_COLORS = np.array([
    'null',
    'xkcd:green',
    'xkcd:cyan',
    'xkcd:olive',
    'xkcd:red',
    'xkcd:blue',
    'xkcd:brown',
    'xkcd:gold',
    'xkcd:lilac'])

voi_unit_dict = {
    'AGB' : 'kgC/m2',
    'BA'  : 'cm2/m2',
    'BA10'  : 'cm2/m2',
    'LAI' : 'm2/m2',
    'NPLANT' : '#/ha',
    'NPLANT10' : '#/ha'
}

def plot_profile(
    ax : 'axis handle for the plot',
    size_list : 'Size bins for the profile',
    pft_list  : 'PFTs to include in the plot',
    weight_matrix : 'weight of each size bin'):
    '''
        Plot the profile of the ED2 results using the given size_lsit and
        weight_list
    '''

    # first quality control, weight_matrix should have the first dimension the
    # same as size_list and the second dimension the same as pft_list
    weight_matrix = np.array(weight_matrix)
    pft_list = np.array(pft_list)
    size_list = np.array(size_list)
    if not (weight_matrix.shape[0] == len(size_list) and
            weight_matrix.shape[1] == len(pft_list)):
        print('Weight_matrix must have the dimensions as (len(size_list),len(pft_list))')
        print('Skip Printing!')
        return -1

    # get size_step from size_list
    size_step = np.zeros_like(size_list)
    size_step[0:len(size_step)-1] = size_list[1::] - size_list[0:len(size_list)-1]
    size_step[-1] = size_step[-2]

    # Loop over different PFTs using bar plot
    for ipft, pft in enumerate(pft_list):
        ax.barh(size_list,weight_matrix[:,ipft],
                height=size_step,left=0.,align='edge',
                color=PFT_COLORS[ipft],alpha=0.5,
                edgecolor=PFT_COLORS[ipft],linewidth=2.)

    return

    
def area_timeseries(
    ax : 'axis handle for the plot',
    time_data : 'time data for x-axis',
    time_unit : 'unit for time data, days, month, or year',
    var_data    : 'dataframe for the variables',
    var_name  : 'name of the variable to plot',
    var_unit  : 'unit of the variable',
    var_labels : 'labels of the variables',
    legend_on : 'whether plot legend' = False,
    xlabel_on : 'whether plot xlabel' = True,
    ylabel_on : 'whether plot ylabel' = True,
    color_list : 'color for PFTs' = PFT_COLORS
):

#    var_name = var_list[0].split('_')[0]
#    if var_name in ['NPLANT','NPLANT10']:
#        var_scaler = 1.e4
#    else:
#        var_scaler = 1.
#
#    y_data = var_df[var_list].values.T * var_scaler  # time is the second dimension
#    y_data[np.isnan(y_data)] = 0.

    hp_list = ax.stackplot(time_data,var_data,baseline='zero',labels=var_labels,
                 colors=color_list,linewidth=0.5,alpha=0.8)

    if xlabel_on:
        ax.set_xlabel(time_unit)

    if ylabel_on:
        ax.set_ylabel('{:s} [{:s}]'.format(var_name,var_unit))


    if legend_on:
        ax.legend(loc='best')
    
    return hp_list

#def bar_size(
#    ax : 'axis handle for the plot',
#    size_list : 'size bins',
#    var_df    : 'dataframe for the variables',
#    time_idx  : 'time to extract',
#    var_name  : 'name of the variable to plot',
#    var_unit  : 'unit of the variable',
#    pft_list  : 'list of PFTs',
#    pft_names : 'name of PFTs',
#    legend_on : 'whether plot legend' = False,
#    xlabel_on : 'whether plot xlabel' = True,
#    ylabel_on : 'whether plot ylabel' = True,
#    color_list : 'color for PFTs' = PFT_COLORS
#):
#    '''
#        Use bar plot to show the size distribution of all the quantities
#    '''
#    # first prepare data
#    if var_name in ['BA','NPLANT','NPLANT10']:
#        pft_mask = (np.array(pft_list) != 1)
#    else:
#        pft_mask = (np.array(pft_list) > 0)
#
#    pft_list_plot = np.array(pft_list)[pft_mask]
#    pft_names_plot = np.array(pft_names)[pft_mask]
#    pft_colors_plot = np.array(color_list[0:len(pft_list)])[pft_mask]
#
#    
#    if var_name in ['NPLANT','NPLANT10']:
#        var_scaler = 1.e4
#    else:
#        var_scaler = 1.
#
#    # create matrix of data
#    data_matrix = np.zeros((len(size_list[1]),len(pft_list_plot)))
#
#    for ipft, pft in enumerate(pft_list_plot):
#        for isize, size_edge in enumerate(size_list[1]):
#            col_name = '{:s}_PFT_{:d}_{:s}_{:d}'.format(
#                    var_name,pft,size_list[0],isize)
#            data_matrix[isize,ipft] = var_df[col_name].values[time_idx] * var_scaler
#
#
#    data_matrix[np.isnan(data_matrix)] = 0.
#    # set the smallest category to 0 for NPLANT
#
#    if var_name in ['NPLANT']:
#        data_matrix[0,:] = 0.
#
#    x_pos = size_list[1]
#    width = x_pos[1] - x_pos[0]
#
#    cur_bot = np.zeros_like(x_pos)
#
#    # loop over pfts
#    for ipft, pft in enumerate(pft_list_plot):
#        ax.bar(x_pos,data_matrix[:,ipft].ravel(),width,bottom=cur_bot,
#               facecolor=pft_colors_plot[ipft],align='edge',label=pft_names_plot[ipft])
#        cur_bot += data_matrix[:,ipft].ravel()
#
#    if size_list[0] == 'D':
#        xlabel = 'DBH [cm]'
#    elif size_list[0] == 'H':
#        xlabel = 'Height [m]'
#
#    if xlabel_on:
#        ax.set_xlabel(xlabel)
#
#    if ylabel_on:
#        ax.set_ylabel('{:s} [{:s}]'.format(var_name,var_unit))
#
#    if legend_on:
#        ax.legend(loc='best')
#
#    return

#######################
# This is based on the new data stucture
# TODO: should replace the old one
#########################
def bar_size_pft(
    ax : 'axis handle for the plot',
    size_list : 'size bins',
    var_df    : 'dataframe for the variables',
    var_name  : 'name of the variable to plot',
    var_unit  : 'unit of the variable',
    pft_list  : 'list of PFTs',
    pft_names : 'name of PFTs',
    legend_on : 'whether plot legend' = False,
    xlabel_on : 'whether plot xlabel' = True,
    ylabel_on : 'whether plot ylabel' = True,
    color_list : 'color for PFTs' = PFT_COLORS
):
    '''
        Use bar plot to show the size distribution of all the quantities
    '''
    # first prepare data
    if var_name in ['BA','NPLANT','NPLANT10']:
        pft_mask = (np.array(pft_list) != 1)
    else:
        pft_mask = (np.array(pft_list) > 0)

    pft_list_plot = np.array(pft_list)[pft_mask]
    pft_names_plot = np.array(pft_names)[pft_mask]
    pft_colors_plot = np.array(color_list[np.array(pft_list_plot).astype(int)])

    
    if var_name in ['NPLANT','NPLANT10']:
        var_scaler = 1.e4
    else:
        var_scaler = 1.

    # create matrix of data
    data_matrix = np.zeros((len(size_list[1]),len(pft_list_plot)))

    for ipft, pft in enumerate(pft_list_plot):
        pft_mask = var_df['PFT'] == pft
        for isize, size_edge in enumerate(size_list[1]):
            size_mask = var_df['SIZE'] == '{:s}_{:d}'.format(size_list[0],isize)

            # find the corresponding data
            data_matrix[isize,ipft] = var_df[var_name].values[pft_mask & size_mask] * var_scaler


    data_matrix[np.isnan(data_matrix)] = 0.
    # set the smallest category to 0 for NPLANT

    if var_name in ['NPLANT']:
        data_matrix[0,:] = 0.

    x_pos = size_list[1]
    width = x_pos[1] - x_pos[0]

    cur_bot = np.zeros_like(x_pos)

    # loop over pfts
    for ipft, pft in enumerate(pft_list_plot):
        ax.bar(x_pos,data_matrix[:,ipft].ravel(),width,bottom=cur_bot,
               facecolor=pft_colors_plot[ipft],align='edge',label=pft_names_plot[ipft])
        cur_bot += data_matrix[:,ipft].ravel()

    if size_list[0] == 'D':
        xlabel = 'DBH [cm]'
    elif size_list[0] == 'H':
        xlabel = 'Height [m]'

    if xlabel_on:
        ax.set_xlabel(xlabel)

    if ylabel_on:
        ax.set_ylabel('{:s} [{:s}]'.format(var_name,var_unit))

    if legend_on:
        ax.legend(loc='best')

    return

def forest_2d_profile(
    ax : 'axis handle for the plot',
    individual_df : 'dataframe for individual level information',
    pft_list : 'list for PFTs',
    pft_names : 'Name of PFTs',
    color_list : 'color for PFTs' = PFT_COLORS
):

    '''
        Plot a 2d forest profile based on the simulation results
    '''

    # some constant for plot
    site_x = 100. # m
    site_y = 100. # m

    # prepare data
    stem_h  = individual_df['HITE'].values
    stem_x  = individual_df['X_COOR'].values
    stem_y  = individual_df['Y_COOR'].values
    stem_r  = individual_df['CROWN_RADIUS'].values
    stem_d  = individual_df['DBH'].values / 100. # convert to m
    stem_p  = individual_df['PFT'].values.astype(int)

    # generate patches for stems and crowns
    stem_taper = 0.6
    # plot the furthest stems then the closer stems

    # dissect the plot into 5 blocks in y direction
    # use different alpha values to indicate distance
    transect_starts = np.arange(100,0.,-20.)
    transect_ends = transect_starts - 20.
    transect_alphas = np.arange(0.6,0.81,0.05)

    for transect_start, transect_end, transect_alpha in zip(
        transect_starts, transect_ends, transect_alphas):
        # get stem ids
        transect_mask = (stem_y <= transect_start) & (stem_y >= transect_end)
        stem_ids = np.arange(len(stem_p))[transect_mask]
    
        patches = []
        facecolors = []
        for ist in stem_ids:
            # first generate the stem polygon
            # assume a tapering of 60% at the top of the stem
            stem_verts = [[stem_x[ist] - stem_d[ist]/2., 0.],
                          [stem_x[ist] + stem_d[ist]/2., 0.],
                          [stem_x[ist] + stem_d[ist]/2. * (1. - stem_taper), stem_h[ist]],
                          [stem_x[ist] - stem_d[ist]/2. * (1. - stem_taper), stem_h[ist]]
                          ]
            polygon = Polygon(stem_verts,color='xkcd:black')
            patches.append(polygon)
            facecolors.append('xkcd:black')

            # second generate the crown polygon
            # assume the crown highet is 1/5 of crown diameter
            crown_verts = [[stem_x[ist], stem_h[ist] + stem_r[ist] * 2. /4.],
                           [stem_x[ist] - stem_r[ist], stem_h[ist] + stem_r[ist] * 2. / 6.],
                           [stem_x[ist] - stem_r[ist], stem_h[ist]],
                           [stem_x[ist] + stem_r[ist], stem_h[ist]],
                           [stem_x[ist] + stem_r[ist], stem_h[ist]],
                           [stem_x[ist] + stem_r[ist], stem_h[ist] + stem_r[ist] * 2. / 6.]
                          ]

            polygon = Polygon(crown_verts,color=color_list[stem_p[ist]])
            patches.append(polygon)
            facecolors.append(color_list[stem_p[ist]])


        p = PatchCollection(patches,facecolor=facecolors,alpha=transect_alpha)
        ax.add_collection(p)

    ax.plot([0.,site_x],[0.,0.],'k-',lw=2.)   
    ax.set_xlim((-10.,site_x+10))
    ax.set_ylim((0.,60.))

    ax.set_xticks([0.,site_x])
    ax.set_xticklabels(['Old Patch','New Patch'],ha='center',fontsize=7)
    ax.set_yticks([0,15,30,45,60])
    ax.set_ylabel('H [m]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # for legend purpose only:
    # loop over pfts
    x_pos = [1]
    bar_data = np.nan
    width = 0.
    bottom = 0.
    for ipft, pft in enumerate(pft_list):
        ax.bar(x_pos,bar_data,width,
               facecolor=color_list[pft],align='edge',label=pft_names[ipft])


    return

def forest_2d_overview(
    ax : 'axis handle for the plot',
    individual_df : 'dataframe for individual level information',
    pft_list : 'list for PFTs',
    pft_names : 'Name of PFTs',
    color_list : 'color for PFTs' = PFT_COLORS
):

    '''
        Plot a 2d forest overview based on the simulation results
    '''

    # some constant for plot
    site_x = 60. # m
    site_y = 60. # m

    # prepare data
    stem_h  = individual_df['HITE'].values
    stem_x  = individual_df['X_COOR'].values
    stem_y  = individual_df['Y_COOR'].values
    stem_r  = individual_df['CROWN_RADIUS'].values
    stem_d  = individual_df['DBH'].values / 100. # convert to m
    stem_p  = individual_df['PFT'].values.astype(int)


    # loop each individual to plot a circle
    patches = []
    facecolors = []
    for ist, pft in enumerate(stem_p):
        # generate stem circle
        polygon = Circle([stem_x[ist],stem_y[ist]],stem_d[ist]/2.,color='xkcd:black')
        patches.append(polygon)
        facecolors.append('xkcd:black')

        # second generate the crown polygon
        polygon = Circle([stem_x[ist],stem_y[ist]],stem_r[ist],color=color_list[stem_p[ist]-1])
        patches.append(polygon)
        facecolors.append(color_list[stem_p[ist]-1])


    p = PatchCollection(patches,facecolor=facecolors,alpha=0.75)
    ax.add_collection(p)
   
    # deal with the axes
    ax.set_xlim((-10.,site_x+10))
    ax.set_ylim((-10,site_y+10))

    ax.set_xticks([0.,60.])
    ax.set_yticks([0.,60.])

    ax.set_aspect['equal']

    return



# use other programs to plot 3d figures e.g. povray

#def forest_3d(
#    ax : 'axis handle for the plot',
#    individual_df : 'dataframe for individual level information',
#    pft_list : 'list for PFTs',
#    pft_names : 'Name of PFTs',
#    color_list : 'color for PFTs' = PFT_COLORS
#):
#
#    '''
#        Plot a 3d forest based on the simulation results
#    '''
#
#    # some constant for plot
#    site_x = 60. # m
#    site_y = 60. # m
#
#    # prepare data
#    stem_h  = individual_df['HITE'].values
#    stem_x  = individual_df['X_COOR'].values
#    stem_y  = individual_df['Y_COOR'].values
#    stem_r  = individual_df['CROWN_RADIUS'].values
#    stem_d  = individual_df['DBH'].values / 100. # convert to m
#    stem_p  = individual_df['PFT'].values.astype(int)
#
#
#
#    # first plot the stems while ignore the PFT 1 (grass)
#    plot_mask = (stem_p != 1)
#    ax.bar3d(stem_x[plot_mask] - stem_d[plot_mask]/2.,
#             stem_y[plot_mask] - stem_d[plot_mask]/2.,np.zeros_like(stem_x[plot_mask])
#            ,stem_d[plot_mask],stem_d[plot_mask],stem_h[plot_mask]
#            ,color='xkcd:black',alpha=0.5,antialiased=True)
#
#
#    # second loop over each individual to plot the crown
#    for i_ind in np.arange(len(stem_h)):
#        print('{:d}/{:d}'.format(i_ind,len(stem_h)))
#        # get ipft
#        ipft = np.where(pft_list == stem_p[i_ind])[0][0]
#        if stem_p[i_ind] == 1:
#            # grass use bar plot
#            dx, dy = 0.05,0.05
#            _x = np.arange(stem_x[i_ind] - stem_d[i_ind] / 2.,
#                           stem_x[i_ind] + stem_d[i_ind] / 2.,
#                           dx*10)
#            _y = np.arange(stem_y[i_ind] - stem_d[i_ind] / 2.,
#                           stem_y[i_ind] + stem_d[i_ind] / 2.,
#                           dy * 10)
#            _xx, _yy = np.meshgrid(_x,_y)
#
#            x,y = _xx.ravel(),_yy.ravel()
#
#            top = np.ones_like(x) * stem_h[i_ind]
#            bot = np.zeros_like(x)
#
#            ax.bar3d(x,y,bot,
#                     dx,dy,top,color=color_list[0])
#
#        else:
#            # tree
#            # use a cone plot
#            # assume crown depth is 1/10 of crown radius
#
#            _r = np.linspace(0,stem_r[i_ind],50)
#            _a = np.linspace(0,2 * np.pi,50)
#            x = np.outer(_r,np.cos(_a))
#            y = np.outer(_r,np.sin(_a))
#
#            # bottom of the canopy
#            z = np.ones_like(x) * stem_h[i_ind]
#            ax.plot_surface(x+stem_x[i_ind],y+stem_y[i_ind],z,
#                            color=color_list[ipft],antialiased=True)
#
#            # top of the canopy
#            z = ((1. - ((x ** 2 + y ** 2) ** 0.5) / stem_r[i_ind]) * (stem_r[i_ind] / 5.)
#                 + stem_h[i_ind])
#            ax.plot_surface(x+stem_x[i_ind],y+stem_y[i_ind],z,
#                            color=color_list[ipft],antialiased=True)
#
#    
#    ax.set_xlim((0.,site_x))
#    ax.set_ylim((0.,site_y))
#    ax.set_zlim((0.,60.))
#    # plot grid lines
#    patch_x = 10.
#    patch_y = 10.
#    vert_x = [0.,0.,60.,60.]
#    vert_y = [0.,60.,60.,0.]
#    for ivert in np.arange(len(vert_x)):
#        ivert_next = (ivert) % len(vert_x)
#        plot_x = [vert_x[ivert],vert_x[ivert_next]]
#        plot_y = [vert_y[ivert],vert_y[ivert_next]]
#        ax.plot(plot_x,plot_y,0.,color='k',lw=1.5,ls='-')
#
#    # plot patch lines
#    vert_xs = np.arange(0.,site_x,patch_x)
#    vert_ys = np.arange(0.,site_y,patch_y)
#    for vert_x in vert_xs:
#        plot_x = [vert_x,vert_x]
#        plot_y = [0.,site_y]
#        ax.plot(plot_x,plot_y,0.,color='k',lw=1.,ls='--')
#
#    for vert_y in vert_ys:
#        plot_x = [0.,site_x]
#        plot_y = [vert_y,vert_y]
#        ax.plot(plot_x,plot_y,0.,color='k',lw=1.,ls='--')
#
#    ax.axis('off')
#    return





