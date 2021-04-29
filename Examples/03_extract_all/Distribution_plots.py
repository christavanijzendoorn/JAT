# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:33:29 2021

@author: cijzendoornvan
"""

######################
# PACKAGES
######################
import yaml
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from JAT.Jarkus_Analysis_Toolbox import Transects, Extraction

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open(r"C:\Users\cijzendoornvan\OneDrive - Delft University of Technology\Documents\DuneForce\JARKUS\JAT\Examples\03_extract_all\jarkus_03.yml"))
location_filter = yaml.safe_load(open(config['inputdir'] + config['data locations']['LocFilter']))
plot_titles = yaml.safe_load(open(config['inputdir'] + config['data locations']['Titles'])) 
  
DirFigures = config['outputdir'] + config['save locations']['DirFig']
if os.path.isdir(DirFigures) == False:
            os.mkdir(DirFigures)

# Load jarkus dataset
data = Transects(config)
conversion_alongshore2ids, conversion_ids2alongshore = data.get_conversion_dicts()

def get_distribution_plot(variable, dimension, figure_title, colorbar_label, start_yr, end_yr, Dir):
   
    # Create an array with locations and an array with labels of the ticks
    ticks_x = [350, 1100, 1700]
    labels_x = ['Wadden Coast', 'Holland Coast', 'Delta Coast']
    
    years_requested = list(range(start_yr, end_yr+1))
    ticks_y = range(0, len(years_requested))[0::5]
    labels_y = [str(yr) for yr in years_requested][0::5]
    
    dimension.rename(columns = conversion_ids2alongshore, inplace = True)
    dimension = dimension.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.
    # dimension.rename(columns = conversion_alongshore2ids, inplace = True)
    
    # Calculate overall average and stddev, used for range of colorbar
    average         = np.nanmean(dimension.values)
    stddev          = np.nanstd(dimension.values, ddof=1)
    range_value     = 2*stddev
    vmin            = average - range_value
    vmax            = average + range_value
    
    average_through_space = dimension.mean(axis=0)
    average_through_time = dimension.mean(axis=1)
    vmin_avg = average_through_time.min()
    vmax_avg = average_through_time.max()
    
    # Set-up of figure
    fig = plt.figure(figsize=(20,10)) 
    fig.suptitle(figure_title, fontsize=24)
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3,2]) 
    
    # PLOT TEMPORAL AND SPATIAL DISTRIBUTION OF VARIABLE
    ax1 = fig.add_subplot(gs[0])
    cmap = plt.cm.get_cmap('Greens') # Define color use for colorbar
    colorplot = plt.pcolor(dimension.loc[start_yr:end_yr], vmin=vmin, vmax=vmax, cmap=cmap)
    # Set labels and ticks of x and y axis
    plt.yticks(ticks_y, labels_y)
    plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
    plt.xticks(ticks_x, labels_x) #rotation='vertical')
    plt.tick_params(axis='x', which='both',length=0, labelsize = 20)
    # plot boundaries between coastal regions
    plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
    plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908
    
    # PLOT YEARLY AVERAGE OF VARIABLE
    ax2 = fig.add_subplot(gs[1])
    plt.plot(average_through_time, average_through_time.index, color='green')
    #plt.scatter(average_through_time, average_through_time.index, c=average_through_time, cmap=cmap, vmin=vmin, vmax=vmax)
    # Set labels and ticks of x and y axis
    ticks_y = average_through_time.index[0::5]
    plt.xlabel(colorbar_label)
    plt.yticks(ticks_y, labels_y)
    plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
    plt.xlim([vmin_avg-0.75*stddev, vmax_avg+0.75*stddev])
    plt.tick_params(axis='x', which='both',length=5, labelsize = 16)
    
    # PLOT SPATIAL AVERAGES OF VARIABLE
    ax3 = fig.add_subplot(gs[2])
    plt.scatter(range(0, len(average_through_space)), average_through_space, c=average_through_space, cmap=cmap, vmin=vmin, vmax=vmax)
    # Set labels and ticks of x and y axis
    plt.xlim([0, len(average_through_space)])
    plt.xticks(ticks_x, labels_x) 
    plt.ylabel(colorbar_label)
    plt.tick_params(axis='x', which='both',length=0, labelsize = 20)
    plt.ylim([vmin, vmax])
    plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
    plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
    plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

    # Add colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.78])
    cbar = fig.colorbar(colorplot, cax=cbar_ax)
    cbar.set_label(colorbar_label,size=18, labelpad = 20)
    cbar.ax.tick_params(labelsize=16)     
    
    plt.tight_layout
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

    plt.savefig(Dir + variable + '_distribution' + '_plot.png')

    plt.close()

#%%##############################
####      EXECUTE            ####
#################################
start_yr = 1965 
end_yr = 2021

extract = Extraction(data, config) # initalize the extra class 
variables = extract.get_requested_variables() # get all variables that were requested (based on jarkus.yml file)

for variable in variables:
    print(variable)
    # Get title and label from yml file
    figure_title = plot_titles[variable][0]
    colorbar_label = plot_titles[variable][1]
    
    # Create distribution plots for unfiltered dataframes
    dimension_unfilt = pickle.load(open(config['outputdir'] + config['save locations']['DirD'] + variable + '_dataframe.pickle','rb')) 
    DirFig_unfilt = config['outputdir'] + config['save locations']['DirFig'] + config['save locations']['DirD']
    if os.path.isdir(DirFig_unfilt) == False:
        os.mkdir(DirFig_unfilt)
    get_distribution_plot(variable, dimension_unfilt, figure_title, colorbar_label, start_yr, end_yr, DirFig_unfilt)

    # Create distribution plots for filtered dataframes
    dimension_filt = pickle.load(open(config['outputdir'] + config['save locations']['DirE'] + variable + '_filtered_dataframe.pickle','rb')) 
    DirFig_filt = config['outputdir'] + config['save locations']['DirFig'] + config['save locations']['DirE']
    if os.path.isdir(DirFig_filt) == False:
            os.mkdir(DirFig_filt)
    get_distribution_plot(variable, dimension_filt, figure_title, colorbar_label, start_yr, end_yr, DirFig_filt)
