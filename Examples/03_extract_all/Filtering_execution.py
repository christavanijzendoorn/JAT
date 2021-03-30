# -*- coding: utf-8 -*-

"""
Created on Wed Aug  5 07:12:48 2020

@author: cijzendoornvan
"""

######################
# PACKAGES
######################
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from IPython import get_ipython
from JAT.Jarkus_Analysis_Toolbox import Transects
import JAT.Filtering_functions as Ff

get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open(r"C:\Users\cijzendoornvan\OneDrive - Delft University of Technology\Documents\DuneForce\JARKUS\JAT\Examples\dune_toe_analysis\jarkus_03.yml"))
location_filter = yaml.safe_load(open(config['inputdir'] + config['data locations']['LocFilter']))
plot_titles = yaml.safe_load(open(config['inputdir'] + config['data locations']['Titles'])) 

# Load jarkus dataset
data = Transects(config)
conversion_alongshore2ids, conversion_ids2alongshore = data.get_conversion_dicts()

#%%##############################
####        FUNCTIONS        ####
#################################   
def get_filtered_transects(variable, start_yr, end_yr):
    dimension = pickle.load(open(config['outputdir'] + config['save locations']['DirD'] + variable + '_dataframe.pickle','rb')) 
    
    dimension_filt = Ff.bad_locations_filter(dimension, location_filter)
    dimension_filt = Ff.availability_filter_locations(config, dimension_filt)
    dimension_filt = Ff.bad_yrs_filter(dimension_filt, start_yr, end_yr) 
    dimension_filt = Ff.availability_filter_years(config, dimension_filt)
    
    dimension_filt[dimension_filt > 10000] = np.nan # Check for values that have not been converted correctly to nans

    # dimension_nourished, dimension_not_nourished = Ff.nourishment_filter(config, dimension_filt)

    # dimension_filt.to_pickle(config['outputdir'] + config['save locations']['DirE'] + variable + '_filtered_dataframe' + '.pickle')
    # dimension_nourished.to_pickle(config['outputdir'] + config['save locations']['DirE'] + variable + '_nourished_dataframe' + '.pickle')
    # dimension_not_nourished.to_pickle(config['outputdir'] + config['save locations']['DirE'] + variable + '_not_nourished_dataframe' + '.pickle')
    
    return dimension_filt

def get_distribution_plot(variable, dimension, figure_title, colorbar_label, start_yr, end_yr):

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
    print(vmax)
    
    # Set-up of figure
    fig = plt.figure() 
    plt.title(figure_title, fontsize=24)
    
    # PLOT TEMPORAL AND SPATIAL DISTRIBUTION OF VARIABLE
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

    # Plot colorbar
    cbar = fig.colorbar(colorplot)
    cbar.set_label(colorbar_label,size=18, labelpad = 20)
    cbar.ax.tick_params(labelsize=16) 

    plt.tight_layout
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

    # plt.savefig(config['outputdir'] + config['save locations']['DirE'] + variable + '_distribution' + '_plot.png')
    # pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirE'] + variable + '_distribution' + '_plot.fig.pickle', 'wb'))

#%%##############################
####      EXECUTE            ####
#################################
start_yr = 1980 
end_yr = 2020

dimension1 = get_filtered_transects('Dunetoe_y_der', start_yr, end_yr)
figure_title = 'Alongshore and temporal variation of dune toe elevation (m)'
colorbar_label = 'Dune toe elevation (m)'
get_distribution_plot('Dunetoe_y_der', dimension1, figure_title, colorbar_label, start_yr, end_yr)

dimension2 = get_filtered_transects('Dunetoe_x_der_normalized', start_yr, end_yr)

dimension3 = get_filtered_transects('Dunetoe_y_pybeach_mix', start_yr, end_yr)
dimension4 = get_filtered_transects('Dunetoe_x_pybeach_mix_normalized', start_yr, end_yr)







# # Set-up of figure
# fig = plt.figure(figsize=(25,7)) 

# # fig.suptitle(figure_title, fontsize=26)

# cmap = plt.cm.get_cmap(colormap_var) # Define color use for colorbar
# # PLOT SPATIAL AVERAGES OF VARIABLE
# colorplot = plt.scatter(range(0, len(average_through_space)), average_through_space, c=average_through_space, cmap=cmap, vmin=0, vmax=6)
# # Set labels and ticks of x and y axis
# plt.xlim([0, len(average_through_space)])
# plt.xticks(ticks_x, labels_x) 
# plt.ylabel('Elevation (m)', fontsize = 24)
# plt.tick_params(axis='x', which='both',length=0, labelsize = 24)
# plt.ylim([0, 6])
# plt.tick_params(axis='y', which='both',length=5, labelsize = 24)
# # plt.plot(range(0, len(average_through_space)),tidal_range, color = '#4169E1', label = 'Tidal range (m)', linewidth = 6) 
# plt.plot(range(0, len(average_through_space)),MHW, color = '#4169E1', label = 'Mean High Water (m)', linewidth = 6) 

# plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
# plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

# plt.legend(fontsize = 23, loc='upper left')

# # Plot colorbar
# cbar = fig.colorbar(colorplot)
# cbar.set_label(colorbar_label,size=24, labelpad = 20)
# cbar.ax.tick_params(labelsize=24) 

# plt.tight_layout
# plt.show()

# filename2_pdf = 'Overview_DF_part1' + file_name + '.pdf'
# filename2_eps = 'Overview_DF_part1' + file_name + '.eps'
# filename2_png = 'Overview_DF_part1' + file_name + '.png'
# plt.savefig(DirFigures + filename2_png, bbox_inches='tight')
# plt.savefig(DirFigures + filename2_eps, format='eps', bbox_inches='tight')
# plt.savefig(DirFigures + filename2_pdf, bbox_inches='tight')
# print('saved figure')
# #plt.close()




# # Set-up of figure
# fig2 = plt.figure() 

# # fig2.suptitle(figure_title, fontsize=26)

# # PLOT TEMPORAL AND SPATIAL DISTRIBUTION OF VARIABLE
# cmap2 = plt.cm.get_cmap(colormap_var) # Define color use for colorbar
# colorplot2 = plt.pcolor(DF_y_der, vmin=vmin, vmax=vmax, cmap=cmap2)
# # Set labels and ticks of x and y axis
# labels_y = [str(yr) for yr in DF_y_der.index][0::5]
# plt.yticks(range(0, len(DF_y_der.index))[0::5], labels_y)
# plt.tick_params(axis='y', which='both',length=5, labelsize = 22)
# plt.xticks(ticks_x, labels_x) #rotation='vertical')
# plt.tick_params(axis='x', which='both',length=0, labelsize = 24)
# # plot boundaries between coastal regions
# plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
# plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

# plt.legend(fontsize = 20)

# # Plot colorbar
# cbar2 = fig2.colorbar(colorplot2)
# cbar2.set_label(colorbar_label,size=22, labelpad = 20)
# cbar2.ax.tick_params(labelsize=20) 

# plt.tight_layout

# manager = plt.get_current_fig_manager()
# manager.window.showMaximized()
# plt.show()

# filename2 = 'Overview_DF_part2' + file_name + '.png'
# # filename3 = 'Overview_DF_part2' + file_name + '.eps'
# # plt.savefig(DirFigures + filename3, format='eps')
# plt.savefig(DirFigures + filename2)
# print('saved figure')
# #plt.close()



# # PLOT YEARLY AVERAGE OF VARIABLE
# ax2 = fig.add_subplot(gs[1])
# plt.plot(average_through_time, average_through_time.index, color=colormap_var[:-1])
# # plt.scatter(average_through_time, average_through_time.index, c=average_through_time, cmap=cmap, vmin=vmin, vmax=vmax)
# # Set labels and ticks of x and y axis
# ticks_y = average_through_time.index[0::5]
# plt.xlabel(colorbar_label, fontsize = 20)
# plt.yticks(ticks_y, labels_y)
# plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
# plt.xlim([vmin_avg+0.75*stddev, vmax_avg-0.75*stddev])
# plt.tick_params(axis='x', which='both',length=5, labelsize = 16)




