'''This file is part of Jarkus Analysis Toolbox.
   
JAT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
   
JAT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
   
You should have received a copy of the GNU General Public License
along with JAT.  If not, see <http://www.gnu.org/licenses/>.
   
JAT  Copyright (C) 2020 Christa van IJzendoorn
c.o.vanijzendoorn@tudelft.nl
Delft University of Technology
Faculty of Civil Engineering and Geosciences
Stevinweg 1
2628CN Delft
The Netherlands
'''

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
from Jarkus_Analysis_Toolbox import Transects
import Filtering_functions as Ff

get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/Jarkus_Toolbox/jarkus.yml"))
location_filter = yaml.safe_load(open(config['root'] + config['data locations']['LocFilter']))
plot_titles = yaml.safe_load(open(config['root'] + config['data locations']['Titles'])) 

# Load jarkus dataset
data = Transects(config)
conversion_alongshore2ids, conversion_ids2alongshore = data.get_conversion_dicts()

#%%##############################
####        FUNCTIONS        ####
#################################   
def get_filtered_transects(variable, start_yr, end_yr):
    dimension = pickle.load(open(config['root'] + config['save locations']['DirD'] + variable + '_dataframe.pickle','rb')) 
    
    dimension_filt = Ff.bad_locations_filter(dimension, location_filter)
    dimension_filt = Ff.availability_filter_locations(config, dimension_filt)
    dimension_filt = Ff.bad_yrs_filter(dimension_filt, start_yr, end_yr) 
    dimension_filt = Ff.availability_filter_years(config, dimension_filt)

    dimension_nourished, dimension_not_nourished = Ff.nourishment_filter(config, dimension_filt)

    dimension_filt.to_pickle(config['root'] + config['save locations']['DirE'] + variable + '_filtered_dataframe' + '.pickle')
    dimension_nourished.to_pickle(config['root'] + config['save locations']['DirE'] + variable + '_nourished_dataframe' + '.pickle')
    dimension_not_nourished.to_pickle(config['root'] + config['save locations']['DirE'] + variable + '_not_nourished_dataframe' + '.pickle')
    
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

    plt.savefig(config['root'] + config['save locations']['DirE'] + variable + '_distribution' + '_plot.png')
    # pickle.dump(fig, open(config['root'] + config['save locations']['DirE'] + variable + '_distribution' + '_plot.fig.pickle', 'wb'))

#%%##############################
####      EXECUTE            ####
#################################
start_yr = 1980 
end_yr = 2020

dimension1 = get_filtered_transects('Dunefoot_y_der', start_yr, end_yr)
figure_title = 'Alongshore and temporal variation of dune foot elevation (m)'
colorbar_label = 'Dune foot elevation (m)'
get_distribution_plot('Dunefoot_y_der', dimension1, figure_title, colorbar_label, start_yr, end_yr)

dimension2 = get_filtered_transects('Dunefoot_x_der_normalized', start_yr, end_yr)

dimension3 = get_filtered_transects('Dunefoot_y_pybeach_mix', start_yr, end_yr)
dimension4 = get_filtered_transects('Dunefoot_x_pybeach_mix_normalized', start_yr, end_yr)




