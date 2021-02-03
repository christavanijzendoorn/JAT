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
Created on Mon Nov  2 11:56:17 2020

@author: cijzendoornvan
"""

######################
# PACKAGES
######################
import yaml
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from IPython import get_ipython
from Jarkus_Analysis_Toolbox_classes import Transects
import Filtering_functions as Ff
get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/Jarkus_Toolbox/jarkus.yml"))
location_filter = yaml.safe_load(open(config['root'] + config['data locations']['LocFilter']))
plot_titles = yaml.safe_load(open(config['root'] + config['data locations']['Titles'])) 

start_yr = 1980 
end_yr = 2020

variable = 'Dunefoot_y_der'

#%%
##################################
####       PREPARATIONS       ####
##################################
# Load jarkus dataset
data = Transects(config)
conversion_alongshore2ids, conversion_ids2alongshore = data.get_conversion_dicts()

dimension = pickle.load(open(config['root'] + config['save locations']['DirD'] + variable + '_dataframe' + '.pickle','rb'))   
# dimension = pickle.load(open(r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\ANALYSIS\D1_dataframes_per_dimension\Dunefoot_y_der_new_dataframe.pickle",'rb')) 
dimension = Ff.bad_locations_filter(dimension, location_filter)
dimension.rename(columns = conversion_ids2alongshore, inplace=True)
dimension = dimension.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.

# Calculate spatial and temporal average
average_through_space = dimension.loc[list(range(start_yr, end_yr))].mean(axis=0)
average_through_time = dimension.loc[list(range(start_yr, end_yr))].mean(axis=1)

# Calculate overall average and stddev, used for range of colorbar
average         = np.nanmean(dimension.values)
stddev          = np.nanstd(dimension.values, ddof=1)
range_value     = 2*stddev
range_value_avg = stddev
vmin            = average - range_value
vmax            = average + range_value
vmin_avg        = average - range_value_avg
vmax_avg        = average + range_value_avg

# Create an array with locations and an array with labels of the ticks
ticks_x = [350, 1100, 1900]
labels_x = ['Wadden Coast', 'Holland Coast', 'Delta Coast']

years_requested = list(range(start_yr, end_yr))
ticks_y = range(0, len(years_requested))[0::5]
labels_y = [str(yr) for yr in years_requested][0::5]

#%% Prepare plotting of Tidal range

# Load and plot pybeach method version
var = 'MHW_y_var'
DF_MHW = pickle.load(open(config['root'] + config['save locations']['DirD'] + var + '_dataframe.pickle','rb'))
# DF_MHW = pickle.load(open(r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\ANALYSIS\D1_dataframes_per_dimension\MHW_y_var_dataframe.pickle",'rb'))
DF_MHW = Ff.bad_locations_filter(DF_MHW, location_filter)
DF_MHW.rename(columns = conversion_ids2alongshore, inplace = True)
DF_MHW = DF_MHW.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.
DF_MHW.rename(columns = conversion_alongshore2ids, inplace = True)
   
MHW = DF_MHW.loc[1965]
plt.rcParams.update({'lines.linewidth': 3})

#%%###############################
####       PLOTTING       ####
##################################
# Plot overviews and trends for Derivative method DF elevation
figure_title = 'Alongshore and temporal variation of dune toe elevation (m)'
colorbar_label = 'Dune toe elevation (m)'
colormap_var = "Greens"
file_name = 'dune_foot_elevation'

# Set-up of figure
fig = plt.figure(figsize=(25,7)) 

# PLOT SPATIAL AVERAGES OF VARIABLE
cmap = plt.cm.get_cmap(colormap_var) # Define color use for colorbar
colorplot = plt.scatter(range(0, len(average_through_space)), average_through_space, c=average_through_space, cmap=cmap, vmin=0, vmax=6)
# Set labels and ticks of x and y axis
plt.xlim([0, len(average_through_space)])
plt.xticks(ticks_x, labels_x) 
plt.ylabel('Elevation (m)', fontsize = 20)
plt.tick_params(axis='x', which='both',length=0, labelsize = 20)
plt.ylim([0, 6])
plt.tick_params(axis='y', which='both',length=5, labelsize = 16)
# plt.plot(range(0, len(average_through_space)),tidal_range, color = '#4169E1', label = 'Tidal range (m)', linewidth = 6) 
plt.plot(range(0, len(average_through_space)),MHW, color = '#4169E1', label = 'Mean High Water (m)', linewidth = 6) 

plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

plt.legend(fontsize = 16)

# Plot colorbar
cbar = fig.colorbar(colorplot)
cbar.set_label(colorbar_label,size=18, labelpad = 20)
cbar.ax.tick_params(labelsize=16) 

plt.tight_layout
plt.show()

filename2 = 'Overview_DF_part1' + file_name + '.pdf'
# plt.savefig(DirDFAnalysis + filename2)
print('saved figure')
#plt.close()













