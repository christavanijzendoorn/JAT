# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 07:12:48 2020

@author: cijzendoornvan
"""

######################################
# THIS SCRIPT CREATES FIGURE 3
######################################

######################
# PACKAGES
######################
import yaml
import pickle
from JAT.Jarkus_Analysis_Toolbox import Transects
import JAT.Filtering_functions as Ff

import matplotlib.pyplot as plt
import numpy as np
import os

######################
# LOAD SETTINGS + DATA
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/dune_toe_analysis/jarkus_04.yml"))
filter_file = yaml.safe_load(open(config['inputdir'] + config['data locations']['LocFilter']))

DirDimensions = config['outputdir'] + config['save locations']['DirD']
DirFigures = config['outputdir'] + config['save locations']['DirFig']
if os.path.isdir(DirFigures) == False:
            os.mkdir(DirFigures)

# Load 2nd derivative method version
var = 'Dunetoe_y_der' # Dune toe elevation
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_y_der = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension     

var = 'Dunetoe_x_der' # Dune toe cross-shore location
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_x_der = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    

DF_x_der[DF_x_der > 10000000] = np.nan
DF_y_der[DF_y_der > 10000000] = np.nan

years = DF_y_der.index
trscts = DF_y_der.columns  

#%%
##################################
####       PREPARATIONS       ####
##################################
# Prepare plotting of Tidal range

# Load and plot pybeach method version
var = 'MHW_y_var'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_MHW = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
DF_MHW = Ff.bad_locations_filter(DF_MHW, filter_file)
   
var = 'MLW_y_var'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_MLW = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
DF_MLW = Ff.bad_locations_filter(DF_MLW, filter_file)

tidal_range = DF_MHW.loc[1965] - DF_MLW.loc[1965]
MHW = DF_MHW.loc[1965]

#%% Prepare plotting of temporal average of dune toe elevation
begin_yr = 1980
end_yr = 2017

# Filter bad locations from dataframe
variable_DF = Ff.bad_locations_filter(DF_y_der, filter_file)

# Get conversion dictionary needed to convert from transect number to alongshore value
data = Transects(config)
conversion_alongshore2ids, conversion_ids2alongshore = data.get_conversion_dicts()

# Convert index from transect number to alongshore value
variable_DF.rename(columns = conversion_ids2alongshore, inplace = True)
variable_DF = variable_DF.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.
variable_DF.rename(columns = conversion_alongshore2ids, inplace = True)

# Calculate spatial and temporal average
average_through_space = variable_DF.loc[list(range(begin_yr, end_yr))].mean(axis=0)
average_through_time = variable_DF.loc[list(range(begin_yr, end_yr))].mean(axis=1)

# Calculate overall average and stddev, used for range of colorbar
average         = np.nanmean(variable_DF.values)
stddev          = np.nanstd(variable_DF.values, ddof=1)
range_value     = 2*stddev
range_value_avg = stddev
vmin            = average - range_value
vmax            = average + range_value
vmin_avg        = average - range_value_avg
vmax_avg        = average + range_value_avg

plt.rcParams.update({'lines.linewidth': 3})

figure_title = 'Alongshore and temporal variation of dune toe elevation (m)'
colorbar_label = 'Dune toe elevation (m)'
colormap_var = "Greens"

# Create an array with locations and an array with labels of the ticks
ticks_x = [350, 1100, 1900]
labels_x = ['Wadden Coast', 'Holland Coast', 'Delta Coast']

years_requested = list(range(begin_yr, end_yr))
ticks_y = range(0, len(years_requested))[0::5]
labels_y = [str(yr) for yr in years_requested][0::5]

#%%###############################
####       PLOTTING       ####
##################################
# # PLOT TEMPORAL AVERAGE OF VARIABLE ALONGSHORE
fig = plt.figure(figsize=(25,7)) 
cmap = plt.cm.get_cmap(colormap_var) # Define color use for colorbar
colorplot = plt.scatter(range(0, len(average_through_space)), average_through_space, c=average_through_space, cmap=cmap, vmin=0, vmax=6)
# Set labels and ticks of x and y axis
plt.xlim([0, len(average_through_space)])
plt.xticks(ticks_x, labels_x) 
plt.ylabel('Elevation (m)', fontsize = 24)
plt.tick_params(axis='x', which='both',length=0, labelsize = 24)
plt.ylim([0, 6])
plt.tick_params(axis='y', which='both',length=5, labelsize = 24)
# plt.plot(range(0, len(average_through_space)),tidal_range, color = '#4169E1', label = 'Tidal range (m)', linewidth = 6) 
plt.plot(range(0, len(average_through_space)),MHW, color = '#4169E1', label = 'Mean High Water (m)', linewidth = 6) 

plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

plt.legend(fontsize = 23, loc='upper left')

# Plot colorbar
cbar = fig.colorbar(colorplot)
cbar.set_label(colorbar_label,size=24, labelpad = 20)
cbar.ax.tick_params(labelsize=24) 

plt.tight_layout
plt.show()

filename2_pdf = 'Alongshore_variation' + '.pdf'
# filename2_eps = 'Alongshore_variation'  + '.eps'
filename2_png = 'Alongshore_variation' + '.png'
plt.savefig(DirFigures + filename2_png, bbox_inches='tight')
# plt.savefig(DirFigures + filename2_eps, format='eps', bbox_inches='tight')
plt.savefig(DirFigures + filename2_pdf, bbox_inches='tight')
print('saved figure alongshore variation')

#%% PLOT TEMPORAL AND SPATIAL DISTRIBUTION OF VARIABLE
fig2 = plt.figure(figsize=(18,9)) 
cmap2 = plt.cm.get_cmap(colormap_var) # Define color use for colorbar
colorplot2 = plt.pcolor(DF_y_der, vmin=vmin, vmax=vmax, cmap=cmap2)
# Set labels and ticks of x and y axis
labels_y = [str(yr) for yr in DF_y_der.index][0::5]
plt.yticks(range(0, len(DF_y_der.index))[0::5], labels_y)
plt.tick_params(axis='y', which='both',length=5, labelsize = 22)
plt.xticks(ticks_x, labels_x) #rotation='vertical')
plt.tick_params(axis='x', which='both',length=0, labelsize = 24)
# plot boundaries between coastal regions
plt.axvline(x=686, color='r') # Boundary kustvak 6 en 7, Den Helder, trsct 7000000, decameter 23417
plt.axvline(x=1507, color='r')# Boundary kustvak 9 en 10, Hoek van Holland, trsct 10000000, decameter 35908

plt.legend(fontsize = 20)

# Plot colorbar
cbar2 = fig2.colorbar(colorplot2)
cbar2.set_label(colorbar_label,size=22, labelpad = 20)
cbar2.ax.tick_params(labelsize=20) 

fig.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.show()

filename2 = 'Temporal_alongshore_variation' + '.png'
plt.savefig(DirFigures + filename2)
print('saved figure temporal and alongshore variation')

