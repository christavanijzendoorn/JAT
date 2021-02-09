# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 12:45:17 2021

@author: cijzendoornvan
"""

######################
# PACKAGES
######################
import yaml
import pickle
from JAT.Jarkus_Analysis_Toolbox import Transects, Extraction

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/02_region_transects/jarkus.yml"))

#%%###################
# LOAD DATA
######################
# Load jarkus dataset
data = Transects(config)

# To view the metadata:
# print(data.dataset)
# print(data.variables)

#%%
# The following step makes sure the requested years and transects are available in the dataset.
data.get_availability(config)

# Those that were available are saved as follows: 
print(data.transects_filtered)
print(data.years_filtered)
# Note that only the years from 1980 onwards are available for this specific transect!

#%%###################
# SAVE + PLOT DATA
######################
# Save elevation dataframes for the available transects
data.save_elevation_dataframes(config)

# to reopen pickle file with elevation:
transect = str(config['transects']['transects_req'][0])
elevation = pickle.load(open(config['root'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

#%%
# Create elevation plots for the available transects - saved as png and pickle
data.get_transect_plot(config)

# to reopen pickle file with figure:
# figx = pickle.load(open(config['root'] + config['save locations']['DirB'] + 'Transect_' + transect + '.fig.pickle','rb'))    
# figx.show()
    
#%%###################################
# EXTRACT CHARACTERISTICS PARAMETERS
######################################

# Extract all requested dimensions for the available transects and years     
extract = Extraction(data, config)
extract.get_all_dimensions()
# note that extract.dimensions holds the dataframe with dimensions for the last transect after applying the function above.

dimensions = extract.get_requested_variables()

# Convert all dimensions extracted per transect location to a dataframe per dimension
extract.get_dataframe_per_dimension()

# Normalize dimensions along the x-axis 
extract.normalize_dimensions()

#%%###################
# OPEN DF's
######################
# For further analysis one can load pickle files from the directory where the dataframes have been saved.
# Note, you need to know the variable names to load them, see documentation or jarkus.yml file.

# For example, loading the dune top to determine the maximum dune height along the Dutch coast.
variable = 'DuneTop_prim_y'
dune_tops = pickle.load(open(config['root'] + config['save locations']['DirD'] + variable + '_dataframe.pickle','rb'))    
print('The maximum dune height is ' + str(max(dune_tops.max())) + ' meters.')

# Or, loading the dimensions of the certain transect to plot the change in the dune toe and landward boundary location:
dimensions = pickle.load(open(config['root'] + config['save locations']['DirC'] + 'Transect_' + transect + '_dataframe.pickle', 'rb'))

ax1 = dimensions.plot(y='Landward_x_variance', lw = '0', marker = '+', markersize=7, markeredgewidth=2, title = 'Cross-shore location of important characteristic parameters through time')
dimensions.plot(y='DuneTop_prim_x', lw = '0', marker = '+', markersize=7, markeredgewidth=2, ax=ax1)
dimensions.plot(y='Dunefoot_x_fix', lw = '0', marker = '+', markersize=7, markeredgewidth=2, ax=ax1)
dimensions.plot(y='Seaward_x_DoC', lw = '0', marker = '+', markersize=7, markeredgewidth=2, ax=ax1)

ax2 = dimensions.plot(y='Dunefoot_y_der', lw = '0', marker = '+', markersize=7, markeredgewidth=2, title = 'Dune foot elevation through time')

ax3 = dimensions.plot(y='DuneTop_prim_y', lw = '0', marker = '+', markersize=7, markeredgewidth=2, title = 'Dune top elevation through time')

ax4 = dimensions.plot(y='DuneVol_fix', lw = '0', marker = '+', markersize=7, markeredgewidth=2, title = 'Dune volume through time')

ax5 = dimensions.plot(y='Beach_width_fix', lw = '0', marker = 'o', markersize=7, markeredgewidth=2, title = 'Beach width through time')

ax6 = dimensions.plot(y='Active_profile_volume', lw = '0', marker = 'o', markersize=7, markeredgewidth=2, title = 'Active profile volume through time')




