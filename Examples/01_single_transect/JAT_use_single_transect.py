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
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/01_single_transect/jarkus.yml"))

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
transect = '8009325'
elevation = pickle.load(open(config['root'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

#%%
# Create elevation plots for the available transects - saved as png and pickle
data.get_transect_plot(config)

# to reopen pickle file with figure:
transect = '8009325'
figx = pickle.load(open(config['root'] + config['save locations']['DirB'] + 'Transect_' + transect + '.fig.pickle','rb'))    
figx.show()
    
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
# For further analysis one can load pickle files from the save locations
# For example, loading the dune top to determine the maximum dune height along the Dutch coast.
variable = 'DuneTop_prim_y'
dune_tops = pickle.load(open(config['root'] + config['save locations']['DirD'] + variable + '_dataframe.pickle','rb'))    
print('The maximum dune height in NL is ' + str(max(dune_tops.max())) + ' meters.')


# Or, loading the dimensionsof a certain transect to plot the change in the dune toe and landward boundary location:
transect = str(trscts_requested[0])
dimensions = pickle.load(open(config['root'] + config['save locations']['DirC'] + 'Transect_' + transect + '_dataframe.pickle', 'rb'))

ax1 = dimensions.plot(y='Dunefoot_x_fix', lw = '0', marker = '+', markersize=7, markeredgewidth=2)
dimensions.plot(y='Landward_x_bma', lw = '0', marker = 'o', markersize=7, markeredgewidth=2, ax=ax1)