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
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/dune_toe_analysis/jarkus_03.yml"))

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

#%%###################
# SAVE + PLOT DATA
######################
# Save elevation dataframes for the available transects
data.save_elevation_dataframes(config)

# to reopen pickle file with elevation:
transect = str(config['transects']['single'])
elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + str(transect) + '_elevation' + '.pickle','rb'))    

#%%
# Create elevation plots for the available transects - saved as png and pickle
data.get_transect_plot(config)

# to reopen pickle file with figure:
figx = pickle.load(open(config['outputdir'] + config['save locations']['DirB'] + 'Transect_' + str(transect) + '.fig.pickle','rb'))    
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