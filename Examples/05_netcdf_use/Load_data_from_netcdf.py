# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:59:09 2021

@author: cijzendoornvan
"""

######################
# PACKAGES
######################
from netCDF4 import Dataset, num2date
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

######################
# LOAD SETTINGS
######################

# create netcdf file
netcdf_dir = r"C:\Users\cijzendoornvan\OneDrive - Delft University of Technology\Documents\DuneForce\JARKUS\JAT\Input\extracted_parameters.nc"

#%%##############################
####      TEST LOADING       ####
#################################

# Load netcdf 
char_params = Dataset(netcdf_dir)

# Get global attributes
for name in char_params.ncattrs():
    print(name + ' = ' + str(getattr(char_params, name)))

# Get all dimensions
print(char_params.dimensions.values())

# Get all variables info
print(char_params.variables.values())

# Get info for one variable
print(char_params.variables['DuneTop_prim_x'].__dict__)

#%%##############################
####      EXTRACT DATA       ####
#################################

# Get values of one variable
dunetops = char_params.variables['DuneTop_prim_x'][:,:]

dunevols = char_params.variables['DuneVol_fix'][:,:]

#%%##############################
####  Convert to DataFrame   ####
#################################
# For those who prefer to work with pandas it is relatively easy to convert

# Get years and transect numbers to set index
time = char_params.variables['time'][:]
years = num2date(time, char_params.variables['time'].units)
years = [yr.year for yr in years]

trscts = char_params.variables['id'][:]

# Create dataframe
dunevols_pd = pd.DataFrame(dunevols, index = years, columns = trscts)

#%%##############################
####       EXAMPLE PLOT      ####
#################################

dunevols_mean = dunevols_pd.mean(axis=1)

# plot mean dune volume through time
plt.plot(dunevols_pd.index, dunevols_mean)
plt.title('Dune volume through time along the Dutch coast')
plt.xlabel('years')
plt.ylabel('Dune volume (m$^3$)')
