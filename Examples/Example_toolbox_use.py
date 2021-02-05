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
Created on Wed Jun 10 17:24:40 2020

@author: cijzendoornvan
"""
######################
# PACKAGES
######################
import yaml
import os
import pickle
import numpy as np
from Jarkus_Analysis_Toolbox_classes import Transects, Extraction
# get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/Jarkus_Toolbox/jarkus.yml"))
location_filter = yaml.safe_load(open(config['root'] + config['data locations']['LocFilter']))
plot_titles = yaml.safe_load(open(config['root'] + config['data locations']['Titles'])) 

#%%###################
# USER-DEFINED REQUEST
######################
start_yr = 1980                                                               # USER-DEFINED request for years
end_yr = 2020

trscts_requested = 8009325
# trscts_requested = [8009325, 8009350]
#trscts_requested = np.arange(8009000, 8009751, 1)                               # USER-DEFINED request for transects


# Set whether all transect should be analysed or define a retrieval request
# Note! If put as true it overrides the input of trscts_requested 
execute_all_transects = False

#%%###################
# LOAD DATA
######################
# Load jarkus dataset
data = Transects(config)

# To view the metadata:
# print(data.dataset)
# print(data.variables)

data.get_availability(start_yr, end_yr, trscts_requested, execute_all_transects)

# Make sure the requested years and transects are available in the dataset. Those that were available are saved as follows: 
# print(data.transects_filtered)
# print(data.years_filtered)

#%%###################
# SAVE + PLOT DATA
######################

# Save elevation dataframes for the available transects
data.save_elevation_dataframes(config, apply_filter1='yes')
data.get_transect_plot(config)

# to reopen pickle file with figure:
transect = '8009325'
figx = pickle.load(open(config['root'] + config['save locations']['DirB'] + 'Transect_' + transect + '.fig.pickle','rb'))    
figx.show()

    
#%%###################
# EXTRACT DATA
######################
    
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