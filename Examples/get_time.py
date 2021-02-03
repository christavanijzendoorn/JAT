# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:08:51 2020

@author: cijzendoornvan
"""

import yaml
import os
import pickle
import xarray as xr
import pandas as pd
import numpy as np

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/Jarkus_Toolbox/jarkus.yml"))

dataset = xr.open_dataset(config['root'] + config['data locations']['DirJK'])
variables = dataset.variables

time_bathy = dataset.time_bathy.values
time_topo = dataset.time_topo.values

start_yr = 2019
end_yr = 2021
transects_requested = 9010883

time = dataset.variables['time'].values                     
years = pd.to_datetime(time).year
years = pd.to_datetime(time).year                        
years_requested = list(range(start_yr, end_yr))
years_filter =  np.isin(years, years_requested)
years_filtered = np.array(years)[np.nonzero(years_filter)[0]]
years_filtered_idxs = np.where(years_filter)[0]

ids = dataset.variables['id'].values                             
transects_filter = np.isin(ids, transects_requested)
transects_filtered = np.array(ids)[np.nonzero(transects_filter)[0]]
transects_filtered_idxs = np.where(transects_filter)[0]

print('Transect ',transects_requested,': ',time_bathy[years_filtered_idxs[0],transects_filtered_idxs])
print('Transect ',transects_requested,': ',time_bathy[years_filtered_idxs[1],transects_filtered_idxs])