# -*- coding: utf-8 -*-

"""
Created on Wed Aug  5 07:12:48 2020

@author: cijzendoornvan
"""

######################
# PACKAGES
######################
import yaml
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from JAT.Jarkus_Analysis_Toolbox import Transects, Extraction
import JAT.Filtering_functions as Ff

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open(r"C:\Users\cijzendoornvan\OneDrive - Delft University of Technology\Documents\DuneForce\JARKUS\JAT\Examples\03_extract_all\jarkus_03.yml"))
location_filter = yaml.safe_load(open(config['inputdir'] + config['data locations']['LocFilter']))
plot_titles = yaml.safe_load(open(config['inputdir'] + config['data locations']['Titles'])) 

DirFiltered = config['outputdir'] + config['save locations']['DirE']
if os.path.isdir(DirFiltered) == False:
    os.mkdir(DirFiltered)

# Load jarkus dataset
data = Transects(config)
conversion_alongshore2ids, conversion_ids2alongshore = data.get_conversion_dicts()

#%%##############################
####        FUNCTIONS        ####
#################################   
def get_filtered_transects(variable, start_yr, end_yr):
    dimension = pickle.load(open(config['outputdir'] + config['save locations']['DirD'] + variable + '_dataframe.pickle','rb')) 
    
    dimension_filt = Ff.locations_filter(dimension, location_filter)
    dimension_filt = Ff.availability_locations_filter(config, dimension_filt)
    dimension_filt = Ff.yrs_filter(dimension_filt, start_yr, end_yr) 
    dimension_filt = Ff.availability_years_filter(config, dimension_filt)
    
    dimension_filt[dimension_filt > 10000] = np.nan # Check for values that have not been converted correctly to nans

    # dimension_nourished, dimension_not_nourished = Ff.nourishment_filter(config, dimension_filt)

    dimension_filt.to_pickle(config['outputdir'] + config['save locations']['DirE'] + variable + '_filtered_dataframe' + '.pickle')
    # dimension_nourished.to_pickle(config['outputdir'] + config['save locations']['DirE'] + variable + '_nourished_dataframe' + '.pickle')
    # dimension_not_nourished.to_pickle(config['outputdir'] + config['save locations']['DirE'] + variable + '_not_nourished_dataframe' + '.pickle')
    
    return dimension_filt

#%%##############################
####      EXECUTE            ####
#################################
start_yr = 1965 
end_yr = 2021

extract = Extraction(data, config) # initalize the extra class 
variables = extract.get_requested_variables() # get all variables that were requested (based on jarkus.yml file)

for variable in variables:
    print(variable)
    dimension = get_filtered_transects(variable, start_yr, end_yr)

