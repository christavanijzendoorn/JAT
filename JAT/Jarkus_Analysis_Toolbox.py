"""
Created on Tue Nov 19 11:31:25 2019

@author: cijzendoornvan
"""

""" Most important functionalities of the JAT inclusing retrieving data and extracting profile dimensions """

import numpy as np
import pandas as pd
import pickle
import os
from netCDF4 import Dataset, num2date
from JAT.Geometric_functions import *

#################################
####     DATA-EXTRACTION     ####
#################################

class Transects:
    """Loading and plotting transects.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """
    def __init__(self, config): 
        # create a dataset object, based on JARKUS dataset saved locally or on server
        if 'http' in config['data locations']['Jarkus']: # check whether it's a url
            self.dataset = Dataset(config['data locations']['Jarkus'])    
        else: # load from local file
            self.dataset = Dataset(config['inputdir'] + config['data locations']['Jarkus'])
        self.variables = self.dataset.variables
        
    def get_years_filtered(self, start_yr, end_yr):
        time = self.variables['time'][:]                     # retrieve years from jarkus dataset
        years = num2date(time, self.variables['time'].units)
        years = [yr.year for yr in years]                    # convert to purely integers indicating the measurement year
        years_requested = list(range(start_yr, end_yr))
        years_filter =  np.isin(years, years_requested)
        self.years_filtered = np.array(years)[np.nonzero(years_filter)]
        self.years_filtered_idxs = np.where(years_filter)[0]
   
    def get_transects_filtered(self, transects):
        ids = self.variables['id'][:]                              # retrieve transect ids from jarkus dataset
        if transects['type'] == 'all':
            transects_requested = ids
        elif transects['type'] == 'single':
            transects_requested = transects['single']
        elif transects['type'] == 'multiple':
            transects_requested = transects['multiple']
        elif transects['type'] == 'range':
            transects_requested = np.arange(transects['range']['start'], transects['range']['end'], 1)
        else:
            print("Error: define type of transect request to all, single, multiple or range")
        transects_filter = np.isin(ids, transects_requested)
        self.transects_filtered = np.array(ids)[np.nonzero(transects_filter)[0]]
        self.transects_filtered_idxs = np.where(transects_filter)[0]
    
    def get_availability(self, config):
        self.get_years_filtered(config['years']['start_yr'], config['years']['end_yr'])    
        self.get_transects_filtered(config['transects'])    
        
    def save_elevation_dataframes(self, config):
                
        crossshore = self.variables['cross_shore'][:]

        for i, trsct_idx in enumerate(self.transects_filtered_idxs):
            trsct = str(self.transects_filtered[i])
            elevation_dataframe = pd.DataFrame(index=self.years_filtered, columns=crossshore)
            for j, yr_idx in enumerate(self.years_filtered_idxs):   
                elevation_dataframe.loc[self.years_filtered[j]] = self.variables['altitude'][yr_idx, trsct_idx, :]  # elevation of profile point
                
            if config['user defined']['filter1']['apply'] == True:
                for idx, row in elevation_dataframe.iterrows():
                    if min(row) > config['user defined']['filter1']['min'] or max(row) < config['user defined']['filter1']['max']:
                        elevation_dataframe.drop(idx, axis=0)
                
            if os.path.isdir(config['outputdir'] + config['save locations']['DirA']) == False:
                os.mkdir(config['outputdir'] + config['save locations']['DirA'])
            elevation_dataframe.to_pickle(config['outputdir'] + config['save locations']['DirA'] + trsct + '_elevation.pickle')
           
    def get_transect_plot(self, config):
        
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        
        crossshore = self.variables['cross_shore'][:]
        
        for i, trsct_idx in enumerate(self.transects_filtered_idxs):
            trsct = str(self.transects_filtered[i])
            
        	# Set figure layout
            fig = plt.figure(figsize=(30,15))
            ax = fig.add_subplot(111)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            
            jet = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin=min(self.years_filtered), vmax=max(self.years_filtered))
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)    
               
            # Load and plot data per year
            for i, yr in enumerate(self.years_filtered):
                yr_idx = self.years_filtered_idxs[i]
                
                colorVal = scalarMap.to_rgba(yr)
                elevation = self.variables['altitude'][yr_idx, trsct_idx, :]
                mask = elevation.mask
                plt.plot(crossshore[~mask], elevation[~mask], color=colorVal, label = str(yr), linewidth = 2.5)
            
            # Added this to get the legend to work
            handles,labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper right',ncol=2, fontsize = 20)
            
            # Label the axes and provide a title
            ax.set_title("Transect {}".format(str(trsct)), fontsize = 28)
            ax.set_xlabel("Cross shore distance [m]", fontsize = 24)
            ax.set_ylabel("Elevation [m to datum]", fontsize = 24)
            
            # Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
            xlim = [] # EXAMPLE: [-400,1000]
            ylim = [] # EXAMPLE: [-10,22]
            if len(xlim) != 0:
                ax.set_xlim(xlim)
            if len(ylim) != 0:
                ax.set_ylim(ylim)
            #ax.grid()
            #ax.invert_xaxis()
                
            # Save figure as png in predefined directory
            if os.path.isdir(config['outputdir'] + config['save locations']['DirB']) == False:
                os.mkdir(config['outputdir'] + config['save locations']['DirB'])
            plt.savefig(config['outputdir'] + config['save locations']['DirB'] + 'Transect_' + str(trsct) + '.png')
            pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB'] + 'Transect_' + str(trsct) + '.fig.pickle', 'wb'))
            
            plt.close()
        
    def get_conversion_dicts(self): # Create conversion dictionary
        trscts = self.variables['id'][:]  
        area_bounds = [2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000, 13000000, 14000000, 15000000, 16000000, 17000000, 18000000]
        
        for i, val in enumerate(area_bounds):
            if i == 0: # Flip numbers for first Wadden Island
                ids_filt = trscts[np.where(np.logical_and(trscts>=area_bounds[i], trscts<area_bounds[i+1]))] #- area_bounds[i]
                transition_value = min(ids_filt) - area_bounds[i]
                
                ids_filt = [transition_value + (max(ids_filt) - ids) for ids in ids_filt]
                
                ids_alongshore = ids_filt
            elif i < 6: # For the Wadden Islands, flipping the alongshore numbers and creating space between islands
                ids_filt = trscts[np.where(np.logical_and(trscts>=area_bounds[i], trscts<area_bounds[i+1]))] #- area_bounds[i]
                transition_value = 100
                ids_old = ids_filt
                
                ids_filt = [transition_value + (max(ids_filt) - ids) for ids in ids_filt]
                ids_filt = [max(ids_alongshore) + ids for ids in ids_filt]
                
                ids_alongshore = np.concatenate((ids_alongshore, ids_filt))
            elif i == 6 or i == 7: # Where alongshore numbers are counting throughout consecutive area codes
                ids_filt = trscts[np.where(np.logical_and(trscts>=area_bounds[i], trscts<area_bounds[i+1]))] #- area_bounds[i]
                
                transition_value = (min(ids_filt) - area_bounds[i])  - (max(ids_old) - area_bounds[i-1])
                ids_old = ids_filt
                
                ids_filt = [transition_value + (ids - min(ids_filt)) for ids in ids_filt]
                ids_filt = [max(ids_alongshore) + ids for ids in ids_filt]
                
                ids_alongshore = np.concatenate((ids_alongshore, ids_filt))
            elif i == 16: # Done
                print("Converted all areacodes to alongshore values")
            else: # Create space between area codes and no flipping necessary.
                ids_filt = trscts[np.where(np.logical_and(trscts>=area_bounds[i], trscts<area_bounds[i+1]))] #- area_bounds[i]
                transition_value = 100
                ids_old = ids_filt
                
                ids_filt = [transition_value + (ids - min(ids_filt)) for ids in ids_filt]
                ids_filt = [max(ids_alongshore) + ids for ids in ids_filt]
        
                ids_alongshore = np.concatenate((ids_alongshore, ids_filt))
            
            # Create conversion dictionary  
            trscts_str = [str(tr) for tr in trscts]
            conversion_alongshore2ids = dict(zip(ids_alongshore, trscts_str))
            conversion_ids2alongshore = dict(zip(trscts_str, ids_alongshore))
            
        return conversion_alongshore2ids, conversion_ids2alongshore

class Extraction:
    
    def __init__(self, data, config):    
        self.dimensions = pd.DataFrame()
        self.data = data
        self.config = config
        self.crossshore = data.variables['cross_shore'][:]
        
    def get_requested_variables(self):
        self.variables_req = []
        for key in self.config['dimensions']['setting']:
            if self.config['dimensions']['setting'][key] == True:
                self.variables_req.extend(self.config['dimensions']['variables'][key])
                
        return self.variables_req
                
    def get_all_dimensions(self):
        import warnings # This error occurs due to nan values in less than boolean operations.
        warnings.filterwarnings("ignore", message="invalid value encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        
        if os.path.isdir(self.config['outputdir'] + self.config['save locations']['DirC']) == False:
            os.mkdir(self.config['outputdir'] + self.config['save locations']['DirC'])
            
        for i, trsct_idx in enumerate(self.data.transects_filtered_idxs):
            trsct = str(self.data.transects_filtered[i])
            print("Extracting parameters of transect " + trsct)
            
            pickle_file = self.config['save locations']['DirC'] + 'Transect_' + trsct + '_dataframe.pickle'
    
            if pickle_file in os.listdir(self.config['outputdir'] + self.config['save locations']['DirC']):
                self.dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
            else:
                self.dimensions = pd.DataFrame({'transect': trsct, 'years':self.data.years_filtered})
                self.dimensions.set_index('years', inplace=True)
            
            if self.config['dimensions']['setting']['primary_dune_top'] == True:
                self.get_primary_dune_top(trsct_idx)
            if self.config['dimensions']['setting']['secondary_dune_top'] == True:
                self.get_secondary_dune_top(trsct_idx)

            if self.config['dimensions']['setting']['mean_sea_level'] == True:
                self.get_mean_sea_level(trsct_idx)
            if self.config['dimensions']['setting']['mean_low_water_fixed'] == True:
                self.get_mean_low_water_fixed(trsct_idx)
            if self.config['dimensions']['setting']['mean_low_water_variable'] == True:
                self.get_mean_low_water_variable(trsct_idx)
            if self.config['dimensions']['setting']['mean_high_water_fixed'] == True:
                self.get_mean_high_water_fixed(trsct_idx)
            if self.config['dimensions']['setting']['mean_high_water_variable'] == True:
                self.get_mean_high_water_variable(trsct_idx)
            if self.config['dimensions']['setting']['mean_sea_level_variable'] == True:
                self.get_mean_sea_level_variable()
                
                
            if self.config['dimensions']['setting']['intertidal_width_fixed'] == True:
                self.get_intertidal_width_fixed()       
            if self.config['dimensions']['setting']['intertidal_width_variable'] == True:
                self.get_intertidal_width_variable()      

            if self.config['dimensions']['setting']['landward_point_variance'] == True:
                self.get_landward_point_variance(trsct_idx)       
            if self.config['dimensions']['setting']['landward_point_derivative'] == True:
                self.get_landward_point_derivative(trsct_idx)     
            if self.config['dimensions']['setting']['landward_point_bma'] == True:
                self.get_landward_point_bma(trsct_idx)    

            if self.config['dimensions']['setting']['seaward_point_foreshore'] == True:
                self.get_seaward_point_foreshore(trsct_idx)       
            if self.config['dimensions']['setting']['seaward_point_activeprofile'] == True:
                self.get_seaward_point_activeprofile(trsct_idx)     
            if self.config['dimensions']['setting']['seaward_point_doc'] == True:
                self.get_seaward_point_doc(trsct_idx)                
                
            if self.config['dimensions']['setting']['dune_toe_fixed'] == True:
                self.get_dune_toe_fixed(trsct_idx)       
            if self.config['dimensions']['setting']['dune_toe_derivative'] == True:
                self.get_dune_toe_derivative(trsct_idx)     
            if self.config['dimensions']['setting']['dune_toe_pybeach'] == True:
                self.get_dune_toe_pybeach(trsct_idx)      
                
            if self.config['dimensions']['setting']['beach_width_fix'] == True:
                self.get_beach_width_fix()       
            if self.config['dimensions']['setting']['beach_width_var'] == True:
                self.get_beach_width_var()     
            if self.config['dimensions']['setting']['beach_width_der'] == True:
                self.get_beach_width_der()   
            if self.config['dimensions']['setting']['beach_width_der_var'] == True:
                self.get_beach_width_der_var()  

            if self.config['dimensions']['setting']['beach_gradient_fix'] == True:
                self.get_beach_gradient_fix(trsct_idx)       
            if self.config['dimensions']['setting']['beach_gradient_var'] == True:
                self.get_beach_gradient_var(trsct_idx)     
            if self.config['dimensions']['setting']['beach_gradient_der'] == True:
                self.get_beach_gradient_der(trsct_idx)   
                
            if self.config['dimensions']['setting']['dune_front_width_prim_fix'] == True:
                self.get_dune_front_width_prim_fix()       
            if self.config['dimensions']['setting']['dune_front_width_prim_der'] == True:
                self.get_dune_front_width_prim_der()     
            if self.config['dimensions']['setting']['dune_front_width_sec_fix'] == True:
                self.get_dune_front_width_sec_fix()  
            if self.config['dimensions']['setting']['dune_front_width_sec_der'] == True:
                self.get_dune_front_width_sec_der()  
                
            if self.config['dimensions']['setting']['dune_front_gradient_prim_fix'] == True:
                self.get_dune_front_gradient_prim_fix(trsct_idx)       
            if self.config['dimensions']['setting']['dune_front_gradient_prim_der'] == True:
                self.get_dune_front_gradient_prim_der(trsct_idx)     
            if self.config['dimensions']['setting']['dune_front_gradient_sec_fix'] == True:
                self.get_dune_front_gradient_sec_fix(trsct_idx)  
            if self.config['dimensions']['setting']['dune_front_gradient_sec_der'] == True:
                self.get_dune_front_gradient_sec_der(trsct_idx)  

            if self.config['dimensions']['setting']['dune_volume_fix'] == True:
                self.get_dune_volume_fix(trsct_idx)  
            if self.config['dimensions']['setting']['dune_volume_der'] == True:
                self.get_dune_volume_der(trsct_idx)              
            
            if self.config['dimensions']['setting']['intertidal_gradient'] == True:
                self.get_intertidal_gradient_fix(trsct_idx)     
            if self.config['dimensions']['setting']['intertidal_volume_fix'] == True:
                self.get_intertidal_volume_fix(trsct_idx)  
            if self.config['dimensions']['setting']['intertidal_volume_var'] == True:
                self.get_intertidal_volume_var(trsct_idx)  
  
            if self.config['dimensions']['setting']['foreshore_gradient'] == True:
                self.get_foreshore_gradient(trsct_idx)  
            if self.config['dimensions']['setting']['foreshore_volume'] == True:
                self.get_foreshore_volume(trsct_idx)  
                
            if self.config['dimensions']['setting']['active_profile_gradient'] == True:
                self.get_active_profile_gradient(trsct_idx)  
            if self.config['dimensions']['setting']['active_profile_volume'] == True:
                self.get_active_profile_volume(trsct_idx)  
                
            # Save dimensions data frame for each transect
            self.dimensions.to_pickle(self.config['outputdir'] + pickle_file)
            
    def get_dataframe_per_dimension(self):
        variable_dataframe = pd.DataFrame({'years': self.data.years_filtered})
        variable_dataframe.set_index('years', inplace=True)   
        
        variables = self.variables_req
                
        for variable in variables:
            for i, trsct_idx in enumerate(self.data.transects_filtered_idxs):
                trsct = str(self.data.transects_filtered[i])
                pickle_file = self.config['outputdir'] + self.config['save locations']['DirC'] + 'Transect_' + trsct + '_dataframe.pickle'
                variable_dataframe.loc[:, trsct] = np.nan
        
                if os.path.exists(pickle_file):
                    dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
                    
                if variable not in dimensions.columns:
                    variable_dataframe.loc[:, trsct] = np.nan
                else:
                    for yr, row in dimensions.iterrows(): 
                        variable_dataframe.loc[yr, trsct] = dimensions.loc[yr, variable] #extract column that corresponds to the requested variable
                print('Extracted transect ' + str(trsct) + ' for variable ' + variable)
                    
            if os.path.isdir(self.config['outputdir'] + self.config['save locations']['DirD']) == False:
                os.mkdir(self.config['outputdir'] + self.config['save locations']['DirD'])
            variable_dataframe.to_pickle(self.config['outputdir'] + self.config['save locations']['DirD'] + variable + '_dataframe' + '.pickle')
            print('The dataframe of ' + variable + ' was saved')

    
    def normalize_dimensions(self):    
        # Get all variables that have to be normalized based on the requirement that _x should be in the column name, 
        # and that change values do not have to be normalized.
        variables = self.variables_req
        normalized_variables = [var for var in variables if '_x' in var and 'change' not in var]            
        for i, variable in enumerate(normalized_variables):      
            pickle_file = self.config['outputdir'] + self.config['save locations']['DirD'] + variable + '_dataframe' + '.pickle'
            dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions   
            normalized = dimensions.copy()
            norm_type = self.config['user defined']['normalization']['type']
            if norm_type == 'mean':
                for i, col in dimensions.iteritems():
                    # Get the mean cross-shore location per transect and subtract that from the values of the variable for each transect
                    normalized.loc[:, i] = col - col.mean()
            elif norm_type == 'norm_year':
                norm_year = self.config['user defined']['normalization'][' year']    
                for i, col in dimensions.iteritems():
                    # Get norm value for the cross-shore location in the norm year and subtract that from the values of the variable for each transect
                    normalized.loc[:, i] = col - col[norm_year]
            
            normalized.to_pickle(self.config['outputdir'] + self.config['save locations']['DirD'] + variable + '_normalized_dataframe' + '.pickle')
            print('The dataframe of ' + variable + ' was normalized and saved')

    def get_primary_dune_top(self, trsct_idx):        
        from scipy.signal import find_peaks
        # Documentation:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences    
        # The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            
            dune_top_prim = find_peaks(elevation, height = self.config['user defined']['primary dune']['height'], prominence = self.config['user defined']['primary dune']['prominence'])

            if len(dune_top_prim[0]) != 0: # If a peak is found in the profile
                # Select the most seaward peak found of the primarypeaks
                dune_top_prim_idx = dune_top_prim[0][-1]
                self.dimensions.loc[yr, 'DuneTop_prim_x'] = self.crossshore[dune_top_prim_idx] 
                self.dimensions.loc[yr, 'DuneTop_prim_y'] = elevation[dune_top_prim_idx]
            else:
                self.dimensions.loc[yr, 'DuneTop_prim_x'] = np.nan
                self.dimensions.loc[yr, 'DuneTop_prim_y'] = np.nan
                
    def get_secondary_dune_top(self, trsct_idx):        
        from scipy.signal import find_peaks
        # Documentation:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences    
        # The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            
            dune_top_sec = find_peaks(elevation, height = self.config['user defined']['secondary dune']['height'], prominence = self.config['user defined']['secondary dune']['prominence'])

            if len(dune_top_sec[0]) != 0: # If a peak is found in the profile
                # Select the most seaward peak found of the secondary peaks
                dune_top_sec_idx = dune_top_sec[0][-1]
                dune_top_prim = self.dimensions.loc[yr, 'DuneTop_prim_x']
                if self.crossshore[dune_top_sec_idx] > dune_top_prim or dune_top_prim == np.nan:
                    # Only if most seaward secondary peak is located seaward of the (most seaward) primary peak, save the secondary peak.
                    self.dimensions.loc[yr, 'DuneTop_sec_x'] = self.crossshore[dune_top_sec_idx]
                    self.dimensions.loc[yr, 'DuneTop_sec_y'] = elevation[dune_top_sec_idx]
            else: # Otherwise, assume that there is no seaward secondary peak, so no value filled in (i.e. it stays nan).
                self.dimensions.loc[yr, 'DuneTop_sec_x'] = np.nan
                self.dimensions.loc[yr, 'DuneTop_sec_y'] = np.nan
                
    def get_mean_sea_level(self, trsct_idx):
        MSL_y = self.config['user defined']['mean sea level'] # in m above reference datum
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            
            intersections = find_intersections(elevation, self.crossshore, MSL_y)
            if len(intersections) != 0 and np.isnan(self.dimensions.loc[yr, 'DuneTop_prim_x']):
                self.dimensions.loc[yr, 'MSL_x'] = intersections[-1] # get most seaward intersect
            
            # The following filtering is implemented to make sure offshore shallow parts are not identified as MSL. This is mostyl applicable for the Wadden Islands and Zeeland.
            elif len(intersections) != 0: 
                # get all intersections seaward of dunetop
                intersection_sw = intersections[intersections > self.dimensions.loc[yr, 'DuneTop_prim_x']] 
                # if distance between intersections seaward of dune peak is larger than 100m:
                if len(intersection_sw) != 0:
                    if max(intersection_sw) - min(intersection_sw) > 100: 
                        # get intersections at least 100m landwards of most offshore intersection 
                        intersection_lw = intersection_sw[intersection_sw < (min(intersection_sw) + 100)] 
                        # Of these, select the most seaward intersection
                        self.dimensions.loc[yr, 'MSL_x'] = intersection_lw[-1] 
                    else: 
                        # If the intersection seaward of the dunetop are within 100m of each other take the most seaward one.
                        self.dimensions.loc[yr, 'MSL_x'] = intersection_sw[-1]
                else:
                    self.dimensions.loc[yr, 'MSL_x'] = np.nan
            else:
                self.dimensions.loc[yr, 'MSL_x'] = np.nan
                
    def get_mean_low_water_fixed(self, trsct_idx):
        MLW_y_fixed   = self.config['user defined']['mean low water'] # in m above reference datum
            
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            intersections = find_intersections(elevation, self.crossshore, MLW_y_fixed)
            if len(intersections) != 0:
                # filter intersections based on the assumption that mean low water should be a maximum of 250 m offshore
                intersections_filt = intersections[(intersections < self.dimensions.loc[yr, 'MSL_x'] + 250)]
                if len(intersections_filt) == 0:
                    self.dimensions.loc[yr, 'MLW_x_fix'] = intersections[-1]
                else: 
                    self.dimensions.loc[yr, 'MLW_x_fix'] = intersections_filt[-1]
            else:
                self.dimensions.loc[yr, 'MLW_x_fix'] = np.nan
    
    def get_mean_low_water_variable(self, trsct_idx):
        MLW_y_variable   = self.data.variables['mean_low_water'][trsct_idx]     # gets the mean low water level as included in the Jarkus dataset, so it varies per location
        self.dimensions.loc[:, 'MLW_y_var'] = MLW_y_variable     # save assumed mean low water level as extracted from the jarkus dataset
            
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            intersections = find_intersections(elevation, self.crossshore, MLW_y_variable)
            if len(intersections) != 0:
                # filter intersections based on the assumption that mean low water should be a maximum of 250 m offshore
                intersections_filt = intersections[(intersections < self.dimensions.loc[yr, 'MSL_x'] + 250)]
                if len(intersections_filt) == 0:
                    self.dimensions.loc[yr, 'MLW_x_var'] = intersections[-1]
                else:
                    self.dimensions.loc[yr, 'MLW_x_var'] = intersections_filt[-1]
            else:
                self.dimensions.loc[yr, 'MLW_x_var'] = np.nan
    
    def get_mean_high_water_fixed(self, trsct_idx):
        MHW_y_fixed   = self.config['user defined']['mean high water'] # in m above reference datum
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            intersections = find_intersections(elevation, self.crossshore, MHW_y_fixed)
            if len(intersections) != 0:
                # filter intersections based on the assumption that mean high water should be a maximum of 250 m landward
                intersections_filt = intersections[(intersections < self.dimensions.loc[yr, 'MSL_x'] - 250)]
                if len(intersections_filt) == 0:
                    self.dimensions.loc[yr, 'MHW_x_fix'] = intersections[-1]
                else: 
                    self.dimensions.loc[yr, 'MHW_x_fix'] = intersections_filt[-1]
            else:
                self.dimensions.loc[yr, 'MHW_x_fix'] = np.nan
    
    def get_mean_high_water_variable(self, trsct_idx):
        MHW_y_variable   = self.data.variables['mean_high_water'][trsct_idx]     # gets the mean low water level as included in the Jarkus dataset, so it varies per location
        self.dimensions.loc[:, 'MHW_y_var'] = MHW_y_variable     # save assumed mean low water level as extracted from the jarkus dataset
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            intersections = find_intersections(elevation, self.crossshore, MHW_y_variable)
            if len(intersections) != 0:
                # filter intersections based on the assumption that mean high water should be a maximum of 250 m landward
                intersections_filt = intersections[(intersections < self.dimensions.loc[yr, 'MSL_x'] - 250)]
                if len(intersections_filt) == 0:
                    self.dimensions.loc[yr, 'MHW_x_var'] = intersections[-1]
                else:
                    self.dimensions.loc[yr, 'MHW_x_var'] = intersections_filt[-1]
            else:
                self.dimensions.loc[yr, 'MHW_x_var'] = np.nan
    
    def get_mean_sea_level_variable(self):
        self.dimensions.loc[:, 'MSL_x_var'] = (self.dimensions.loc[:, 'MLW_x_var'] + self.dimensions.loc[:, 'MHW_x_var'])/2 # Base MSL on the varying location of the low and high water line

    def get_intertidal_width_variable(self):
        # Collect info on seaward boundary in dataframe
        self.dimensions.loc[:,'Intertidal_width_var'] = self.dimensions.loc[:,'MLW_x_var'] - self.dimensions.loc[:,'MHW_x_var']
    
    def get_intertidal_width_fixed(self):
        self.dimensions.loc[:, 'Intertidal_width_fix'] = self.dimensions.loc[:, 'MLW_x_fix'] - self.dimensions.loc[:, 'MHW_x_fix']

    def get_landward_point_variance(self, trsct_idx):
            
        ####  Variance method - Sierd de Vries ####
        var_threshold = self.config['user defined']['landward variance threshold'] # very dependent on area and range of years!
        
        elevation = pd.DataFrame(self.data.variables['altitude'][:, trsct_idx, :], columns = self.crossshore)
        var_y = elevation.var()
        
        # Gives locations where variance is below threshold
        stable_points = var_y[(var_y < var_threshold)].index
        # Gives locations landward of primary dune
        dunes = elevation.columns[elevation.columns < self.dimensions['DuneTop_prim_x'].max()]
            
        try: 
            # Get most seaward stable point that is landward of dunes and with a variance below the threshold
            stable_point = np.intersect1d(stable_points, dunes)[-1]
        except:
            print("No stable point found")
            stable_point = np.nan
        
        # add info on landward boundary to dataframe
        self.dimensions['Landward_x_variance'] = stable_point
            
    def get_landward_point_derivative(self, trsct_idx):
            
        ####  Derivative method - Diamantidou ####
        ###################################
        # Get landward boundary from peaks in profile
        from scipy.signal import find_peaks
        
        height_of_peaks = self.config['user defined']['landward derivative']['min height'] #m
        height_constraint = self.config['user defined']['landward derivative']['height constraint'] #m
        peaks_threshold = height_of_peaks + self.dimensions['MHW_y_var'].iloc[0]  # adjust this based on matlab version?
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            peaks = find_peaks(elevation, prominence = height_of_peaks)[0] # Documentation see get_dune_top
            
            peaks = elevation[peaks]
            peaks_filt = peaks[peaks >= peaks_threshold]
            
            if len(peaks) != 0 and np.nanmax(peaks) > height_constraint:
                intersections_derivative = find_intersections(elevation, self.crossshore, height_constraint)
                if len(intersections_derivative) != 0:
                    self.dimensions.loc[yr, 'Landward_x_der'] = intersections_derivative[-1]
            elif len(peaks_filt) != 0:
                self.dimensions.loc[yr, 'Landward_x_der'] = peaks_filt[-1]
            else:
                self.dimensions.loc[yr, 'Landward_x_der'] = np.nan

                            
    def get_landward_point_bma(self, trsct_idx):
        ####       Bma calculation     ####
        ###################################
        # Calculating the approximate boundary between the marine and aeolian zone.
        # Based on De Vries et al, 2010, published in Coastal Engeineering.
    
        bma_y = self.config['user defined']['landward bma']
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            intersections_bma = find_intersections(elevation, self.crossshore, bma_y)
            if len(intersections_bma) != 0:
                self.dimensions.loc[yr, 'Landward_x_bma'] = intersections_bma[-1]
            else:
                self.dimensions.loc[yr, 'Landward_x_bma'] = np.nan

    def get_seaward_point_foreshore(self, trsct_idx):
        seaward_FS_y = self.config['user defined']['seaward foreshore']
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :] 
            intersections_FS = find_intersections(elevation, self.crossshore, seaward_FS_y)
            if len(intersections_FS) != 0:
                self.dimensions.loc[yr, 'Seaward_x_FS'] = intersections_FS[-1]
            else:
                self.dimensions.loc[yr, 'Seaward_x_FS'] = np.nan
            
    def get_seaward_point_activeprofile(self, trsct_idx):
        seaward_ActProf_y = self.config['user defined']['seaward active profile']
    
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :] 
            intersections_AP = find_intersections(elevation, self.crossshore, seaward_ActProf_y)
            if len(intersections_AP) != 0:
                self.dimensions.loc[yr, 'Seaward_x_AP'] = intersections_AP[-1]
            else:
                self.dimensions.loc[yr, 'Seaward_x_AP'] = np.nan
    
    def get_seaward_point_doc(self, trsct_idx):
        # Gives most seaward cross-shore location where where depth is -5.0 m NAP
        min_depth = self.config['user defined']['seaward DoC']['min depth']
    
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :] 
            intersections_mindepth = find_intersections(elevation, self.crossshore, min_depth)
            if len(intersections_mindepth) != 0:
                self.dimensions.loc[yr, 'Seaward_x_mindepth'] = intersections_mindepth[-1]
            else:
                self.dimensions.loc[yr, 'Seaward_x_mindepth'] = np.nan
        
        elevation = pd.DataFrame(self.data.variables['altitude'][:, trsct_idx, :], columns = self.crossshore)
                
        # Gives locations seaward of minimal seaward boundary at -5.0m NAP
        offshore = elevation.columns[elevation.columns > self.dimensions['Seaward_x_mindepth'].max()]
        
        ####  Method by Hinton (2000)  ####
        stdThr = self.config['user defined']['seaward DoC']['stddev threshold'] # standard deviation threshold, dependent on area and range of years!
        lowstd_length = self.config['user defined']['seaward DoC']['low stddev length'] # the average standard deviation of a section with this length has to be below the threshold
        stdv_y = elevation[offshore].std()
              
        # Gives locations where stddev is on average below threshold for a stretch of 200m
        window_size = lowstd_length
        stable_points = []
        for x_val in offshore:
            if x_val < max(stdv_y.index) - window_size + 5:
                this_window = stdv_y[x_val:x_val+window_size]
                with np.errstate(invalid='ignore'):
                    window_average = np.nanmean(this_window)
                if window_average < stdThr and stdv_y[x_val] < stdThr:
                    stable_points.append(x_val)
                
        try: 
            # Get most seaward stable point that is seaward of the minimal depth of -5.0 m NAP, has an average stddev below threshold 200m seaward and is itself below the threshold
            stable_point = stable_points[0]
        except:
            print("No stable point found")
            stable_point = np.nan
        
        # add info on landward boundary to dataframe
        self.dimensions['Seaward_x_DoC'] = stable_point

    def get_dune_toe_fixed(self, trsct_idx):
        #### Fixed dunetoe definition ####
        DF_fixed_y = self.config['user defined']['dune toe fixed'] # in m above reference datum
    
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :] 
            intersections_DF = find_intersections(elevation, self.crossshore, DF_fixed_y)
            if len(intersections_DF) != 0:
                self.dimensions.loc[yr, 'Dunetoe_x_fix'] = intersections_DF[-1]
            else:
                self.dimensions.loc[yr, 'Dunetoe_x_fix'] = np.nan
            
    def get_dune_toe_derivative(self, trsct_idx):        
            ####  Derivative method - E.D. ####
            ###################################
            ## Variable dunetoe definition based on first and second derivative of profile
            if 'http' in self.config['data locations']['Dunetoe']: # check whether it's a url
                dunetoes = Dataset(self.config['data locations']['Dunetoe'])    
            else: # load from local file
                dunetoes = Dataset(self.config['inputdir'] + self.config['data locations']['Dunetoe'])
                
            time = dunetoes.variables['time'][:]
            years = num2date(time, dunetoes.variables['time'].units)
            years = [yr.year for yr in years]                    # convert to purely integers indicating the measurement year
            years_filter =  np.isin(years, self.data.years_filtered)
            years_filter_idxs = np.where(years_filter)[0]
                
            dunetoes_y = dunetoes.variables['dune_foot_2nd_deriv'][years_filter_idxs, trsct_idx]
            dunetoes_x = dunetoes.variables['dune_foot_2nd_deriv_cross'][years_filter_idxs, trsct_idx]
            
            if len(years_filter_idxs) < len(self.data.years_filtered):
                rows = len(self.data.years_filtered) - len(years_filter_idxs)
                dunetoes_y = np.append(dunetoes_y, np.empty((rows,1))*np.nan)
                np.append(dunetoes_x, np.empty((rows,1))*np.nan)
                        
            self.dimensions.loc[:, 'Dunetoe_y_der'] = dunetoes_y
            self.dimensions.loc[:, 'Dunetoe_x_der'] = dunetoes_x
                    
    def get_dune_toe_pybeach(self, trsct_idx):
        ####  Pybeach methods ####
        from pybeach.beach import Profile
                
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MHW_x_var']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Landward_x_der']   
    
            # Remove everything outside of boundaries
            elevation = elevation.drop(elevation.index[elevation.index > seaward_x]) # drop everything seaward of seaward boundary
            elevation = elevation.drop(elevation.index[elevation.index < landward_x]).interpolate() # drop everything landward of landward boundary and interpolate remaining data

            if np.isnan(sum(elevation.values)) == False and len(elevation) > 5:
                x_ml = np.array(elevation.index) # pybeach asks ndarray, so convert with np.array(). Note it should be land-left, sea-right otherwise use np.flip()
                y_ml = np.array(elevation.values) 
                
                try:
                    p = Profile(x_ml, y_ml)
                    toe_ml, prob_ml = p.predict_dunetoe_ml(self.config['user defined']['dune toe classifier'])  # predict toe using machine learning model
                    self.dimensions.loc[yr, 'Dunetoe_y_pybeach'] = y_ml[toe_ml[0]]
                    self.dimensions.loc[yr, 'Dunetoe_x_pybeach'] = x_ml[toe_ml[0]]
                except Warning:
                    self.dimensions.loc[yr, 'Dunetoe_y_pybeach'] = np.nan
                    self.dimensions.loc[yr, 'Dunetoe_x_pybeach'] = np.nan

            else:
                self.dimensions.loc[yr, 'Dunetoe_y_pybeach'] = np.nan
                self.dimensions.loc[yr, 'Dunetoe_x_pybeach'] = np.nan

    def get_beach_width_fix(self):
        self.dimensions['Beach_width_fix'] = self.dimensions['MSL_x'] - self.dimensions['Dunetoe_x_fix']
    
    def get_beach_width_var(self):
        self.dimensions['Beach_width_var'] = self.dimensions['MSL_x_var'] - self.dimensions['Dunetoe_x_fix']
            
    def get_beach_width_der(self):
        self.dimensions['Beach_width_der'] = self.dimensions['MSL_x'] - self.dimensions['Dunetoe_x_der'] 
    
    def get_beach_width_der_var(self):
        self.dimensions['Beach_width_der_var'] = self.dimensions['MSL_x_var'] - self.dimensions['Dunetoe_x_der'] 

    def get_beach_gradient_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunetoe_x_fix'] 
                
            self.dimensions.loc[yr,'Beach_gradient_fix'] = get_gradient(elevation, seaward_x, landward_x)
    
    def get_beach_gradient_var(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x_var']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunetoe_x_fix'] 
                
            self.dimensions.loc[yr,'Beach_gradient_var'] = get_gradient(elevation, seaward_x, landward_x)
    
    def get_beach_gradient_der(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunetoe_x_der'] 
                
            self.dimensions.loc[yr,'Beach_gradient_der'] = get_gradient(elevation, seaward_x, landward_x)
       
    def get_dune_front_width_prim_fix(self):
        self.dimensions['Dunefront_width_prim_fix'] = self.dimensions['Dunetoe_x_fix'] - self.dimensions['DuneTop_prim_x']
    
    def get_dune_front_width_prim_der(self):
        self.dimensions['Dunefront_width_prim_der'] = self.dimensions['Dunetoe_x_der'] - self.dimensions['DuneTop_prim_x']
            
    def get_dune_front_width_sec_fix(self):
        self.dimensions['Dunefront_width_sec_fix'] = self.dimensions['Dunetoe_x_fix'] - self.dimensions['DuneTop_sec_x'] 
    
    def get_dune_front_width_sec_der(self):
        self.dimensions['Dunefront_width_sec_der'] = self.dimensions['Dunetoe_x_der'] - self.dimensions['DuneTop_sec_x']
    
    def get_dune_front_gradient_prim_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_prim_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunetoe_x_fix'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_prim_fix'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_prim_der(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_prim_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunetoe_x_der'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_prim_der'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_sec_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_sec_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunetoe_x_fix'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_sec_fix'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_sec_der(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_sec_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunetoe_x_der'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_sec_der'] = get_gradient(elevation, seaward_x, landward_x)
   
    def get_dune_volume_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Dunetoe_x_fix'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'])
            
            self.dimensions.loc[yr,'DuneVol_fix'] = get_volume(elevation, seaward_x, landward_x)
        
    def get_dune_volume_der(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Dunetoe_x_der'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'] )
            
            self.dimensions.loc[yr, 'DuneVol_der'] = get_volume(elevation, seaward_x, landward_x)
  
    def get_intertidal_gradient_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MLW_x_fix']
            landward_x = self.dimensions.loc[yr, 'MHW_x_fix']
                
            self.dimensions.loc[yr, 'Intertidal_gradient_fix'] = get_gradient(elevation, seaward_x, landward_x)
            
    
    def get_intertidal_volume_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'MLW_x_fix'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'MHW_x_fix'] )
            
            self.dimensions.loc[yr, 'Intertidal_volume_fix'] = get_volume(elevation, seaward_x, landward_x)

    def get_intertidal_volume_var(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'MLW_x_var'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'MHW_x_var'] )
            
            self.dimensions.loc[yr, 'Intertidal_volume_var'] = get_volume(elevation, seaward_x, landward_x)

    def get_foreshore_gradient(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'Seaward_x_FS']
            landward_x = self.dimensions.loc[yr, 'Landward_x_bma']
                
            self.dimensions.loc[yr, 'Foreshore_gradient'] = get_gradient(elevation, seaward_x, landward_x)

    def get_foreshore_volume(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Seaward_x_FS'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_bma'] )
            
            self.dimensions.loc[yr, 'Foreshore_volume'] = get_volume(elevation, seaward_x, landward_x)

    def get_active_profile_gradient(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'Seaward_x_AP']
            landward_x = self.dimensions.loc[yr, 'Landward_x_bma']
                
            self.dimensions.loc[yr, 'Active_profile_gradient'] = get_gradient(elevation, seaward_x, landward_x)            
    
    def get_active_profile_volume(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Seaward_x_DoC'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'] )
            
            self.dimensions.loc[yr, 'Active_profile_volume'] = get_volume(elevation, seaward_x, landward_x)
