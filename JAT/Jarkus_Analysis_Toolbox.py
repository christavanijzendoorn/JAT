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

"""
Created on Tue Nov 19 11:31:25 2019

@author: cijzendoornvan
"""

import numpy as np
import pandas as pd
import pickle
import os
import xarray as xr

#################################
####     DATA-EXTRACTION     ####
#################################

class Transects:

    def __init__(self, config): 
        # create a dataset object, based on locally saved JARKUS dataset
        self.dataset = xr.open_dataset(config['root'] + config['data locations']['DirJK'])
        self.variables = self.dataset.variables
        
    def get_years_filtered(self, start_yr, end_yr):
        time = self.variables['time'].values                     # retrieve years from jarkus dataset
        years = pd.to_datetime(time).year                        # convert to purely integers indicating the measurement year
        years_requested = list(range(start_yr, end_yr))
        years_filter =  np.isin(years, years_requested)
        self.years_filtered = np.array(years)[np.nonzero(years_filter)[0]]
        self.years_filtered_idxs = np.where(years_filter)[0]
   
    def get_transects_filtered(self, transects_requested, execute_all_transects):
        ids = self.variables['id'].values                              # retrieve transect ids from jarkus dataset
        if execute_all_transects == True:
            transects_requested = ids
        transects_filter = np.isin(ids, transects_requested)
        self.transects_filtered = np.array(ids)[np.nonzero(transects_filter)[0]]
        self.transects_filtered_idxs = np.where(transects_filter)[0]
    
    def get_availability(self, start_yr, end_yr, transects_requested, execute_all_transects):
        self.get_years_filtered(start_yr, end_yr)    
        self.get_transects_filtered(transects_requested, execute_all_transects)    
        
    def save_elevation_dataframes(self, config, apply_filter1=''):
                
        crossshore = self.variables['cross_shore'].values

        for i, trsct_idx in enumerate(self.transects_filtered_idxs):
            trsct = str(self.transects_filtered[i])
            elevation_dataframe = pd.DataFrame(index=self.years_filtered, columns=crossshore)
            #!!! When searching for a selection of years after 1965 there is a problem here with indexing!! 
            for j, yr_idx in enumerate(self.years_filtered_idxs):   
                elevation_dataframe.loc[self.years_filtered[j]] = self.variables['altitude'].values[yr_idx, trsct_idx, :]  # elevation of profile point
                
            if apply_filter1 == 'yes':
                for idx, row in elevation_dataframe.iterrows():
                    if min(row) > config['user defined']['filter1']['min'] or max(row) < config['user defined']['filter1']['max']:
                        elevation_dataframe.drop(idx, axis=0)
                
            elevation_dataframe.to_pickle(config['root'] + config['save locations']['DirA'] + trsct + '_elevation.pickle')
           
    def get_transect_plot(self, config):
        
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        
        crossshore = self.variables['cross_shore'].values
        
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
                elevation = self.variables['altitude'].values[yr_idx, trsct_idx, :]
                mask = np.isfinite(elevation)
                plt.plot(crossshore[mask], elevation[mask], color=colorVal, label = str(yr), linewidth = 2.5)
            
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
            plt.savefig(config['root'] + config['save locations']['DirB'] + 'Transect_' + str(trsct) + '.png')
            pickle.dump(fig, open(config['root'] + config['save locations']['DirB'] + 'Transect_' + str(trsct) + '.fig.pickle', 'wb'))
            
            plt.close()
        
    def get_conversion_dicts(self): # Create conversion dictionary
        trscts = self.variables['id'].values  
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

def find_intersections(elevation, crossshore, y_value):
    value_vec = np.array([y_value] * len(elevation))
    elevation = pd.Series(elevation).interpolate().tolist()
    
    diff = np.nan_to_num(np.diff(np.sign(elevation - value_vec)))
    intersection_idxs = np.nonzero(diff)
    intersection_x = np.array([crossshore[idx] for idx in intersection_idxs[0]])
    
    return intersection_x

def zero_runs(y_der, threshold_zeroes):                    
    # Create an array that is 1 where y_der is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(y_der, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    zero_sections = np.where(absdiff == 1)[0].reshape(-1, 2)                     
    zero_section_len = zero_sections[:,1] - zero_sections[:,0]
    
    zero_sections = zero_sections[zero_section_len > threshold_zeroes]
            
    return zero_sections

def get_gradient(elevation, seaward_x, landward_x):

    # Remove everything outside of boundaries
    elevation = elevation.drop(elevation.index[elevation.index > seaward_x]) # drop everything seaward of seaward boundary
    elevation = elevation.drop(elevation.index[elevation.index < landward_x]).interpolate() # drop everything landward of landward boundary and interpolate remaining data
    
    # remove nan values otherqise polyfit does not work
    elevation = elevation.dropna(axis=0)
    
    # Calculate gradient for domain
    if sum(elevation.index) == 0:
        gradient = np.nan
    elif pd.isnull(seaward_x) or pd.isnull(landward_x):
        gradient = np.nan
    elif pd.isnull(elevation.first_valid_index()) or pd.isnull(elevation.last_valid_index()):
        gradient = np.nan 
    elif elevation.first_valid_index() > landward_x or elevation.last_valid_index() < seaward_x:
        gradient = np.nan
    else:
        gradient = np.polyfit(elevation.index, elevation.values, 1)[0]    
            
    return gradient

def get_volume(elevation, seaward_x, landward_x):
    from scipy import integrate
        
    if pd.isnull(seaward_x) == True or pd.isnull(landward_x) == True:
        volume = np.nan
    elif pd.isnull(elevation.first_valid_index()) == True or pd.isnull(elevation.last_valid_index()) == True:
        volume = np.nan    
    elif elevation.first_valid_index() > landward_x or elevation.last_valid_index() < seaward_x:
        volume = np.nan
    else:
        # Remove everything outside of boundaries
        elevation = elevation.drop(elevation.index[elevation.index > seaward_x]) # drop everything seaward of seaward boundary
        elevation = elevation.drop(elevation.index[elevation.index < landward_x]).interpolate() # drop everything landward of landward boundary and interpolate remaining data
        
        if elevation.empty == False:
            volume_y = elevation - elevation.min()
            # volume_trapz = np.trapz(volume_y, x = volume_y.index)
            volume_simps = integrate.simps(volume_y.values.transpose(), x = volume_y.index)
            volume = volume_simps
        else:
            volume = np.nan
    
    return volume

    
class Extraction:
    
    def __init__(self, data, config):    
        self.dimensions = pd.DataFrame()
        self.data = data
        self.config = config
        self.crossshore = data.variables['cross_shore'].values
        
    def get_requested_variables(self):
        self.variables_req = []
        for key in self.config['dimensions']['setting']:
            if self.config['dimensions']['setting'][key] == True:
                self.variables_req.extend(self.config['dimensions']['variables'][key])
                
        return self.variables_req
                
    def get_all_dimensions(self):
        for i, trsct_idx in enumerate(self.data.transects_filtered_idxs):
            trsct = str(self.data.transects_filtered[i])
            
            pickle_file = self.config['save locations']['DirC'] + 'Transect_' + trsct + '_dataframe.pickle'
    
            if pickle_file in os.listdir(self.config['root'] + self.config['save locations']['DirC']):
                self.dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
            else:
                self.dimensions = pd.DataFrame({'transect': trsct, 'years':self.data.years_filtered})
                self.dimensions.set_index('years', inplace=True)
            
            if self.config['dimensions']['setting']['dune_height_and_location'] == True:
                self.get_dune_height_and_location(trsct_idx)

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
                
            if self.config['dimensions']['setting']['dune_foot_fixed'] == True:
                self.get_dune_foot_fixed(trsct_idx)       
            if self.config['dimensions']['setting']['dune_foot_derivative'] == True:
                self.get_dune_foot_derivative(trsct_idx)     
            if self.config['dimensions']['setting']['dune_foot_pybeach'] == True:
                self.get_dune_foot_pybeach(trsct_idx)      
                
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
                
            if self.config['dimensions']['setting']['setting']['dune_front_gradient_prim_fix'] == True:
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
            self.dimensions.to_pickle(self.config['root'] + pickle_file)
            
    def get_dataframe_per_dimension(self):
        variable_dataframe = pd.DataFrame({'years': self.data.years_filtered})
        variable_dataframe.set_index('years', inplace=True)   
        
        variables = self.variables_req
                
        for variable in variables:
            for i, trsct_idx in enumerate(self.data.transects_filtered_idxs):
                trsct = str(self.data.transects_filtered[i])
                pickle_file = self.config['root'] + self.config['save locations']['DirC'] + 'Transect_' + trsct + '_dataframe.pickle'
                variable_dataframe[trsct] = np.nan
        
                if os.path.exists(pickle_file):
                    dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
                if variable not in dimensions.columns:
                    variable_dataframe.loc[trsct] = np.nan
                else:
                    for yr, row in dimensions.iterrows(): 
                        variable_dataframe.loc[yr, trsct] = dimensions.loc[yr, variable] #extract column that corresponds to the requested variable
                print('Extracted transect ' + str(trsct) + ' for variable ' + variable)
                    
            variable_dataframe.to_pickle(self.config['root'] + self.config['save locations']['DirD'] + variable + '_dataframe' + '.pickle')
            print('The dataframe of ' + variable + ' was saved')

    
    def normalize_dimensions(self):
        
        norm_year = self.config['user defined']['normalization year']
        variables = self.variables_req
        
        # Get all variables that have to be normalized based on the requirement that _x should be in the column name, 
        # and that change values do not have to be normalized.
        normalized_variables = [var for var in variables if '_x' in var and 'change' not in var]
        
        for i, variable in enumerate(normalized_variables):      
            pickle_file = self.config['root'] + self.config['save locations']['DirD'] + variable + '_dataframe' + '.pickle'
            dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions   
            normalized = dimensions.copy()
            normalization_values = dimensions.loc[norm_year] 
            
            for trsct in normalized.columns:
                # Get norm value for the cross-shore location in the norm year and subtract that from the values of the variable for each transect
                normalized[trsct] = normalized[trsct] - normalization_values.loc[trsct] 
            
            normalized.to_pickle(self.config['root'] + self.config['save locations']['DirD'] + variable + '_normalized_dataframe' + '.pickle')
            print('The dataframe of ' + variable + ' was normalized and saved')

    def get_dune_height_and_location(self, trsct_idx):        
        from scipy.signal import find_peaks
        # Documentation:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences    
        # The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is defined as the vertical distance between the peak and its lowest contour line.
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
            
            dune_top_prim = find_peaks(elevation, height = 5, prominence = 2.0) # , distance = 5
            dune_top_sec = find_peaks(elevation, height = 3, prominence = 0.5) # , distance = 5

            if len(dune_top_prim[0]) != 0: # If a peak is found in the profile
                # Select the most seaward peak found of the primary and secondary peaks
                dune_top_prim_idx = dune_top_prim[0][-1]
                dune_top_sec_idx = dune_top_sec[0][-1]
                if  dune_top_sec_idx <= dune_top_prim_idx: 
                    # If most seaward secondary peak is located at the same place or landward of the most seaward primary peak
                    self.dimensions.loc[yr, 'DuneTop_prim_x'] = self.crossshore[dune_top_prim_idx]  # Save the primary peak location
                    self.dimensions.loc[yr, 'DuneTop_prim_y'] = elevation[dune_top_prim_idx]
                    #Assume that there is no seaward secondary peak, so no value filled in (i.e. it stays nan).
                else:            
                    # Otherwise save both the primary and secondary peak location
                    self.dimensions.loc[yr, 'DuneTop_prim_x'] = self.crossshore[dune_top_prim_idx] 
                    self.dimensions.loc[yr, 'DuneTop_prim_y'] = elevation[dune_top_prim_idx]
                    self.dimensions.loc[yr, 'DuneTop_sec_x'] = self.crossshore[dune_top_sec_idx]
                    self.dimensions.loc[yr, 'DuneTop_sec_y'] = elevation[dune_top_sec_idx]
            else:
                self.dimensions.loc[yr, 'DuneTop_prim_x'] = np.nan
                self.dimensions.loc[yr, 'DuneTop_prim_y'] = np.nan
                self.dimensions.loc[yr, 'DuneTop_sec_x'] = np.nan
                self.dimensions.loc[yr, 'DuneTop_sec_y'] = np.nan
                
    def get_mean_sea_level(self, trsct_idx):
        MSL_y = self.config['user defined']['mean sea level'] # in m above reference datum
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
            
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
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
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
        MLW_y_variable   = self.data.variables['mean_low_water'].values[trsct_idx]     # gets the mean low water level as included in the Jarkus dataset, so it varies per location
        self.dimensions.loc[:, 'MLW_y_var'] = MLW_y_variable     # save assumed mean low water level as extracted from the jarkus dataset
            
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
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
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
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
        MHW_y_variable   = self.data.variables['mean_high_water'].values[trsct_idx]     # gets the mean low water level as included in the Jarkus dataset, so it varies per location
        self.dimensions.loc[:, 'MHW_y_var'] = MHW_y_variable     # save assumed mean low water level as extracted from the jarkus dataset
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
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
        
        elevation = pd.DataFrame(self.data.variables['altitude'].values[:, trsct_idx, :], columns = self.crossshore)
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
        peaks_threshold = height_of_peaks + self.dimensions['MHW_y_var'].values[0]
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
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
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :]
            intersections_bma = find_intersections(elevation, self.crossshore, bma_y)
            if len(intersections_bma) != 0:
                self.dimensions.loc[yr, 'Landward_x_bma'] = intersections_bma[-1]
            else:
                self.dimensions.loc[yr, 'Landward_x_bma'] = np.nan

    def get_seaward_point_foreshore(self, trsct_idx):
        seaward_FS_y = self.config['user defined']['seaward foreshore']
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[0][i]
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :] 
            intersections_FS = find_intersections(elevation, self.crossshore, seaward_FS_y)
            if len(intersections_FS) != 0:
                self.dimensions.loc[yr, 'Seaward_x_FS'] = intersections_FS[-1]
            else:
                self.dimensions.loc[yr, 'Seaward_x_FS'] = np.nan
            
    def get_seaward_point_activeprofile(self, trsct_idx):
        seaward_ActProf_y = self.config['user defined']['seaward active profile']
    
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :] 
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
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :] 
            intersections_mindepth = find_intersections(elevation, self.crossshore, min_depth)
            if len(intersections_mindepth) != 0:
                self.dimensions.loc[yr, 'Seaward_x_mindepth'] = intersections_mindepth[-1]
            else:
                self.dimensions.loc[yr, 'Seaward_x_mindepth'] = np.nan
        
        elevation = pd.DataFrame(self.data.variables['altitude'].values[:, trsct_idx, :], columns = self.crossshore)
                
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

    def get_dune_foot_fixed(self, trsct_idx):
        #### Fixed dunefoot definition ####
        DF_fixed_y = self.config['user defined']['dune foot fixed'] # in m above reference datum
    
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[0][i]
            
            elevation = self.data.variables['altitude'].values[yr_idx, trsct_idx, :] 
            intersections_DF = find_intersections(elevation, self.crossshore, DF_fixed_y)
            if len(intersections_DF) != 0:
                self.dimensions.loc[yr, 'Dunefoot_x_fix'] = intersections_DF[-1]
            else:
                self.dimensions.loc[yr, 'Dunefoot_x_fix'] = np.nan
            
    def get_dune_foot_derivative(self, trsct_idx):        
            ####  Derivative method - E.D. ####
            ###################################
            ## Variable dunefoot definition based on first and second derivative of profile
    
            dunefoots = xr.open_dataset(self.config['root'] + self.config['data locations']['DirDF'])
            dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'].values[self.data.years_filtered_idxs, trsct_idx][0] 
            dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'].values[self.data.years_filtered_idxs, trsct_idx][0] 
            
            self.dimensions.loc[:, 'Dunefoot_y_der'] = dunefoots_y
            self.dimensions.loc[:, 'Dunefoot_x_der'] = dunefoots_x
                    
    def get_dune_foot_pybeach(self, trsct_idx):
        ####  Pybeach methods ####
        from pybeach.beach import Profile
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
                
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
                
                p = Profile(x_ml, y_ml)
                toe_ml, prob_ml = p.predict_dunetoe_ml('mixed_clf')  # predict toe using machine learning model
                
                self.dimensions.loc[yr, 'Dunefoot_y_pybeach_mix'] = y_ml[toe_ml[0]]
                self.dimensions.loc[yr, 'Dunefoot_x_pybeach_mix'] = x_ml[toe_ml[0]]
            else:
                self.dimensions.loc[yr, 'Dunefoot_y_pybeach_mix'] = np.nan
                self.dimensions.loc[yr, 'Dunefoot_x_pybeach_mix'] = np.nan

    def get_beach_width_fix(self):
        self.dimensions['Beach_width_fix'] = self.dimensions['MSL_x'] - self.dimensions['Dunefoot_x_fix']
    
    def get_beach_width_var(self):
        self.dimensions['Beach_width_var'] = self.dimensions['MSL_x_var'] - self.dimensions['Dunefoot_x_fix']
            
    def get_beach_width_der(self):
        self.dimensions['Beach_width_der'] = self.dimensions['MSL_x'] - self.dimensions['Dunefoot_x_der'] 
    
    def get_beach_width_der_var(self):
        self.dimensions['Beach_width_der_var'] = self.dimensions['MSL_x_var'] - self.dimensions['Dunefoot_x_der'] 

    def get_beach_gradient_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunefoot_x_fix'] 
                
            self.dimensions.loc[yr,'Beach_gradient_fix'] = get_gradient(elevation, seaward_x, landward_x)
    
    def get_beach_gradient_var(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x_var']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunefoot_x_fix'] 
                
            self.dimensions.loc[yr,'Beach_gradient_var'] = get_gradient(elevation, seaward_x, landward_x)
    
    def get_beach_gradient_der(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunefoot_x_der'] 
                
            self.dimensions.loc[yr,'Beach_gradient_der'] = get_gradient(elevation, seaward_x, landward_x)
       
    def get_dune_front_width_prim_fix(self):
        self.dimensions['Dunefront_width_prim_fix'] = self.dimensions['Dunefoot_x_fix'] - self.dimensions['DuneTop_prim_x']
    
    def get_dune_front_width_prim_der(self):
        self.dimensions['Dunefront_width_prim_der'] = self.dimensions['Dunefoot_x_der'] - self.dimensions['DuneTop_prim_x']
            
    def get_dune_front_width_sec_fix(self):
        self.dimensions['Dunefront_width_sec_fix'] = self.dimensions['Dunefoot_x_fix'] - self.dimensions['DuneTop_sec_x'] 
    
    def get_dune_front_width_sec_der(self):
        self.dimensions['Dunefront_width_sec_der'] = self.dimensions['Dunefoot_x_der'] - self.dimensions['DuneTop_sec_x']
    
    def get_dune_front_gradient_prim_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_prim_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunefoot_x_fix'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_prim_fix'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_prim_der(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_prim_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunefoot_x_der'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_prim_der'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_sec_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_sec_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunefoot_x_fix'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_sec_fix'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_sec_der(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_sec_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunefoot_x_der'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_sec_der'] = get_gradient(elevation, seaward_x, landward_x)
   
    def get_dune_volume_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Dunefoot_x_fix'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'])
            
            self.dimensions.loc[yr,'DuneVol_fix'] = get_volume(elevation, seaward_x, landward_x)
        
    def get_dune_volume_der(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Dunefoot_x_der'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'] )
            
            self.dimensions.loc[yr, 'DuneVol_der'] = get_volume(elevation, seaward_x, landward_x)
  
    def get_intertidal_gradient_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MLW_x_fix']
            landward_x = self.dimensions.loc[yr, 'MHW_x_fix']
                
            self.dimensions.loc[yr, 'Intertidal_gradient_fix'] = get_gradient(elevation, seaward_x, landward_x)
            
    
    def get_intertidal_volume_fix(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'MLW_x_fix'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'MHW_x_fix'] )
            
            self.dimensions.loc[yr, 'Intertidal_volume_fix'] = get_volume(elevation, seaward_x, landward_x)

    def get_intertidal_volume_var(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'MLW_x_var'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'MHW_x_var'] )
            
            self.dimensions.loc[yr, 'Intertidal_volume_var'] = get_volume(elevation, seaward_x, landward_x)

    def get_foreshore_gradient(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'Seaward_x_FS']
            landward_x = self.dimensions.loc[yr, 'Landward_x_bma']
                
            self.dimensions.loc[yr, 'Foreshore_gradient'] = get_gradient(elevation, seaward_x, landward_x)

    def get_foreshore_volume(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Seaward_x_FS'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_bma'] )
            
            self.dimensions.loc[yr, 'Foreshore_volume'] = get_volume(elevation, seaward_x, landward_x)

    def get_active_profile_gradient(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'Seaward_x_AP']
            landward_x = self.dimensions.loc[yr, 'Landward_x_bma']
                
            self.dimensions.loc[yr, 'Active_profile_gradient'] = get_gradient(elevation, seaward_x, landward_x)            
    
    def get_active_profile_volume(self, trsct_idx):
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'].values[yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Seaward_x_DoC'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'] )
            
            self.dimensions.loc[yr, 'Active_profile_volume'] = get_volume(elevation, seaward_x, landward_x)
