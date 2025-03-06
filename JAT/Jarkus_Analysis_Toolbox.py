# """
# Created on Tue Nov 19 11:31:25 2019

# @author: cijzendoornvan
# """

""" Includes the most important functionalities of the JAT including 
retrieving data and extracting profile dimensions """

import numpy as np
import pandas as pd
import pickle
import os
from netCDF4 import Dataset, num2date
from JAT.Geometric_functions import find_intersections, get_gradient, get_volume

#################################
####     DATA-EXTRACTION     ####
#################################

class Transects:
    """Loading and plotting transects.
    
    This class provides the functionalities to retrieve the jarkus dataset 
    and filter out the years and locations requested by the user. This 
    includes determining whether the user defined request is available. 
    Additionally, the elevation of each requested transect can be saved and 
    plotted to provide easy access for analysis, and the conversion of the 
    transect number to the alongshore kilometer is provided.
    """
    
    def __init__(self, config): 
        """Initialization

        The initialization loads the jarkus dataset.
    
        Parameters
        ----------
        config : dict
            Configuration that includes the location of the jarkus dataset
        """
        # create a dataset object, based on JARKUS dataset saved locally or on server
        if 'http' in config['data locations']['Jarkus']:        # check whether it's a url
            try:
                self.dataset = Dataset(config['data locations']['Jarkus'])
            except IOError as e:
                print("Unable to open netcdf file containing jarkus dataset - check url under data locations - Jarkus in jarkus.yml file")
        else: # load from local file
            try:
                self.dataset = Dataset(config['inputdir'] + config['data locations']['Jarkus'])
            except IOError as e:
                print("Unable to open netcdf file containing jarkus dataset - check directory under data locations - Jarkus in jarkus.yml file")
        self.variables = self.dataset.variables
        
    def get_years_filtered(self, start_yr, end_yr):
        """Filtering requested years

        All years in the jarkus dataset are extracted and compared to the 
        user-requested years. Only the available (requested) years and their 
        indices are retained.
    
        Parameters
        ----------
        start_yr : int
            Starting year of the user-requested period
        end_yr : int
            Ending year of the user-requested period

        """
        
        time = self.variables['time'][:]                     # retrieve years from jarkus dataset
        years = num2date(time, self.variables['time'].units)
        years = [yr.year for yr in years]                    # convert to purely integers indicating the measurement year
        years_requested = list(range(start_yr, end_yr))
        years_filter =  np.isin(years, years_requested)
        self.years_filtered = np.array(years)[np.nonzero(years_filter)]
        self.years_filtered_idxs = np.where(years_filter)[0]
   
    def get_transects_filtered(self, transects):
        """Filtering requested transects

        It is determined what type of request is made and which transects are 
        associated with this request. Then all transects in the jarkus dataset 
        are extracted and compared to the user-requested years. Only the 
        available (requested) years and their indices are retained.
    
        Parameters
        ----------
        transects : dict
            Part of the configuration file that includes which type of 
            transects are requested (single, multiple, range or all) and (if 
            applicable) which transects are associated with this request.   
        """
        
        ids = self.variables['id'][:] # retrieve transect ids from jarkus dataset
        if transects['type'] == 'all':
            transects_requested = ids
        elif transects['type'] == 'single':
            transects_requested = transects['single']
        elif transects['type'] == 'multiple':
            transects_requested = transects['multiple']
        elif transects['type'] == 'range':
            transects_requested = np.arange(transects['range']['start'], transects['range']['end'], 1)
        else:
            print("Error: define type of transect request to all, single, multiple or range") # give error if type of request is not indicated
        transects_filter = np.isin(ids, transects_requested) # check whether requested transects are available in the jarkus database
        self.transects_filtered = np.array(ids)[np.nonzero(transects_filter)[0]]
        self.transects_filtered_idxs = np.where(transects_filter)[0]
    
    def get_availability(self, config):
        """Getting available years and transects

        This function executes the get_years_filtered and 
        get_transects_filtered functions based on a configuration file 
        containing the requested years and transects.
    
        Parameters
        ----------
        config : dict
            The configuration file that contains the user-requested years and 
            transects
            
        See Also
        --------
        Transects.get_years_filtered
        Transects.get_transects_filtered
        """
        
        self.get_years_filtered(config['years']['start_yr'], config['years']['end_yr'])    
        self.get_transects_filtered(config['transects'])    
        
    def save_elevation_dataframes(self, config):
        """Save elevation of all years for each transect as a dataframe

        The elevation and corresponding cross-shore location of each requested 
        year and requested transect location are saved as a dataframe. Note 
        that each resulting file contains the profiles for all requested years 
        of one requested transect. The function provides the option to use a 
        filter that leaves out profiles when there is no elevation data 
        present between a certain minimum and maximum elevation. This can, 
        for instance, be useful when only the foreshore is studied and all 
        transects that do not have elevation data in this region are 
        redundant. The user-defined values for filter1 are included in the 
        configuration file. Currently this filter does not have an effect on 
        the extraction of the characteristic parameters because these are 
        determined based on the elevation that is directly extracted from the 
        jarkus dataset. Therefore, the default setting for filter1 is that is 
        it not applied (config['user defined']['filter1']['apply']=False), 
        but this could be changed in the future.
    
        Parameters
        ----------
        config : dict
            The configuration file that contains the user-requested years and 
            transects, reference to the jarkus dataset, the filter1 settings 
            and the save locations.
        """        

        crossshore = self.variables['cross_shore'][:]

        for i, trsct_idx in enumerate(self.transects_filtered_idxs): # go through each requested (and filtered) transect. 
            trsct = str(self.transects_filtered[i])
            elevation_dataframe = pd.DataFrame(index=self.years_filtered, columns=crossshore) # create elevation dataframe
            for j, yr_idx in enumerate(self.years_filtered_idxs): # go through each requested (and filtered) year.   
                elevation_dataframe.loc[self.years_filtered[j]] = self.variables['altitude'][yr_idx, trsct_idx, :]  # extract elevation from jarkus dataset for specific transect and year
                
            if config['user defined']['filter1']['apply'] == True: # determine whether filter1 should be applied
                for idx, row in elevation_dataframe.iterrows():
                    if min(row) > config['user defined']['filter1']['min'] or max(row) < config['user defined']['filter1']['max']: # determine whether elevation data is present between min and max value
                        elevation_dataframe.drop(idx, axis=0) # remove elevation if there's no elevation data present
                
            if os.path.isdir(config['outputdir'] + config['save locations']['DirA']) == False: # create folder for elevation dataframes if it does not exist
                os.mkdir(config['outputdir'] + config['save locations']['DirA'])
            elevation_dataframe.to_pickle(config['outputdir'] + config['save locations']['DirA'] + trsct + '_elevation.pickle') # save elevation dataframe
           
    def get_transect_plot(self, config):
        """Save plot with all coastal profiles for each requested transect

        For each requested transect a quickplot is created and saved (as png 
        and picle file) that shows all the requested years. The colors in the 
        plot go from the start year in red to the end year in blue. Currently 
        the axes are set automatically but this can be changed to user-defined 
        limits in the future, which is mostly relevant for single transect 
        plotting.
    
        Parameters
        ----------
        config : dict
            The configuration file that contains the user-requested years and 
            transects, reference to the jarkus dataset and the save locations.
        """    
          
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
                plt.plot(crossshore[~mask], elevation[~mask], color=colorVal, label = str(yr), linewidth = 2.5) # mask nans otherwise plotting goes wrong
            
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
                
            # Save figure as png and pickle in predefined directory
            # the advantage of a pickle file is that the figure can be reloaded and altered
            if os.path.isdir(config['outputdir'] + config['save locations']['DirB']) == False:
                os.mkdir(config['outputdir'] + config['save locations']['DirB'])
            plt.savefig(config['outputdir'] + config['save locations']['DirB'] + 'Transect_' + str(trsct) + '.png')
            pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB'] + 'Transect_' + str(trsct) + '.fig.pickle', 'wb'))
            
            # close figure
            plt.close()
        
    def get_conversion_dicts(self): # Create conversion dictionary
        """Create conversion from transect number to alongshore meter and 
        vice versa

        For each transect number in the jarkus dataset the alongshore 
        kilometer is calculated. A dictionary is created that relates each 
        transect number to its alongshore kilometer. Additionally, a 
        dictionary is created that does the reverse.
             
        Returns
        -------
        dict
            conversion_ids2alongshore: does the conversion from transect 
            number to alongshore meter
            
            conversion_alongshore2ids: does the conversion from alongshore 
            meter to transect number
        """
        
        trscts = self.variables['id'][:] # load all transect numbers
        # create list with the transect numbers that are associated with all coastal section (kustvak) boundaries
        area_bounds = [2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000, 13000000, 14000000, 15000000, 16000000, 17000000, 18000000]
        
        for i, val in enumerate(area_bounds): # Go through each coastal section
            if i == 0: # Flip numbers for first Wadden Island (Rottumerplaat and -oog)
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
    """Extracting characteristic parameters from coastal profiles.
    
    This class provides the functionalities to extract the characteristic 
    parameters requested by the user from transects of the jarkus dataset. 
    Additionally, it provides functions to post-process the outcome of the
    extraction.
    """
    
    def __init__(self, data, config):    
        """Initialization

        The initialization loads the output of the Transects class and 
        configuration file.
    
        Parameters
        ----------
        data : object
            Output resulting from the Transects class
        config : dict
            Configuration file that includes all user-defined settings
        """
        self.dimensions = pd.DataFrame()
        self.data = data
        self.config = config
        self.crossshore = data.variables['cross_shore'][:]
        
    def get_requested_variables(self):
        """Retrieve all variables that are related to the requested 
        characteristic parameters
          
        Returns
        -------
        list
            variables_req: List of all variable that are related to the 
            requested characteristic parameters as included in the 
            configuration file
        
        """    
        self.variables_req = []
        for key in self.config['dimensions']['setting']:
            if self.config['dimensions']['setting'][key] == True:
                self.variables_req.extend(self.config['dimensions']['variables'][key])
                
        return self.variables_req
                
    def get_all_dimensions(self):
        """Extracts all requested characteristic parameters for all requested 
        years and transects.
        
        Checks whether the saving directory is in place and proceeds to go
        through all requested transects. Per characteristic parameter it is 
        checked whether it was requested, and, if so, the values for all 
        requested years are extracted. Ultimately, per transect a dataframe 
        is saved that includes the values of all requested characteristic 
        parameters for all years at that location.
                
        """    
        # Repress errors that occur due to the profiles with many nans
        import warnings # This error occurs due to nan values in less than boolean operations.
        warnings.filterwarnings("ignore", message="invalid value encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
        
        # Check whether saving directory is avaialable, if not, create it.
        if os.path.isdir(self.config['outputdir'] + self.config['save locations']['DirC']) == False:
            os.mkdir(self.config['outputdir'] + self.config['save locations']['DirC'])
        
        # Go through all requested transects            
        for i, trsct_idx in enumerate(self.data.transects_filtered_idxs):
            trsct = str(self.data.transects_filtered[i])
            print("Extracting parameters of transect " + trsct)
            
            pickle_file = self.config['save locations']['DirC'] + 'Transect_' + trsct + '_dataframe.pickle'
            
            # If file is already present, open the dataframe with all characteristic parameters
            if pickle_file in os.listdir(self.config['outputdir'] + self.config['save locations']['DirC']):
                self.dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
            else: # Create dataframe
                self.dimensions = pd.DataFrame({'transect': trsct, 'years':self.data.years_filtered})
                self.dimensions.set_index('years', inplace=True)
            
            # Go through all characteristic parameters to see whether they were requested in the config file
            # If so, execute the function that extracts the parameter
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
        """Creates and saves a dataframe per characteristic parameter from the 
        dataframes with all requested characteristic parameters per transect.

        """
        variable_dataframe = pd.DataFrame({'years': self.data.years_filtered})
        variable_dataframe.set_index('years', inplace=True)   
        
        variables = self.variables_req
        # Go through all requested variables                
        for variable in variables:
            for i, trsct_idx in enumerate(self.data.transects_filtered_idxs):
                # Go through all transects 
                trsct = str(self.data.transects_filtered[i])
                pickle_file = self.config['outputdir'] + self.config['save locations']['DirC'] + 'Transect_' + trsct + '_dataframe.pickle'
                variable_dataframe.loc[:, trsct] = np.nan
                
                # Check whether each corresponding dataframe with characteristic parameters exists, and if so, load it
                if os.path.exists(pickle_file):
                    dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of transect
                    
                # If a requested variable is not available in the dataframe it is set to nan in the new dataframe
                if variable not in dimensions.columns:
                    variable_dataframe.loc[:, trsct] = np.nan
                else:
                    # Go through each year and set the value in the new dataframe.
                    for yr, row in dimensions.iterrows(): 
                        variable_dataframe.loc[yr, trsct] = dimensions.loc[yr, variable] #extract column that corresponds to the requested variable
                print('Extracted transect ' + str(trsct) + ' for variable ' + variable)
            
            # Check whether saving directory exists, create it if necessary and save dataframe with all years and transect locations for one characteristic parameter.                    
            if os.path.isdir(self.config['outputdir'] + self.config['save locations']['DirD']) == False:
                os.mkdir(self.config['outputdir'] + self.config['save locations']['DirD'])
            variable_dataframe.to_pickle(self.config['outputdir'] + self.config['save locations']['DirD'] + variable + '_dataframe' + '.pickle')
            print('The dataframe of ' + variable + ' was saved')

    
    def normalize_dimensions(self):  
        """Normalize the cross-shore location values of all requested 
        characteristic parameters
        
        Normalization of the cross-shore locations is done to make cross-shore
        values between transects comparable. The normalization is executed by 
        subtracting a normalization value from the value of each year of a 
        characteristic parameter. This function provides the option to apply 
        a normalization based on the mean of all the years available for a 
        transect. Additionally, a normalization based on the value of a fixed 
        user-defined year is available. The normalized cross-shore locations 
        are saved as a dataframe.
            
        """
        
        # Get all variables that have to be normalized based on the requirement that _x should be in the column name, 
        # and that change values do not have to be normalized.
        variables = self.variables_req
        normalized_variables = [var for var in variables if '_x' in var and 'change' not in var]            
        for i, variable in enumerate(normalized_variables):      
            pickle_file = self.config['outputdir'] + self.config['save locations']['DirD'] + variable + '_dataframe' + '.pickle'
            dimensions = pickle.load(open(pickle_file, 'rb')) #load pickle of dimensions   
            normalized = dimensions.copy()
            # Retrieve normalization type that should be applied from the configuration file
            norm_type = self.config['user defined']['normalization']['type']
            if norm_type == 'mean':
                for i, col in dimensions.items():
                    # Get the mean cross-shore location per transect and subtract that from the values of the variable for each transect
                    normalized.loc[:, i] = col - col.mean()
            elif norm_type == 'norm_year':
                norm_year = self.config['user defined']['normalization'][' year']    
                for i, col in dimensions.items():
                    # Get norm value for the cross-shore location in the norm year and subtract that from the values of the variable for each transect
                    normalized.loc[:, i] = col - col[norm_year]
            
            normalized.to_pickle(self.config['outputdir'] + self.config['save locations']['DirD'] + variable + '_normalized_dataframe' + '.pickle')
            print('The dataframe of ' + variable + ' was normalized and saved')

    def get_primary_dune_top(self, trsct_idx): 
        """Extract the primary dune top height (DuneTop_prim_y) and 
        cross-shore location (DuneTop_prim_x).
        
        The primary dune top is defined as the most seaward dune top that
        has a height that is larger than a user-defined threshold (default = 
        5 m) and a prominence that is larger than a user-defined value 
        (default = 2.0). This function uses scipy.signal.find_peaks [1]. The 
        prominence of a peak measures how much a peak stands out from the 
        surrounding baseline of the signal and is defined as the vertical 
        distance between the peak and its lowest contour line [2].
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.


        .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
        .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences                
            
        """

        from scipy.signal import find_peaks
        # Go through all years
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            # Extract elevation
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            
            # Find all peaks that meet the user-defined height and prominence
            dune_top_prim = find_peaks(elevation.filled(np.nan), height = self.config['user defined']['primary dune']['height'], prominence = self.config['user defined']['primary dune']['prominence'])

            if len(dune_top_prim[0]) != 0: # If a peak is found in the profile
                # Select the most seaward peak found of the primarypeaks
                dune_top_prim_idx = dune_top_prim[0][-1]
                self.dimensions.loc[yr, 'DuneTop_prim_x'] = self.crossshore[dune_top_prim_idx] 
                self.dimensions.loc[yr, 'DuneTop_prim_y'] = elevation[dune_top_prim_idx]
            else:
                self.dimensions.loc[yr, 'DuneTop_prim_x'] = np.nan
                self.dimensions.loc[yr, 'DuneTop_prim_y'] = np.nan
                
    def get_secondary_dune_top(self, trsct_idx):        
        """Extract the secondary dune top height (DuneTop_sec_y) and 
        cross-shore location (DuneTop_sec_x).
        
        The secondary dune top is defined as the most seaward dune top that
        has a height that is larger than a user-defined threshold (default = 
        3 m) and a prominence that is larger than a user-defined value 
        (default = 0.5) and is located seaward of the primary dune top. 
        This function uses scipy.signal.find_peaks [1]. The prominence of a 
        peak measures how much a peak stands out from the surrounding baseline 
        of the signal and is defined as the vertical distance between the peak 
        and its lowest contour line [2]. The goal of this function is to be
        able to identify embryo dune formation.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_primary_dune_top
        
        
        .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
        .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences                
        
        """
        
        from scipy.signal import find_peaks
        # initliaze location in dataframe
        self.dimensions.loc[:, 'DuneTop_sec_x'] = np.nan
        self.dimensions.loc[:, 'DuneTop_sec_y'] = np.nan
        # Go through all years
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            # Extract elevation
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            
            # Find all peaks that meet the user-defined height and prominence
            dune_top_sec = find_peaks(elevation.filled(np.nan) , height = self.config['user defined']['secondary dune']['height'], prominence = self.config['user defined']['secondary dune']['prominence'])

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
        """Extract the cross-shore location of mean sea level (MSL_x).
        
        The mean sea level is defined as a fixed, user-defined elevation 
        (default = 0 m). The intersections between this elevation and the 
        coastal profile are determined. The most seaward intersection is 
        selected as the cross-shore location if no primary dune top is 
        available. Otherwise, all intersections landward of the primary dune 
        top are filtered out. Then, if the distance between the most seaward 
        and landward intersection is equal or smaller than 100 m the most 
        seaward intersection is selected as the cross-shore MSL location. 
        Otherwise, if the distance is larger than 100 m, only the intersecions 
		that are landwards of the location that is 100 m seaward of the most 
		landward intersection are selected.Of this selection, the most 
		seaward intersection is selected as the cross-shore MSL location. 
		This filtering is necessary to make sure landward intersections 
		behind the dunes and seaward intersections due to the presence of 
		shoals are not selected as the MSL location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_primary_dune_top
        
        Todo
        -----
        Convert default value of 100 meter to user-defined value included in 
        configuration file.

        """
        
        MSL_y = self.config['user defined']['mean sea level'] # in m above reference datum
        
        for i, yr in enumerate(self.data.years_filtered):
            # Go through all years
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            # Find location where MSL intersects with the coastal profile
            intersections = find_intersections(elevation, self.crossshore, MSL_y)
            # Select most seaward intersection if no primary dune top is available.
            if len(intersections) != 0 and np.isnan(self.dimensions.loc[yr, 'DuneTop_prim_x']):
                self.dimensions.loc[yr, 'MSL_x'] = intersections[-1] # get most seaward intersect
            
            # The following filtering is implemented to make sure offshore shallow parts are not identified as MSL. 
            # This is mostly applicable to the Wadden Islands and Zeeland.
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
        """Extract the cross-shore location of mean low water (MLW_x_fix).
        
        The mean low water is defined as a fixed, user-defined elevation 
        (default = -1 m). The intersections between this elevation and the 
        coastal profile are determined. Then, intersections that are further 
        than 250 m seaward of the location of the mean sea level (MSL_x) are
        excluded. This filtering is necessary to make sure seaward 
        intersections caused by for instance the presence of shoals are not 
        selected as the MLW location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level
        
        Todo
        -----
        Convert default value of 250 meter to user-defined value included in 
        configuration file.

        """
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
        """Extract the elevation (MLW_y_var) and cross-shore location 
        (MLW_x_var) of mean low water.
        
        The mean low water is defined as a spatially variable elevation. 
        This elevation is provided per transect location in the jarkus 
        database (determined with tidal modeling). The intersections between 
        this elevation and the coastal profile are determined. Then, 
        intersections that are further than 250 m seaward of the location of 
        the mean sea level (MSL_x) are excluded. This filtering is necessary 
        to make sure seaward intersections caused by for instance the presence 
        of shoals are not selected as the MLW location. Both the cross-shore
        location and variable elevation are saved. Note, that the spatially 
        variable elevation is not variable in time, so each transect has a 
        constant MLW elevation assigned that is constant throughout the years.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level
        
        Todo
        -----
        Convert default value of 250 meter to user-defined value included in 
        configuration file.

        """
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
        """Extract the cross-shore location of mean high water (MHW_x_fix).
        
        The mean high water is defined as a fixed, user-defined 
        elevation (default = + 1 m). The intersections between this elevation 
        and the coastal profile are determined. Then, intersections that are 
        further than 250 m landward of the location of the mean sea level 
        (MSL_x) are excluded. This filtering is necessary to make sure 
        landward intersections behind the dunes are not selected as the MHW 
        location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level
        
        Todo
        -----
        Convert default value of 250 meter to user-defined value included in 
        configuration file.

        """
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
        """Extract the elevation (MHW_y_var) and cross-shore location 
        (MHW_x_var) of mean low water.
        
        The mean high water is defined as a spatially variable elevation. 
        This elevation is provided per transect location in the jarkus 
        database (determined with tidal modeling). The intersections between 
        this elevation and the coastal profile are determined. Then, 
        intersections that are further than 250 m landward of the location of 
        the mean sea level (MSL_x) are excluded. This filtering is necessary 
        to make sure landward intersections behind the dunes are not selected 
        as the MHW location. Both the cross-shore location and variable 
        elevation are saved. Note, that the spatially variable elevation is 
        not variable in time, so each transect has a constant MHW elevation 
        assigned that is constant throughout the years.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level
        
        Todo
        -----
        Convert default value of 250 meter to user-defined value included in 
        configuration file.

        """
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
        """Extract the cross-shore mean sea level location (MSL_x_var) based 
        on the variable mean high and low water.
        
        The mean sea level location is determined by calculating the 
        middle point between the cross-shore location of the variable mean 
        high water and the variable mean low water.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_low_water_variable
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_high_water_variable

        """
        self.dimensions.loc[:, 'MSL_x_var'] = (self.dimensions.loc[:, 'MLW_x_var'] + self.dimensions.loc[:, 'MHW_x_var'])/2 # Base MSL on the varying location of the low and high water line

    def get_intertidal_width_fixed(self):
        """Extract the width of the intertidal area (Intertidal_width_fix).
              
        The width of the intertidal area is determined by calculating 
        the cross-shore distance between the fixed mean low water and the 
        fixed mean high water.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_low_water_fixed
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_high_water_fixed

        """
        
        self.dimensions.loc[:, 'Intertidal_width_fix'] = self.dimensions.loc[:, 'MLW_x_fix'] - self.dimensions.loc[:, 'MHW_x_fix']

    def get_intertidal_width_variable(self):
        """Extract the width of the intertidal area (Intertidal_width_var).
              
        The width of the intertidal area is determined by calculating 
        the cross-shore distance between the variable mean low water and the 
        variable mean high water.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_low_water_variable
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_high_water_variable

        """
        
        self.dimensions.loc[:,'Intertidal_width_var'] = self.dimensions.loc[:,'MLW_x_var'] - self.dimensions.loc[:,'MHW_x_var']
    

    def get_landward_point_variance(self, trsct_idx):
        """Extract the cross-shore location of the landward boundary based on 
        variance (Landward_x_variance).
        
        The landward boundary is determined by calculating the variance of the 
        elevation of a transect location throughout all available years. 
        Stable points are located based on where the variance of the elevation 
        through time is below a user-defined threshold (default = 0.1). The 
        stable points landward of the primary dune top are filtered out and 
        the cross-shore location and elevation of the most seaward stable 
        point is are selected as the landward boundary.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_primary_dune_top
        
        """
        
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
        """Extract the cross-shore location of the landward boundary based on 
        steps in the second derivative method (Landward_x_der) [3].
        
        The landward boundary is determined by finding the peaks with a 
        prominence larger than a fixed threshold (default = +2.4 m). If peaks 
        are found and those peaks are larger than a user-defined elevation 
        (default = 6.0), the cross-shore location of the intersection of this 
        elevation with the coastal profile is the landward boundary. 
        Otherwise, the peaks above the peaks threshold (variable MHW + 
        prominence threshold) are selected and the most seaward selected peak 
        is the landward boundary. If none of these selection cannot be applied 
        a NaN value is inserted. This function uses scipy.signal.find_peaks 
        [1]. The prominence of a peak measures how much a peak stands out from 
        the surrounding baseline of the signal and is defined as the vertical 
        distance between the peak and its lowest contour line [2].

        Parameters
        ----------
        trsct_idx : index of the transect necessary to extract the elevation of the profiles.
        
        Todo
        -----
        Alter based on matlab version
        
                
        .. [3] Diamantidou, E., Santinelli, G., Giardino, A., Stronkhorst, J., & de Vries, S.   "An Automatic Procedure for Dune toe Position Detection: Application to the Dutch Coast."     Journal of Coastal Research, 36(3)(2020): 668-675. https://doi.org/10.2112/JCOASTRES-D-19-00056.1
        .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks
        .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html#scipy.signal.peak_prominences                
        
        """

        from scipy.signal import find_peaks
        
        # Get user-defined values
        height_of_peaks = self.config['user defined']['landward derivative']['min height'] # in meter
        height_constraint = self.config['user defined']['landward derivative']['height constraint'] # in meter
        peaks_threshold = height_of_peaks + self.dimensions['MHW_y_var'].iloc[0]  # adjust this based on matlab version?
        # Go through years
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            # Extract elevation
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            # Find peaks that have a prominence larger than height_of_peaks
            peaks = find_peaks(elevation.filled(np.nan) , prominence = height_of_peaks)[0] 
            # Get elevation of peaks
            peaks = elevation[peaks]
            # Select peaks that are larger or equal to the peaks threshold
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
        """Extract the cross-shore location of the landward boundary based on 
        the boundary between the marine and aeolian zone (Landward_x_bma) [4].
        
        The landward boundary is defined as a fixed, user-defined elevation 
        (default = +2 m). The intersections between this elevation and the 
        coastal profile are determined. Then, the cross-shore location of the 
        most seaward intersection is selected as the landward boundary.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
               
            
        .. [4] De Vries, S., de Schipper, M., Stive, M., & Ranasinghe, R. "Sediment exchange between the sub-aqeous and sub-aerial coastal zones." Coastal Engineering. 2 (2010).
        
        """
    
        bma_y = self.config['user defined']['landward bma']
        # Go through years
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            # Extract elevation
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            # Find intersections
            intersections_bma = find_intersections(elevation, self.crossshore, bma_y)
            if len(intersections_bma) != 0:
                self.dimensions.loc[yr, 'Landward_x_bma'] = intersections_bma[-1] # Select most seaward intersection
            else:
                self.dimensions.loc[yr, 'Landward_x_bma'] = np.nan

    def get_seaward_point_foreshore(self, trsct_idx):
        """Extract the cross-shore location of the seaward foreshore boundary 
        (Seaward_x_FS).
        
        The seaward boundary is defined as a fixed, user-defined elevation 
        (default = -4 m). The intersections between this elevation and the 
        coastal profile are determined. Then, the cross-shore location of the 
        most seaward intersection is selected as the seaward boundary.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
               
        """
        seaward_FS_y = self.config['user defined']['seaward foreshore']
        # Go through years
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            # Extract elevation
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :]
            # Find intersections
            intersections_FS = find_intersections(elevation, self.crossshore, seaward_FS_y)
            if len(intersections_FS) != 0:
                self.dimensions.loc[yr, 'Seaward_x_FS'] = intersections_FS[-1] # Select most seaward intersection
            else:
                self.dimensions.loc[yr, 'Seaward_x_FS'] = np.nan
            
    def get_seaward_point_activeprofile(self, trsct_idx):
        """Extract the cross-shore location of the seaward active profile 
        boundary (Seaward_x_AP).
        
        The seaward boundary is defined as a fixed, user-defined elevation 
        (default = -8 m). The intersections between this elevation and the 
        coastal profile are determined. Then, the cross-shore location of the 
        most seaward intersection is selected as the seaward boundary.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
               
        """
        
        seaward_ActProf_y = self.config['user defined']['seaward active profile']
        # Go through years
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            # Extract elevation
            elevation = self.data.variables['altitude'][yr_idx, trsct_idx, :] 
            # Find intersections
            intersections_AP = find_intersections(elevation, self.crossshore, seaward_ActProf_y)
            if len(intersections_AP) != 0:
                self.dimensions.loc[yr, 'Seaward_x_AP'] = intersections_AP[-1] # Select most seaward intersection
            else:
                self.dimensions.loc[yr, 'Seaward_x_AP'] = np.nan
    
    def get_seaward_point_doc(self, trsct_idx):
        """Extract the cross-shore location (Seaward_x_DoC) of the depth of 
        closure based on the method of Hinton [5].
        
        Approximation of the depth of closure below a user-defined minimum  
        (default = -5.0 m) (Seaward_x_mindepth) where the standard deviation 
        of the elevation through time is below a user-defined value (default 
        = 0.25) for at least a user-defined length (default = 200m), based on 
        the method by Hinton [5]. 

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        Todo
        -----
        Check whether this method gives the same results as the work of
        Nicha Zwarenstein (2021) [6].
                
        
        .. [5] Hinton, Claire L. Decadal morphodynamic behaviour of the Holland shoreface. Diss. Middlesex University, 2000. https://eprints.mdx.ac.uk/id/eprint/6601
        .. [6] Zwarenstein Tutunji, Nicha. "Classification of coastal profile development in the Hoogheemraadschap Hollands Noorderkwartier area: Using advanced data analysis techniques." (2021).

        """

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
        """Extract the cross-shore location of the dune toe (Dunetoe_x_fix).
        
        The dune toe is defined as a fixed, user-defined elevation 
        (default = +3 m). The intersections between this elevation and the 
        coastal profile are determined. Then, the cross-shore location of the 
        most seaward intersection is selected as the dune toe.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
               
        """
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
        """Extract the elevation (Dunetoe_y_der) and cross-shore location
        (Dunetoe_x_der) of the dune toe based on the second derivative 
        method [3]. 
        
        The dune toe elevation is extracted from the repository where the work
        of Diamantidou et al. [3] is saved. First, the method selects part of 
        the coastal profile. This selection is between the landward boundary 
        (get_landward_point_derivative) and the variable MHW. Then, the first 
        and second derivative of this part of the coastal profile is 
        calculated. The most seaward location where the first derivative is 
        lower than 0.001 and the second derivative is lower than 0.01 is 
        selected as the dune toe [3].

        Parameters
        ----------
        trsct_idx : index of the transect necessary to extract the elevation of the profiles.
        
        Todo
        -----
        Alter based on matlab version
        
        See Also
        ---------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_derivative
        
        
        .. [3] Diamantidou, E., Santinelli, G., Giardino, A., Stronkhorst, J., & de Vries, S.   "An Automatic Procedure for Dune toe Position Detection: Application to the Dutch Coast."     Journal of Coastal Research, 36(3)(2020): 668-675. https://doi.org/10.2112/JCOASTRES-D-19-00056.1
        
        """

        ## Variable dunetoe definition based on first and second derivative of profile
        if 'http' in self.config['data locations']['Dunetoe']: # check whether it's a url
            try:
                dunetoes = Dataset(self.config['data locations']['Dunetoe'])   
            except IOError as e:
                print("Unable to open netcdf file containing dunetoes dataset - check url under data locations - Dunetoe in jarkus.yml file")
        else: # load from local file
            try:
                dunetoes = Dataset(self.config['inputdir'] + self.config['data locations']['Dunetoe'])
            except IOError as e:
                print("Unable to open netcdf file containing dunetoes dataset - check directory under data locations - Dunetoe in jarkus.yml file")
            
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
        """Extract the elevation (Dunetoe_y_pybeach) and cross-shore location
        (Dunetoe_x_pybeach) of the dune toe based on pybeach machine learning 
        method [7]_. 
        
        Pybeach provides three different pre-trained machine learning models 
        (barrier-island, wave-embayed and mixed) that can extract the dune toe
        location. Here, the user can define which model to use (default = 
        'mixed') These models were based on the identification of the dune 
        toe by experts. To make the applicaiton of the pybeach machine 
        learning method comparable to the second derivative method a similar 
        reduction of the coastal profile (with a landward and seaward 
        boundary) is executed.

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.

        Todo
        -----
        Alter based on matlab version
        
        See Also
        ---------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_derivative
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative
        
        
        .. [7] Beuzen, Tomas. "pybeach: A Python package for extracting the location of dune toes on beach profile transects." Journal of Open Source Software 4(44) (2019): 1890. https://doi.org/10.21105/joss.01890
        
        """
        
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
        """Extract the width of the beach (Beach_width_fix).
              
        The width of the beach is determined by calculating 
        the cross-shore distance between the fixed mean sea level and the 
        fixed dune toe location.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed

        """
        
        self.dimensions['Beach_width_fix'] = self.dimensions['MSL_x'] - self.dimensions['Dunetoe_x_fix']
    
    def get_beach_width_var(self):
        """Extract the width of the beach (Beach_width_var).
              
        The width of the beach is determined by calculating 
        the cross-shore distance between the variable mean sea level and the 
        fixed dune toe location.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level_variable
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed

        """
        
        self.dimensions['Beach_width_var'] = self.dimensions['MSL_x_var'] - self.dimensions['Dunetoe_x_fix']
            
    def get_beach_width_der(self):
        """Extract the width of the beach (Beach_width_der).
              
        The width of the beach is determined by calculating 
        the cross-shore distance between the fixed mean sea level and the 
        dune toe location based on the second derivative method.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative

        """
                
        self.dimensions['Beach_width_der'] = self.dimensions['MSL_x'] - self.dimensions['Dunetoe_x_der'] 
    
    def get_beach_width_der_var(self):
        """Extract the width of the beach (Beach_width_der_var).
              
        The width of the beach is determined by calculating 
        the cross-shore distance between the variable mean sea level and the 
        dune toe location based on the second derivative method.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level_variable
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative

        """
                     
        self.dimensions['Beach_width_der_var'] = self.dimensions['MSL_x_var'] - self.dimensions['Dunetoe_x_der'] 

    def get_beach_gradient_fix(self, trsct_idx):
        """Extract the gradient of the beach (Beach_gradient_fix).
              
        The gradient of the beach is determined by finding the slope of  
        the line of best fit along the coastal profile between the fixed 
        mean sea level and the fixed dune toe location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed
        JAT.Geometric_functions.get_gradient

        """
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunetoe_x_fix'] 
                
            self.dimensions.loc[yr,'Beach_gradient_fix'] = get_gradient(elevation, seaward_x, landward_x)
    
    def get_beach_gradient_var(self, trsct_idx):
        """Extract the gradient of the beach (Beach_gradient_var).
              
        The gradient of the beach is determined by finding the slope of  
        the line of best fit along the coastal profile between the variable 
        mean sea level and the fixed dune toe location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level_variable
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed
        JAT.Geometric_functions.get_gradient

        """
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x_var']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunetoe_x_fix'] 
                
            self.dimensions.loc[yr,'Beach_gradient_var'] = get_gradient(elevation, seaward_x, landward_x)
    
    def get_beach_gradient_der(self, trsct_idx):
        """Extract the gradient of the beach (Beach_gradient_der).
              
        The gradient of the beach is determined by finding the slope of  
        the line of best fit along the coastal profile between the fixed 
        mean sea level and the second derivative dune toe location.

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative
        JAT.Geometric_functions.get_gradient

        """
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
            
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # Get seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MSL_x']
            # Get landward boundary 
            landward_x = self.dimensions.loc[yr, 'Dunetoe_x_der'] 
                
            self.dimensions.loc[yr,'Beach_gradient_der'] = get_gradient(elevation, seaward_x, landward_x)
       
    def get_dune_front_width_prim_fix(self):
        """Extract the width of the primary dune front 
        (Dunefront_width_prim_fix).
              
        The width of the primary dune front is determined by calculating 
        the cross-shore distance between the cross-shore location of the 
        primary dune top and the fixed dune toe location.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_primary_dune_top
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed

        """
        
        self.dimensions['Dunefront_width_prim_fix'] = self.dimensions['Dunetoe_x_fix'] - self.dimensions['DuneTop_prim_x']
    
    def get_dune_front_width_prim_der(self):
        """Extract the width of the primary dune front 
        (Dunefront_width_prim_der).
              
        The width of the primary dune front is determined by calculating 
        the cross-shore distance between the cross-shore location of the 
        primary dune top and the derivative dune toe location.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_primary_dune_top
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative

        """
        
        self.dimensions['Dunefront_width_prim_der'] = self.dimensions['Dunetoe_x_der'] - self.dimensions['DuneTop_prim_x']
            
    def get_dune_front_width_sec_fix(self):
        """Extract the width of the secondary dune front 
        (Dunefront_width_sec_fix).
              
        The width of the secondary dune front is determined by calculating 
        the cross-shore distance between the cross-shore location of the 
        secondary dune top and the fixed dune toe location.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_secondary_dune_top
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed

        """
        
        self.dimensions['Dunefront_width_sec_fix'] = self.dimensions['Dunetoe_x_fix'] - self.dimensions['DuneTop_sec_x'] 
    
    def get_dune_front_width_sec_der(self):
        """Extract the width of the secondary dune front 
        (Dunefront_width_sec_der).
              
        The width of the secondary dune front is determined by calculating 
        the cross-shore distance between the cross-shore location of the 
        secondary dune top and the derivative dune toe location.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_secondary_dune_top
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative

        """
                
        self.dimensions['Dunefront_width_sec_der'] = self.dimensions['Dunetoe_x_der'] - self.dimensions['DuneTop_sec_x']
    
    def get_dune_front_gradient_prim_fix(self, trsct_idx):
        """Extract the gradient of the primary dune front 
        (Dunefront_gradient_prim_fix).
              
        The gradient of the dune front is determined by finding the slope of  
        the line of best fit along the coastal profile between the primary 
        dune top  and the fixed dune toe location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_primary_dune_top
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed
        JAT.Geometric_functions.get_gradient

        """
        
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'Dunetoe_x_fix']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'DuneTop_prim_x'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_prim_fix'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_prim_der(self, trsct_idx):
        """Extract the gradient of the primary dune front 
        (Dunefront_gradient_prim_der).
              
        The gradient of the dune front is determined by finding the slope of  
        the line of best fit along the coastal profile between the primary 
        dune top  and the derivative dune toe location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
                
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_primary_dune_top
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative
        JAT.Geometric_functions.get_gradient

        """
        
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_prim_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunetoe_x_der'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_prim_der'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_sec_fix(self, trsct_idx):
        """Extract the gradient of the secondary dune front 
        (Dunefront_gradient_sec_fix).
              
        The gradient of the dune front is determined by finding the slope of  
        the line of best fit along the coastal profile between the secondary 
        dune top and the fixed dune toe location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_secondary_dune_top
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed
        JAT.Geometric_functions.get_gradient

        """
                
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_sec_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunetoe_x_fix'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_sec_fix'] = get_gradient(elevation, seaward_x, landward_x)
 
    def get_dune_front_gradient_sec_der(self, trsct_idx):
        """Extract the gradient of the secondary dune front 
        (Dunefront_gradient_sec_der).
              
        The gradient of the dune front is determined by finding the slope of  
        the line of best fit along the coastal profile between the secondary 
        dune top and the derivative dune toe location.

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_secondary_dune_top
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative
        JAT.Geometric_functions.get_gradient

        """
                      
        for i, yr in enumerate(self.data.years_filtered):
             yr_idx = self.data.years_filtered_idxs[i]
             
             elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
             
             # Get seaward boundary
             seaward_x = self.dimensions.loc[yr, 'DuneTop_sec_x']
             # Get landward boundary 
             landward_x = self.dimensions.loc[yr, 'Dunetoe_x_der'] 
                 
             self.dimensions.loc[yr,'Dunefront_gradient_sec_der'] = get_gradient(elevation, seaward_x, landward_x)
   
    def get_dune_volume_fix(self, trsct_idx):
        """Extract the dune volume (DuneVol_fix).
              
        The dune volume is determined by finding the surface under the coastal 
        profile between the landward boundary based on the variance and the 
		fixed dune toe location.

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_variance
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed
        JAT.Geometric_functions.get_volume

        """
                
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Dunetoe_x_fix'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'])
            
            self.dimensions.loc[yr,'DuneVol_fix'] = get_volume(elevation, seaward_x, landward_x)
        
    def get_dune_volume_der(self, trsct_idx):
        """Extract the dune volume (DuneVol_der).
              
        The dune volume is determined by finding the surface under the coastal 
        profile between the landward boundary based on the variance and the 
		derivative dune toe location.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_variance
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed
        JAT.Geometric_functions.get_volume

        """        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Dunetoe_x_der'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'] )
            
            self.dimensions.loc[yr, 'DuneVol_der'] = get_volume(elevation, seaward_x, landward_x)
  
    def get_intertidal_gradient_fix(self, trsct_idx):
        """Extract the gradient of the intertidal area 
        (Intertidal_gradient_fix).
              
        The gradient of the intertidal area is determined by finding the slope 
        of the line of best fit along the coastal profile between the fixed 
        mean low water and the fixed mean high water.

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_low_water_fixed
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_high_water_fixed
        JAT.Geometric_functions.get_gradient

        """
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'MLW_x_fix']
            landward_x = self.dimensions.loc[yr, 'MHW_x_fix']
                
            self.dimensions.loc[yr, 'Intertidal_gradient_fix'] = get_gradient(elevation, seaward_x, landward_x)
            
    
    def get_intertidal_volume_fix(self, trsct_idx):
        """Extract the volume of the intertidal area (Intertidal_volume_fix).
              
        The intertidal area volume is determined by finding the surface under 
        the coastal profile between the fixed mean low water and the fixed 
        mean high water.
        
        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.

        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_low_water_fixed
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_high_water_fixed
        JAT.Geometric_functions.get_volume

        """
        for i, yr in enumerate(self.data.years_filtered):            
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'MLW_x_fix'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'MHW_x_fix'] )
            
            self.dimensions.loc[yr, 'Intertidal_volume_fix'] = get_volume(elevation, seaward_x, landward_x)

    def get_intertidal_volume_var(self, trsct_idx):
        """Extract the volume of the intertidal area (Intertidal_volume_var).
              
        The intertidal area volume is determined by finding the surface under 
        the coastal profile between the variable mean low water and the 
        variable mean high water.

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_low_water_variable
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_high_water_variable
        JAT.Geometric_functions.get_volume

        """
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'MLW_x_var'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'MHW_x_var'] )
            
            self.dimensions.loc[yr, 'Intertidal_volume_var'] = get_volume(elevation, seaward_x, landward_x)

    def get_foreshore_gradient(self, trsct_idx):
        """Extract the gradient of the foreshore (Foreshore_gradient).
              
        The gradient of the foreshore is determined by finding the slope 
        of the line of best fit along the coastal profile between the seaward 
        foreshore boundary and the landward boundary between the marine and 
        aeolian zone (BMA).

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_seaward_point_foreshore
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_bma
        JAT.Geometric_functions.get_gradient

        """
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'Seaward_x_FS']
            landward_x = self.dimensions.loc[yr, 'Landward_x_bma']
                
            self.dimensions.loc[yr, 'Foreshore_gradient'] = get_gradient(elevation, seaward_x, landward_x)

    def get_foreshore_volume(self, trsct_idx):
        """Extract the volume of the foreshore (Foreshore_volume).
              
        The foreshore volume is determined by finding the surface under 
        the coastal profile between the seaward foreshore boundary and the 
        landward boundary between the marine and aeolian zone (BMA).

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_seaward_point_foreshore
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_bma
        JAT.Geometric_functions.get_volume

        """
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Seaward_x_FS'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_bma'] )
            
            self.dimensions.loc[yr, 'Foreshore_volume'] = get_volume(elevation, seaward_x, landward_x)

    def get_active_profile_gradient(self, trsct_idx):
        """Extract the gradient of the active profile 
        (Active_profile_gradient).
              
        The gradient of the active profile is determined by finding the slope 
        of the line of best fit along the coastal profile between the seaward 
        active profile boundary and the landward boundary between the marine 
        and aeolian zone (BMA).

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_seaward_point_activeprofile
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_bma
        JAT.Geometric_functions.get_gradient

        """
        
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
                
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
            
            # dimensions used as landward and seaward boundary
            seaward_x = self.dimensions.loc[yr, 'Seaward_x_AP']
            landward_x = self.dimensions.loc[yr, 'Landward_x_bma']
                
            self.dimensions.loc[yr, 'Active_profile_gradient'] = get_gradient(elevation, seaward_x, landward_x)            
    
    def get_active_profile_volume(self, trsct_idx):
        """Extract the volume of the active profile (Active_profile_volume).
              
        The volume of the active profile is determined by finding the surface 
        under the coastal profile between the seaward active profile boundary 
        and the landward boundary between the marine and aeolian zone (BMA).

        Parameters
        ----------
        trsct_idx : int
            index of the transect necessary to extract the elevation of the profiles.
        
        See also
        --------
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_seaward_point_activeprofile
        JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_bma
        JAT.Geometric_functions.get_volume

        """
                
        for i, yr in enumerate(self.data.years_filtered):
            yr_idx = self.data.years_filtered_idxs[i]
             
            elevation = pd.DataFrame(self.data.variables['altitude'][yr_idx, trsct_idx, :], index = self.crossshore) 
                 
            # Get seaward boundary
            seaward_x = np.ceil(self.dimensions.loc[yr, 'Seaward_x_DoC'])
            # Get landward boundary 
            landward_x = np.floor(self.dimensions.loc[yr, 'Landward_x_variance'] )
            
            self.dimensions.loc[yr, 'Active_profile_volume'] = get_volume(elevation, seaward_x, landward_x)