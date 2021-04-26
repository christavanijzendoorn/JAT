# -*- coding: utf-8 -*-
# """
# Created on Fri Apr 16 16:58:13 2021

# @author: cijzendoornvan
# """

"""
Provides basic geometric functions that execute calculations based on the 
coastal profile.
"""

import numpy as np
import pandas as pd

def find_intersections(elevation, crossshore, y_value):
    """Find cross-shore location of intersection between profile and 
    horizontal line at a fixed elevation.

    Parameters
    ----------
    elevation : np.array
        np.array containing the elevation of the coastal profile in meters.
    crossshore : np.array
        np.array containing the crossshore location in meters.
    y_value : float
        Elevation of the horizontal line in meters.
        
    Returns
    -------
    int
        intersection_x: Cross-shore location of the intersection between the 
        coastal profile and horizontal line.
    
    See Also
    --------
    Jarkus_Analysis_Toolbox.Extraction
        
    """    
    
    value_vec = np.array([y_value] * len(elevation))
    elevation = pd.Series(elevation).interpolate().tolist()
        
    with np.errstate(invalid='ignore'):
        diff = np.nan_to_num(np.diff(np.sign(elevation - value_vec)))
    intersection_idxs = np.nonzero(diff)
    intersection_x = np.array([crossshore[idx] for idx in intersection_idxs[0]])
    
    return intersection_x

def get_gradient(elevation, seaward_x, landward_x):
    """Find gradient of a profile between two points
    
    The gradient of a coastal profile is determined by finding the slope of 
    the line of best fit along the elevation between a landward and seaward
    boundary. 

    Parameters
    ----------
    elevation : np.array
        np.array containing the elevation of the coastal profile in meters
    seaward_x : float or int
        Cross-shore seaward boundary
    landward_x : float or int
        Cross-shore seaward boundary
        
    Returns
    -------
    float
        gradient: slope of the best fit line
    
    See Also
    --------
    Jarkus_Analysis_Toolbox.Extraction
        
    """
    
    # Remove all elevation outside of boundaries
    elevation = elevation.drop(elevation.index[elevation.index > seaward_x]) # drop everything seaward of seaward boundary
    elevation = elevation.drop(elevation.index[elevation.index < landward_x]).interpolate() # drop everything landward of landward boundary and interpolate remaining data
    
    # remove nan values otherwise polyfit does not work
    elevation = elevation.dropna(axis=0)
    
    # Check availability of elevation data
    if sum(elevation.index) == 0:
        gradient = np.nan
    # Assign nan if one of the boundaries is nan
    elif pd.isnull(seaward_x) or pd.isnull(landward_x):
        gradient = np.nan
    # Check availability of elevation data
    elif pd.isnull(elevation.first_valid_index()) or pd.isnull(elevation.last_valid_index()):
        gradient = np.nan 
    elif elevation.first_valid_index() > landward_x or elevation.last_valid_index() < seaward_x:
        gradient = np.nan
    else:
        # Calculate gradient for domain
        gradient = np.polyfit(elevation.index, elevation.values, 1)[0]    
            
    return gradient

def get_volume(elevation, seaward_x, landward_x):
    """Determine volume under coastal profile between two two points
    
    The volume of the coastal profile between a landward and seaward boundary 
    is determined by integrating over the surface beneath the coastal profile 
    between those two points. 

    Parameters
    ----------
    elevation : np.array
        np.array containing the elevation of the coastal profile in meters
    seaward_x : float or int
        Cross-shore seaward boundary
    landward_x : float or int
        Cross-shore seaward boundary
        
    Returns
    -------
    float
        volume: surface under the graph in m^2. Can be interpreted as m^3 by 
        assuming the profile is 1 m wide.
    
    See Also
    --------
    Jarkus_Analysis_Toolbox.Extraction
        
    """
    
    from scipy import integrate
    
    # Assign nan if one of the boundaries is nan
    if pd.isnull(seaward_x) == True or pd.isnull(landward_x) == True:
        volume = np.nan
    # Check availability of elevation data
    elif pd.isnull(elevation.first_valid_index()) == True or pd.isnull(elevation.last_valid_index()) == True:
        volume = np.nan    
    elif elevation.first_valid_index() > landward_x or elevation.last_valid_index() < seaward_x:
        volume = np.nan
    else:
        # Remove all elevation outside of boundaries
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