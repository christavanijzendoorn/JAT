# -*- coding: utf-8 -*-
# """
# Created on Mon Nov 30 19:08:50 2020

# @author: cijzendoornvan
# """

"""
Provides functions that allow filtering of the extracted characteristic 
parameters.
"""

import math
import pandas as pd
import numpy as np

def locations_filter(dimension, filter_file):
    """Filter out user-defined transects.
    
    Filter out locations that are specified by the user from a dataframe of a 
    characteristic parameter. Default settings filter out locations like the 
    Hondsbossche Dunes and Maasvlakte, redundant transects at the outer edges 
    of the Wadden Islands, and dams in Zeeland.

    Parameters
    ----------
    dimension : pd.dataframe
        dataframe containing the values of a characteristic parameter through 
        time and more multiple transect locations.
    filter_file : dict
        Includes a numbered list of sections that should be excluded. The 
        first transect number represents the start of the section, the second 
        transect number the end.
        
    Returns
    -------
    pd.dataframe
        dimension_filt: dataframe containing the values of a 
        characteristic parameter where filtered transects have been removed.
        
    """    

    # Filter dataframe
    dimension_filt = dimension.copy()
    removed_transects = []
    for i, col in dimension_filt.iteritems():
        for key in filter_file.keys():
            if int(i) >= filter_file[key][0] and int(i) <= filter_file[key][1]:
                removed_transects.append(i)
                dimension_filt.loc[:, i] = np.nan
    
    percentage = len(removed_transects)/len(dimension_filt.columns)*100
    print('Removed percentage of transects is ' + str(percentage))
            
    return dimension_filt

def yrs_filter(dimension, begin_year, end_year):
    """Filter out user-defined years.
    
    Filter values of a characteristic parameter that are associated with a
    range of years that is specified by the user. 

    Parameters
    ----------
    dimension : pd.dataframe
        dataframe containing the values of a characteristic parameter through 
        time and more multiple transect locations.
    begin_yr : int
        Start year of the range that should be filtered
    end_yr : int
        End year of the range that should be filtered
        
    Returns
    -------
    pd.dataframe
        dimension_filt: dataframe containing the values of a 
        characteristic parameter where filtered years have been removed.
        
    """

    dimension_filt = dimension.copy()
    for i, row in dimension_filt.iterrows():
        if i < begin_year or i > end_year:
            dimension_filt.loc[i, :] = np.nan
            
    return dimension_filt

def availability_locations_filter(config, dimension):
    """Filter out transects based on data availability.
    
    Filter out transects that have a data availability that is lower than the 
    user-defined threshold. 

    Parameters
    ----------
    config : dict
        configuration file that includes the user defined availability 
        threshold in percentage (filter2, locations)
    dimension : pd.dataframe
        dataframe containing the values of a characteristic parameter through 
        time and more multiple transect locations.
        
    Returns
    -------
    pd.dataframe
        dimension_filt: dataframe containing the values of a 
        characteristic parameter where filtered transects have been removed.
        
    """ 
    
    availability_per_transect = pd.DataFrame({'transects': dimension.columns})
    availability_per_transect.set_index('transects', inplace=True)
    
    availability_per_transect['availability'] = dimension.count() / len(dimension) * 100 
    # print(availability_per_transect)
    mask = availability_per_transect['availability'] >= config['user defined']['filter2']['locations']

    # Filter dataframe
    dimension_filt = dimension.copy()
    for i, col in dimension_filt.iteritems():
        if mask[i] == False:
            dimension_filt.loc[:, i] = np.nan
            
    return dimension_filt

def availability_years_filter(config, dimension):
    """Filter out years based on data availability.
    
    Filter out years that have a data availability that is lower than the 
    user-defined threshold.  

    Parameters
    ----------
    config : dict
        configuration file that includes the user defined availability 
        threshold in percentage (filter2, years)
    dimension : pd.dataframe
        dataframe containing the values of a characteristic parameter through 
        time and for multiple transect locations.
        
    Returns
    -------
    pd.dataframe
        dimension_filt: dataframe containing the values of a 
        characteristic parameter where filtered years have been removed.
    """ 
    
       #This should be based on analysis of the data availability in which this filter is NOT used.
    # so use the availability_filter_years to see the years in which the availability is lowest.
    
    availability_per_year = pd.DataFrame({'years': dimension.index})
    availability_per_year.set_index('years', inplace=True)
    
    availability_per_year['availability'] = dimension.count(axis=1) / len(dimension.columns) * 100 
    # print(availability_per_year)
    mask = availability_per_year['availability'] >= config['user defined']['filter2']['years']

    # Filter dataframe
    dimension_filt = dimension.copy()
    for i, row in dimension_filt.iterrows():
        if mask[i] == False:
            dimension_filt.loc[:, i] = np.nan
            
    return dimension_filt

def nourishment_filter(config, dimension):
    """Split characteristic parameter values into nourished and not nourished 
    transects.
    
    Parameters
    ----------
    config : dict
        configuration file that includes the directory where the nourishment
        database is stored.
    dimension : pd.dataframe
        dataframe containing the values of a characteristic parameter through 
        time and more multiple transect locations.
        
    Returns
    -------
    pd.dataframe
        nourished_dataframe: dataframe containing the values of a 
        characteristic parameter of only transect that have been nourished.
    pd.dataframe
        not_nourished_dataframe: dataframe containing the values of a 
        characteristic parameter of only transect that have not been nourished.
        
    """ 
    
    Nourishments = pd.read_excel(config['inputdir'] + config['data locations']['Nourishment'])
    filtered = []
    for index, row in Nourishments.iterrows():
        if math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
            continue
        else:
            code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
            code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
            nourished_transects = [i for i in dimension.columns if int(i) >= code_beginraai and int(i) <= code_eindraai]
            filtered.extend(nourished_transects)
    filtered = set(filtered)

    # Filter dataframe
    not_nourished_dataframe = dimension.copy()
    for i, col in not_nourished_dataframe.iteritems():
        if i in filtered:
                not_nourished_dataframe.loc[:, i] = np.nan
    
    nourished_dataframe = dimension.copy()
    for i, col in nourished_dataframe.iteritems():
        if i not in filtered:
                nourished_dataframe.loc[:, i] = np.nan
            
    return nourished_dataframe, not_nourished_dataframe

# def nourishment_filter(variable_dataframe):
#     Nourishments = pd.read_excel("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/Duneforce/JARKUS/Suppletiedatabase.xlsx")
#     filtered = []
#     for index, row in Nourishments.iterrows():
#         if math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
#             continue
#         else:
#             code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
#             code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
#             nourished_transects = [i for i in variable_dataframe.columns if i >= code_beginraai and i <= code_eindraai]
#             filtered.extend(nourished_transects)
#     filtered = set(filtered)
#     not_nourished_transects = [i for i in variable_dataframe.columns if i not in filtered]

#     # Filter dataframe
#     not_nourished_dataframe = variable_dataframe.copy()
#     for i, col in not_nourished_dataframe.iteritems():
#         if i in filtered:
#                 not_nourished_dataframe.loc[:, i] = np.nan
    
#     nourished_dataframe = variable_dataframe.copy()
#     for i, col in nourished_dataframe.iteritems():
#         if i not in filtered:
#                 nourished_dataframe.loc[:, i] = np.nan
                
#     types = ['anders', 'duin', 'duinsuppletie', 'duinverzwaring', 'strand-duinsuppleties', 'strandsuppletie']
#     filtered = []
#     for index, row in Nourishments.iterrows():
#         if row['Type'] not in types or math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
#             continue
#         else:
#             code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
#             code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
#             other_nourishments = [i for i in variable_dataframe.columns if i >= code_beginraai and i <= code_eindraai]
#             filtered.extend(other_nourishments)
#     filtered = set(filtered)
                
#     justSFnourished_dataframe = nourished_dataframe.copy()
#     for i, col in justSFnourished_dataframe.iteritems():
#         if i in filtered:
#                 justSFnourished_dataframe.loc[:, i] = np.nan
            
#     return nourished_dataframe, not_nourished_dataframe, justSFnourished_dataframe#, not_nourished_transects
 