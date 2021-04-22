# -*- coding: utf-8 -*-
# """
# Created on Mon Nov 30 19:08:50 2020

# @author: cijzendoornvan
# """

"""
Blabla
"""

import math
import pandas as pd
import numpy as np

def bad_locations_filter(dimension, filter_file):
    """
    Blabla
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

def bad_yrs_filter(dimension, begin_year, end_year):
    """
    Blabla
    """
    
    # Filter dataframebased on a user-defined range. This should be based on analysis of the data availability in which this filter is NOT used.
    # so use the availability_filter_years to see the years in which the availability is lowest.
    dimension_filt = dimension.copy()
    for i, row in dimension_filt.iterrows():
        if i < begin_year or i > end_year:
            dimension_filt.loc[i, :] = np.nan
            
    return dimension_filt

def availability_filter_locations(config, dimension):
    """
    Blabla
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

def availability_filter_years(config, dimension):
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

def nourishment_filter(config, variable_dataframe):
    Nourishments = pd.read_excel(config['inputdir'] + config['data locations']['Nourishment'])
    filtered = []
    for index, row in Nourishments.iterrows():
        if math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
            continue
        else:
            code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
            code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
            nourished_transects = [i for i in variable_dataframe.columns if int(i) >= code_beginraai and int(i) <= code_eindraai]
            filtered.extend(nourished_transects)
    filtered = set(filtered)

    # Filter dataframe
    not_nourished_dataframe = variable_dataframe.copy()
    for i, col in not_nourished_dataframe.iteritems():
        if i in filtered:
                not_nourished_dataframe.loc[:, i] = np.nan
    
    nourished_dataframe = variable_dataframe.copy()
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
 