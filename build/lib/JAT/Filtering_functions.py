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
Created on Mon Nov 30 19:08:50 2020

@author: cijzendoornvan
"""

import math
import pandas as pd
import numpy as np

def bad_locations_filter(dimension, location_filter):
    # Filter dataframe
    dimension_filt = dimension.copy()
    removed_transects = []
    for i, col in dimension_filt.iteritems():
        for key in location_filter.keys():
            if int(i) >= int(location_filter[key][0]) and int(i) <= int(location_filter[key][-1]):
                removed_transects.append(i)
                dimension_filt.loc[:, i] = np.nan
    
    percentage = len(removed_transects)/len(dimension_filt.columns)*100
    print('Removed percentage of transects is ' + str(percentage))
            
    return dimension_filt

def bad_yrs_filter(dimension, begin_year, end_year):
    # Filter dataframebased on a user-defined range. This should be based on analysis of the data availability in which this filter is NOT used.
    # so use the availability_filter_years to see the years in which the availability is lowest.
    dimension_filt = dimension.copy()
    for i, row in dimension_filt.iterrows():
        if i < begin_year or i > end_year:
            dimension_filt.loc[i, :] = np.nan
            
    return dimension_filt

def availability_filter_locations(config, dimension):
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
    Nourishments = pd.read_excel(config['root'] + config['data locations']['DirNourish'])
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