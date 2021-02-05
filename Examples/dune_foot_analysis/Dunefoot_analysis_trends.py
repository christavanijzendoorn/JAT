# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:27:59 2020

@author: cijzendoornvan
"""

import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import math
from pylab import *

get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

plt.rcParams.update({'font.size': 26})
plt.rcParams.update({'lines.linewidth': 2.5})

# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Settings.txt") as file:
    settings = json.load(file)                                                  # include USER-DEFINED settings
    
# load basic settings from .txt file
with open("C:/Users/cijzendoornvan/Documents/GitHub/jarkus/jarkus/Filter2.txt") as ffile:
    filter_file = json.load(ffile)                                                  # include USER-DEFINED settings

DirDFAnalysis = r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\ANALYSIS\DF_analysis\\"    
DirDF = r"C:\Users\cijzendoornvan\Documents\DuneForce\JARKUS\Comparison_methods\Derivative_Diamantidou\DF_2nd_deriv.nc"
DirDimensions = settings['Dir_D1']

# create a dataset object, based on locally saved JARKUS dataset
dataset = xr.open_dataset(DirDF)

# Load and plot second derivative method
DF_elev = dataset['dune_foot_2nd_deriv'].values
DF_cross = dataset['dune_foot_2nd_deriv_cross'].values
years = dataset['time'].values.astype('datetime64[Y]').astype(int) + 1970   
trscts = dataset['id'].values    
area_codes = dataset['areacode'].values

DF_y_Dia = pd.DataFrame(DF_elev, columns=trscts, index = years)
DF_x_Dia = pd.DataFrame(DF_cross, columns=trscts, index=years) 

DF_x_Dia[DF_x_Dia > 10000000] = np.nan
DF_y_Dia[DF_y_Dia > 10000000] = np.nan

# Load and plot pybeach method version
var = 'Dunefoot_x_pybeach_mix_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_x_pybeach_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
   
var = 'Dunefoot_y_pybeach_mix_new'
pickle_file = DirDimensions + var + '_dataframe.pickle'    
DF_y_pybeach_new = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    

#%%

def normalisation(DF, Dir_per_variable, variable): # Get norm values for the cross-shore location for each transect in the norm year
    DF_norm = DF.copy()
    for i, col in DF.iteritems():
        DF_norm.loc[:, i] = col - col.mean()
    DF.to_pickle(Dir_per_variable + var + '_normalized_mean_dataframe' + '.pickle')
    print('The dataframe of ' + var + ' was normalized and saved')
       
    return DF_norm

variable = 'Dunefoot_y_secder'
DF_x_Dia_norm = normalisation(DF_x_Dia, DirDimensions, variable)
  
var = 'Dunefoot_x_pybeach_mix_new'
DF_x_pybeach_norm = normalisation(DF_x_pybeach_new, DirDimensions, variable)    


#%%
import statsmodels.api as sm
def fit_line2(x, y):
    """Return slope, intercept of best fit line."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop') # ignores entires where x or y is NaN
    fit = model.fit()
    rsquared = fit.rsquared
    return fit.params[1], fit.params[0], rsquared # could also return stderr in each via fit.bse

def get_stats(variable_dataframe):
    mean_per_transect = variable_dataframe.mean(axis = 0)
    median_per_transect = variable_dataframe.median(axis = 0)
    mean_per_year = variable_dataframe.mean(axis = 1)
    median_per_year = variable_dataframe.median(axis = 1)
    stddev_per_year = variable_dataframe.std(axis = 1)
    n_per_year = variable_dataframe.count(axis = 1)
    ci95 = 1.96*stddev_per_year/n_per_year.transform('sqrt')
    
    mean_trend, intercept, rsquared = fit_line2(mean_per_year.index, mean_per_year)    
    
    return mean_per_transect, median_per_transect, mean_per_year, median_per_year, ci95

def get_trend(variable_dataframe):
    
    mean_per_transect, median_per_transect, mean_per_year, median_per_year, ci95 = get_stats(variable_dataframe)
    
    mean_trend, intercept, rsquared = fit_line2(mean_per_year.index, mean_per_year)    
    
    return mean_trend, intercept, rsquared


def get_trends_per_transect(variable_dataframe):
    trend_per_transect = pd.DataFrame({'transects': variable_dataframe.columns})
    trend_per_transect.set_index('transects', inplace=True)
    
    for i, column in variable_dataframe.iteritems():
        count_notnan = len(column) - column.isnull().sum(axis = 0)
        if count_notnan > 1: 
            trend, intercept, rsquared = fit_line2(column.index, column)
            
            trend_per_transect.loc[i, 'trend'] = trend
            trend_per_transect.loc[i, 'intercept'] = intercept
            trend_per_transect.loc[i, 'r_squared'] = rsquared
        else:
            trend_per_transect.loc[i, 'trend'] = np.nan
            trend_per_transect.loc[i, 'intercept'] = np.nan
            trend_per_transect.loc[i, 'r_squared'] = np.nan
            
    trend_mean = trend_per_transect['trend'].mean()
    intercept_mean = trend_per_transect['intercept'].mean()
    
    return trend_per_transect, trend_mean, intercept_mean

def get_filtered_trends(variable_dataframe, threshold):
    # Calculate trend per transect
    trend_per_transect, mean_of_trends, mean_of_intercepts = get_trends_per_transect(variable_dataframe)
    trend_per_transect['availability'] = variable_dataframe.count() / len(variable_dataframe) * 100 
    mask = trend_per_transect['availability'] >= threshold

    # Filter dataframe
    variable_dataframe_filt = variable_dataframe.copy()
    for i, col in variable_dataframe_filt.iteritems():
        if mask[i] == False:
            variable_dataframe_filt.loc[:, i] = np.nan
            
    # Calculate trend per transect
    trend_per_transect_filt, mean_of_trends_filt, mean_of_intercepts_filt = get_trends_per_transect(variable_dataframe_filt)
    # Calculate averaged dune foot location and trend
    mean_per_transect_filt, median_per_transect_filt, mean_per_year_filt, median_per_year_filt, ci95_per_year_filt = get_stats(variable_dataframe_filt)
    trend_of_yearly_mean_filt, intercept_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_trend(variable_dataframe_filt)
    
    # Calculate trend array
    mean_of_trends_filt_array = mean_of_trends_filt*mean_per_year_filt.index + mean_of_intercepts_filt
    trend_of_yearly_mean_filt_array = trend_of_yearly_mean_filt*mean_per_year_filt.index + intercept_of_yearly_mean_filt
    
    return mean_per_year_filt, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt, ci95_per_year_filt

def bad_locations_filter(variable_dataframe, filter_dict):
    # Filter dataframe
    filtered_dataframe = variable_dataframe.copy()
    removed_transects = []
    for i, col in filtered_dataframe.iteritems():
        for key in filter_file.keys():
            if i >= int(filter_file[key]['begin']) and i <= int(filter_file[key]['eind']):
                removed_transects.append(i)
                filtered_dataframe.loc[:, i] = np.nan
    
    percentage = len(removed_transects)/len(filtered_dataframe.columns)*100
    print('Removed percentage of transects is ' + str(percentage))
            
    return filtered_dataframe

def bad_yrs_filter(variable_dataframe, begin_year, end_year):
    # Filter dataframe
    filtered_dataframe = variable_dataframe.copy()
    for i, row in filtered_dataframe.iterrows():
        if i < begin_year or i > end_year:
            filtered_dataframe.loc[i, :] = np.nan
            
    return filtered_dataframe

def nourishment_filter(variable_dataframe):
    Nourishments = pd.read_excel("C:/Users/cijzendoornvan/Documents/Duneforce/JARKUS/Suppletiedatabase.xlsx")
    filtered = []
    for index, row in Nourishments.iterrows():
        if math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
            continue
        else:
            code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
            code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
            nourished_transects = [i for i in variable_dataframe.columns if i >= code_beginraai and i <= code_eindraai]
            filtered.extend(nourished_transects)
    filtered = set(filtered)
    not_nourished_transects = [i for i in variable_dataframe.columns if i not in filtered]

    # Filter dataframe
    not_nourished_dataframe = variable_dataframe.copy()
    for i, col in not_nourished_dataframe.iteritems():
        if i in filtered:
                not_nourished_dataframe.loc[:, i] = np.nan
    
    nourished_dataframe = variable_dataframe.copy()
    for i, col in nourished_dataframe.iteritems():
        if i not in filtered:
                nourished_dataframe.loc[:, i] = np.nan
                
    types = ['anders', 'duin', 'duinsuppletie', 'duinverzwaring', 'strand-duinsuppleties', 'strandsuppletie']
    filtered = []
    for index, row in Nourishments.iterrows():
        if row['Type'] not in types or math.isnan(row['BeginRaai']) or math.isnan(row['EindRaai']):# or row['Volume/m'] > 50: # or row['JaarBeginUitvoering'] < 2010: 
            continue
        else:
            code_beginraai = int(row['KustVakNummer'] * 1000000 + row['BeginRaai'] * 100)
            code_eindraai = int(row['KustVakNummer'] * 1000000 + row['EindRaai'] * 100)
            other_nourishments = [i for i in variable_dataframe.columns if i >= code_beginraai and i <= code_eindraai]
            filtered.extend(other_nourishments)
    filtered = set(filtered)
                
    justSFnourished_dataframe = nourished_dataframe.copy()
    for i, col in justSFnourished_dataframe.iteritems():
        if i in filtered:
                justSFnourished_dataframe.loc[:, i] = np.nan
            
    return nourished_dataframe, not_nourished_dataframe, justSFnourished_dataframe#, not_nourished_transects

# Create conversion dictionary
def get_conversion_dicts(ids): 
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
        conversion_alongshore2ids = dict(zip(ids_alongshore, trscts))
        conversion_ids2alongshore = dict(zip(trscts, ids_alongshore))
        
    return conversion_alongshore2ids, conversion_ids2alongshore

def plot_overview(variable_DF, variable, method, DF_type, DirVarPlots):
    
    variable_DF = variable_DF.copy()
    conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(variable_DF.columns) 

    variable_DF.rename(columns = conversion_ids2alongshore, inplace = True)
    variable_DF = variable_DF.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.
    
    variable_DF.rename(columns = conversion_alongshore2ids, inplace = True)
    
    plt.figure(figsize=(50,25))
    average = np.nanmean(variable_DF.values)
    stddev = np.nanstd(variable_DF.values, ddof=1)
    range_value = 2*stddev
    fig = plt.pcolor(variable_DF, vmin = average-range_value, vmax = average + range_value)
    plt.title(variable + ' ' + method + ' ' + DF_type)
    ticks_y = range(0, len(years))[0::5]
    ticks_x = range(0, len(variable_DF.columns))[0::25]
    labels_y = [str(yr) for yr in years][0::5]
    labels_x = [str(tr) for tr in variable_DF.columns][0::25]
    plt.yticks(ticks_y, labels_y)
    plt.xticks(ticks_x, labels_x, rotation='vertical')
    plt.colorbar()
    plt.savefig(DirVarPlots + 'Overview_' + variable.replace(' ', '') + '_' + DF_type.replace(' ','') + '_' + method + '.png')
            
    plt.show()
    # plt.close()

def apply_filters(variable_dataframe, filter_file, begin_yr, end_yr, variable, method, DirVarPlots): 

    DF_filtered = bad_locations_filter(variable_dataframe, filter_file)
    DF_filtered = bad_yrs_filter(DF_filtered, begin_yr, end_yr)

    return DF_filtered

# Set new column ids based on alongshore values

# Plot overviews and trends for Derivative method DF elevation
begin_yr = 1980
end_yr = 2017
DF = DF_y_Dia
variable = 'dune toe elevation'
method = 'secder'
DF_y_Dia_filtered = apply_filters(DF, filter_file, begin_yr, end_yr, variable, method, DirDFAnalysis)

# Plot overviews and trends for pybeach method DF elevation
begin_yr = 1980
end_yr = 2017
DF = DF_y_pybeach_new
variable = 'dune toe elevation'
method = 'pybeach'
DF_y_pybeach_filtered = apply_filters(DF, filter_file, begin_yr, end_yr, variable, method, DirDFAnalysis)

# Plot overviews and trends for Derivative method DF location
begin_yr = 1980
end_yr = 2017
DF = DF_x_Dia_norm
variable = 'dune toe location'
method = 'secder'
DF_x_Dia_filtered = apply_filters(DF, filter_file, begin_yr, end_yr, variable, method, DirDFAnalysis)

# Plot overviews and trends for pybeach method DF location
begin_yr = 1980
end_yr = 2017
DF = DF_x_pybeach_norm
variable = 'dune toe location'
method = 'pybeach'
DF_x_pybeach_filtered = apply_filters(DF, filter_file, begin_yr, end_yr, variable, method, DirDFAnalysis)

#%%

def get_area_DF(DF, area_bounds):
    
    conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF.columns) 
    
    if len(area_bounds) == 1:
        DF_area = DF.loc[:, area_bounds[0] <= DF.columns]
    else:
        DF_area = DF.loc[:, (area_bounds[0] <= DF.columns) & (DF.columns < area_bounds[1])]
    
    DF_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_area = DF_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
    
    ids_block = [conversion_alongshore2ids[col] for col in DF_area.columns]    

    missing_trscts = 0
    DF_noNans = DF_area.isna().sum()
    for j in DF_noNans:
        if j == len(DF_area.index):
            missing_trscts += 1
    n = str(len(ids_block) - missing_trscts)
    print('There are ' + n + 'transects')
        
    return DF_area, n

#%%   

DF_y_Holland, n_y_H = get_area_DF(DF_y_Dia_filtered, [7000000, 10000000])
DF_y_Delta, n_y_D = get_area_DF(DF_y_Dia_filtered, [10000000])

DF_x_Holland, n_x_H = get_area_DF(DF_x_Dia_filtered, [7000000, 10000000])
DF_x_Delta, n_x_D = get_area_DF(DF_x_Dia_filtered, [10000000])

threshold = 0
mean_per_year_y_H, trend_of_yearly_mean_array_y_H, trend_of_yearly_mean_y_H, rsquared_of_yearly_mean_y_H, ci95_y_H = get_filtered_trends(DF_y_Holland, threshold)
mean_per_year_y_D, trend_of_yearly_mean_array_y_D, trend_of_yearly_mean_y_D, rsquared_of_yearly_mean_y_D, ci95_y_D = get_filtered_trends(DF_y_Delta, threshold)
mean_per_year_x_H, trend_of_yearly_mean_array_x_H, trend_of_yearly_mean_x_H, rsquared_of_yearly_mean_x_H, ci95_x_H = get_filtered_trends(DF_x_Holland, threshold)
mean_per_year_x_D, trend_of_yearly_mean_array_x_D, trend_of_yearly_mean_x_D, rsquared_of_yearly_mean_x_D, ci95_x_D = get_filtered_trends(DF_x_Delta, threshold)

#%% 
fig, axs = plt.subplots(2,2, figsize=(25, 27), facecolor='w', edgecolor='k')
# fig.suptitle('Trend in ' + variable)
axs = axs.ravel()

def create_subplot(i, mean_per_year, trend_of_yearly_mean_array, trend_of_yearly_mean, rsquared_of_yearly_mean, ci95, color, ylimit, label1, variable, unit, region, begin_yr, end_yr):
    p1_ex = axs[i].plot(mean_per_year[0:2], color = 'grey', label = label2)#, marker = 'o', markersize = 8)
    axs[i].plot(mean_per_year, color = color, label = label2)#, marker = 'o', markersize = 8)
    p2_ex = axs[i].errorbar(mean_per_year.index[0:2], mean_per_year[0:2], ci95[0:2], color = 'grey', elinewidth = 3, capsize = 8, capthick = 3)
    axs[i].errorbar(mean_per_year.index, mean_per_year, ci95, color = color, elinewidth = 3, capsize = 8, capthick = 3, label = '95% confidence interval')
    p3, = axs[i].plot(mean_per_year.index, trend_of_yearly_mean_array, color = color, linestyle = 'dashed', label = label1)
    
    axs[i].set_ylim(ylimit)
    axs[i].set_xlim([begin_yr, end_yr])
    
    if i == 2 or i ==3:
        axs[i].set_xlabel('Time (yr)')
    # fig.text(0.005, 0.5, variable + '(' + unit + ')', va='center', rotation='vertical')
    axs[i].set_ylabel(variable + '(' + unit + ')')
    if i == 0 or i == 1:
        axs[i].set_title(region)
    
    axs[i].tick_params(axis='both', pad=25)
    
    mpl.rcParams['legend.handlelength'] = 1.5
    mpl.rcParams['legend.markerscale'] = 1
    
    
    handles,labels=axs[i].get_legend_handles_labels()
    # handles = [handles[0], handles[1]]
    leg3 = axs[i].legend(p3, labels=[label1], loc = 'upper left') # [p3], labels = [label1], loc = 'upper left'

    if i == 3:                
        leg1 = axs[i].legend([p1_ex, p2_ex], [label2, '95% confidence interval'], loc = 'lower right')
        leg2 = axs[i].legend([handles[0]], [label2], loc = 'lower left')
        axs[i].add_artist(leg1)
        axs[i].add_artist(leg2)
        axs[i].add_artist(leg3)
    return

begin_yr = 1980
end_yr = 2017
unit = 'm'
label2 = 'Spatial average'

## LEFT SIDE
            
region = 'Holland Coast'   
color = '#136306'

variable = 'Dune toe elevation'
ylimit = [2.5,4.25]       
label1 = 'Trend of ' + str(round(trend_of_yearly_mean_y_H*1000,2)) + ' mm/yr \n($r^{2}$ = ' + str(round(rsquared_of_yearly_mean_y_H,3)) + ', n = ' + n_y_H + ')'

i = 0
axs[0] = create_subplot(i, mean_per_year_y_H, trend_of_yearly_mean_array_y_H, trend_of_yearly_mean_y_H, rsquared_of_yearly_mean_y_H, ci95_y_H, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)


variable = 'Dune toe cross shore location'
ylimit = [-25,40] 
label1 = 'Trend of ' + str(round(trend_of_yearly_mean_x_H,2)) + ' m/yr \n($r^{2}$ = ' + str(round(rsquared_of_yearly_mean_x_H,3)) + ', n = ' + n_x_H + ')'

i = 2
axs[i] = create_subplot(i, mean_per_year_x_H, trend_of_yearly_mean_array_x_H, trend_of_yearly_mean_x_H, rsquared_of_yearly_mean_x_H, ci95_x_H, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

## RIGHT SIDE

region = 'Delta Coast'
color = '#4169E1'

variable = 'Dune toe elevation'
ylimit = [2.5,4.25]       
label1 = 'Trend of ' + str(round(trend_of_yearly_mean_y_D*1000,2)) + ' mm/yr \n($r^{2}$ = ' + str(round(rsquared_of_yearly_mean_y_D,3)) + ', n = ' + n_y_D + ')'

i = 1
axs[i] = create_subplot(i, mean_per_year_y_D, trend_of_yearly_mean_array_y_D, trend_of_yearly_mean_y_D, rsquared_of_yearly_mean_y_D, ci95_y_D, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

variable = 'Dune toe cross shore location'
ylimit = [-25,40] 
label1 = 'Trend of ' + str(round(trend_of_yearly_mean_x_D,2)) + ' m/yr \n($r^{2}$ = ' + str(round(rsquared_of_yearly_mean_x_D,3)) + ', n = ' + n_x_D + ')' 

i = 3
axs[i] = create_subplot(i, mean_per_year_x_D, trend_of_yearly_mean_array_x_D, trend_of_yearly_mean_x_D, rsquared_of_yearly_mean_x_D, ci95_x_D, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

fig.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.subplots_adjust(wspace = 0.25, hspace = 0.25)


filename = 'Trends_H_D_x_y'+ '.pdf'
plt.savefig(DirDFAnalysis + filename, format='pdf')
print('saved figure')
