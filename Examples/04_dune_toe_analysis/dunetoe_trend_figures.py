# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:27:59 2020

@author: cijzendoornvan
"""

###########################################################
# THIS SCRIPT CREATES FIGURE 2 AND SUPPLEMENTARY FIGURE 1
###########################################################

######################
# PACKAGES
######################
import yaml
import pickle
from JAT.Jarkus_Analysis_Toolbox import Transects
import JAT.Filtering_functions as Ff

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

######################
# LOAD SETTINGS + DATA
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/04_dune_toe_analysis/jarkus_04.yml"))
filter_file = yaml.safe_load(open(config['inputdir'] + config['data locations']['LocFilter']))

DirDimensions = config['outputdir'] + config['save locations']['DirD']
DirFigures = config['outputdir'] + config['save locations']['DirFig']
if os.path.isdir(DirFigures) == False:
            os.mkdir(DirFigures)

# Load 2nd derivative method version
var = 'Dunetoe_y_der' # Dune toe elevation
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_y_der = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension     

var = 'Dunetoe_x_der' # Dune toe cross-shore location
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_x_der = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    

# Load pybeach method version
var = 'Dunetoe_x_pybeach_mix' # Dune toe elevation
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_x_pybeach = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    
   
var = 'Dunetoe_y_pybeach_mix' # Dune toe cross-shore location
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_y_pybeach = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    

DF_x_der[DF_x_der > 10000000] = np.nan
DF_y_der[DF_y_der > 10000000] = np.nan

years = DF_y_der.index
trscts = DF_y_der.columns

# #%% Get normalized cross-shore locations
var = 'Dunetoe_x_der_normalized' # Dune toe cross-shore location
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_x_der_norm = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    

# Load pybeach method version
var = 'Dunetoe_x_pybeach_mix_normalized' # Dune toe elevation
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_x_pybeach_norm = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension    

#%%# Filtering based on transects and years

def apply_filters(variable_dataframe, filter_file, begin_yr, end_yr, variable): 
    DF_filtered = Ff.bad_locations_filter(variable_dataframe, filter_file)
    DF_filtered = Ff.bad_yrs_filter(DF_filtered, begin_yr, end_yr)

    return DF_filtered

# Plot overviews and trends for Derivative method DF elevation
begin_yr = 1980
end_yr = 2017
variable = 'dune toe elevation'
method = 'secder'
DF_y_der_filtered = apply_filters(DF_y_der, filter_file, begin_yr, end_yr, variable)

# Plot overviews and trends for pybeach method DF elevation
begin_yr = 1980
end_yr = 2017
variable = 'dune toe elevation'
method = 'pybeach'
DF_y_pybeach_filtered = apply_filters(DF_y_pybeach, filter_file, begin_yr, end_yr, variable)

# Plot overviews and trends for Derivative method DF location
begin_yr = 1980
end_yr = 2017
variable = 'dune toe location'
method = 'secder'
DF_x_der_filtered = apply_filters(DF_x_der_norm, filter_file, begin_yr, end_yr, variable)

# Plot overviews and trends for pybeach method DF location
begin_yr = 1980
end_yr = 2017
variable = 'dune toe location'
method = 'pybeach'
DF_x_pybeach_filtered = apply_filters(DF_x_pybeach_norm, filter_file, begin_yr, end_yr, variable)

#%% Split up into Holland and Delta coast, leave out Wadden coast
def get_area(DF, area_bounds):
    # Get conversion dictionary needed to convert from transect number to alongshore value
    data = Transects(config)
    conversion_alongshore2ids, conversion_ids2alongshore = data.get_conversion_dicts()
    
    if len(area_bounds) == 1:
        DF_area = DF.loc[:, area_bounds[0] <= DF.columns.astype(int)]
    else:
        DF_area = DF.loc[:, (area_bounds[0] <= DF.columns.astype(int)) & (DF.columns.astype(int) < area_bounds[1])]
    
    DF_area.rename(columns = conversion_ids2alongshore, inplace = True)
    DF_area = DF_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    

    missing_trscts = 0
    DF_countNans = DF_area.isna().sum()
    for j in DF_countNans:
        if j == len(DF_area.index):
            missing_trscts += 1
    n = str(len(DF_area.columns) - missing_trscts)
    print('There are ' + n + ' transects')
        
    return DF_area, n

# Execute for second derivative method
DF_y_der_Holland, n_y_der_H = get_area(DF_y_der_filtered, [7000000, 10000000])
DF_y_der_Delta, n_y_der_D = get_area(DF_y_der_filtered, [10000000])

DF_x_der_Holland, n_x_der_H = get_area(DF_x_der_filtered, [7000000, 10000000])
DF_x_der_Delta, n_x_der_D = get_area(DF_x_der_filtered, [10000000])

# As a check, the data availability per year in calculated
data_percent_y_der_H = DF_y_der_Holland.count(axis=1).div(int(n_y_der_H)*0.01)
data_percent_y_der_D = DF_y_der_Delta.count(axis=1).div(int(n_y_der_D)*0.01)

# Execute for pybeach method
DF_y_pyb_Holland, n_y_pyb_H = get_area(DF_y_pybeach_filtered, [7000000, 10000000])
DF_y_pyb_Delta, n_y_pyb_D = get_area(DF_y_pybeach_filtered, [10000000])

DF_x_pyb_Holland, n_x_pyb_H = get_area(DF_x_pybeach_filtered, [7000000, 10000000])
DF_x_pyb_Delta, n_x_pyb_D = get_area(DF_x_pybeach_filtered, [10000000])

# As a check, the data availability per year in calculated
data_percent_y_pyb_H = DF_y_pyb_Holland.count(axis=1).div(int(n_y_pyb_H)*0.01)
data_percent_y_pyb_D = DF_y_pyb_Delta.count(axis=1).div(int(n_y_pyb_D)*0.01)
#%% Get statistics from the filtered dataframes
import statsmodels.api as sm
def fit_line2(x, y):
    """Return slope, intercept of best fit line."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop') # ignores entires where x or y is NaN
    fit = model.fit()
    rsquared = fit.rsquared
    return fit.params[1], fit.params[0], rsquared # could also return stderr in each via fit.bse

def get_stats(variable_dataframe):
    # Calculate averaged dune foot location and related stats
    # mean_per_transect = variable_dataframe.mean(axis = 0)
    mean_per_year = variable_dataframe.mean(axis = 1)
    stddev_per_year = variable_dataframe.std(axis = 1)
    n_per_year = variable_dataframe.count(axis = 1)
    ci95 = 1.96*stddev_per_year/n_per_year.transform('sqrt')
    
    # Calculate trend in dune foot location
    trend, intercept, rsquared = fit_line2(mean_per_year.index, mean_per_year)   
    # Calculate trend array
    trend_array = trend*mean_per_year.index + intercept
    
    return mean_per_year, trend_array, trend, rsquared, ci95

# Execute for second derivative method
mean_per_year_y_der_H, trend_array_y_der_H, trend_y_der_H, rsquared_y_der_H, ci95_y_der_H = get_stats(DF_y_der_Holland)
mean_per_year_y_der_D, trend_array_y_der_D, trend_y_der_D, rsquared_y_der_D, ci95_y_der_D = get_stats(DF_y_der_Delta)
mean_per_year_x_der_H, trend_array_x_der_H, trend_x_der_H, rsquared_x_der_H, ci95_x_der_H = get_stats(DF_x_der_Holland)
mean_per_year_x_der_D, trend_array_x_der_D, trend_x_der_D, rsquared_x_der_D, ci95_x_der_D = get_stats(DF_x_der_Delta)

# Execute for pybeach method
mean_per_year_y_pyb_H, trend_array_y_pyb_H, trend_y_pyb_H, rsquared_y_pyb_H, ci95_y_pyb_H = get_stats(DF_y_pyb_Holland)
mean_per_year_y_pyb_D, trend_array_y_pyb_D, trend_y_pyb_D, rsquared_y_pyb_D, ci95_y_pyb_D = get_stats(DF_y_pyb_Delta)
mean_per_year_x_pyb_H, trend_array_x_pyb_H, trend_x_pyb_H, rsquared_x_pyb_H, ci95_x_pyb_H = get_stats(DF_x_pyb_Holland)
mean_per_year_x_pyb_D, trend_array_x_pyb_D, trend_x_pyb_D, rsquared_x_pyb_D, ci95_x_pyb_D = get_stats(DF_x_pyb_Delta)

#%% Plotting function
def create_subplot(i, mean_per_year, trend_array, trend, rsquared, ci95, color, ylimit, label1, variable, unit, region, begin_yr, end_yr):
    
    #SLR
    if variable == 'Dune toe elevation':
        start = trend_array[15]
        SLR_rate = 0.0019
        sea_level_array = start + np.arange(0,38)*SLR_rate
        labelSLR = 'SLR rate'
        p4, = axs[i].plot(mean_per_year.index[15:], sea_level_array, color = 'grey', linestyle = 'dashed', label = labelSLR, linewidth = 3)
    
    #spatial average
    p1_ex, = axs[i].plot(mean_per_year.index[0:2], mean_per_year.values[0:2], color = color, linewidth = 3)#, marker = 'o', markersize = 8)
    axs[i].plot(mean_per_year.index, mean_per_year.values, color = color, label = label2, linewidth = 3)#, marker = 'o', markersize = 8)
    # conf int
    p2_ex = axs[i].errorbar(mean_per_year.index[0:2], mean_per_year[0:2], ci95[0:2], color = color, elinewidth = 2, capsize = 8, capthick = 3)
    axs[i].errorbar(mean_per_year.index, mean_per_year, ci95, color = color, elinewidth = 2, capsize = 6, capthick = 2, label = '95% confidence interval')
    # trend
    p3, = axs[i].plot(mean_per_year.index, trend_array, color = color, label = label1,  linestyle = 'dashed', linewidth = 4)

    axs[i].set_ylim(ylimit)
    axs[i].set_xlim([begin_yr, end_yr])
    
    if i == 2 or i ==3:
        axs[i].set_xlabel('Time (yr)', fontsize = 36)
    # fig.text(0.005, 0.5, variable + '(' + unit + ')', va='center', rotation='vertical')
    axs[i].set_ylabel(variable + '(' + unit + ')', fontsize = 36)
    if i == 0 or i == 1:
        axs[i].set_title(region, weight = 'bold')
    
    axs[i].tick_params(axis='both', pad=25)
    
    mpl.rcParams['legend.handlelength'] = 1.5
    mpl.rcParams['legend.markerscale'] = 1
    mpl.rcParams['xtick.major.size'] = 15
    mpl.rcParams['xtick.minor.size'] = 10
    mpl.rcParams['ytick.major.size'] = 15
    mpl.rcParams['ytick.minor.size'] = 10
    
    handles,labels=axs[i].get_legend_handles_labels()
    # handles = [handles[0], handles[1]]
    leg3 = axs[i].legend([p3], [label1], loc = 'upper center') # [p3], labels = [label1], loc = 'upper left'
    letters = ['a.', 'b.', 'c.', 'd.']
    axs[i].annotate(letters[i], xy=(0.05, 0.92), xycoords="axes fraction", weight='bold', fontsize = 36)

    if i == 0 or i == 1:
        leg4 = axs[i].legend([p4], [labelSLR], loc = 'lower right')
        axs[i].add_artist(leg3)
    if i == 2 or i == 3:                
        leg1 = axs[i].legend([p1_ex, p2_ex], [label2, '95% confidence interval'], loc = 'lower right')
        # leg2 = axs[i].legend([p1_ex], [label2], loc = 'lower left')
        axs[i].add_artist(leg1)
        # axs[i].add_artist(leg2)
        axs[i].add_artist(leg3)
        
    return

plt.rcParams.update({'font.size': 34})
plt.rcParams.update({'lines.linewidth': 2.5})
#%% Plot trends for the dune toe position based on the second derivative method

fig, axs = plt.subplots(2,2, figsize=(25, 27), facecolor='w', edgecolor='k')
axs = axs.ravel()

begin_yr = 1980
end_yr = 2017
unit = 'm'
label2 = 'Spatial average'

## LEFT SIDE
        
region = 'Holland Coast'   
color = '#136306'
variable = 'Dune toe elevation'  
ylimit = [2.6,4.3]   
label1 = 'Trend of ' + str(round(trend_y_der_H*1000,1)) + ' mm/yr \n($r^{2}$ = ' + str(round(rsquared_y_der_H,2)) + ', n = ' + n_y_der_H + ')'

i = 0
axs[0] = create_subplot(i, mean_per_year_y_der_H, trend_array_y_der_H, trend_y_der_H, rsquared_y_der_H, ci95_y_der_H, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

variable = 'Dune toe cross shore location'
ylimit = [-30,40] 
label1 = 'Trend of ' + str(round(trend_x_der_H,2)) + ' m/yr \n($r^{2}$ = ' + str(round(rsquared_x_der_H,2)) + ', n = ' + n_x_der_H + ')'

i = 2
axs[i] = create_subplot(i, mean_per_year_x_der_H, trend_array_x_der_H, trend_x_der_H, rsquared_x_der_H, ci95_x_der_H, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

## RIGHT SIDE
region = 'Delta Coast'
color = '#4169E1'
variable = 'Dune toe elevation'
ylimit = [2.6,4.3]   
label1 = 'Trend of ' + str(round(trend_y_der_D*1000,1)) + ' mm/yr \n($r^{2}$ = ' + str(round(rsquared_y_der_D,2)) + ', n = ' + n_y_der_D + ')'

i = 1
axs[i] = create_subplot(i, mean_per_year_y_der_D, trend_array_y_der_D, trend_y_der_D, rsquared_y_der_D, ci95_y_der_D, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

variable = 'Dune toe cross shore location'
ylimit = [-30,40] 
label1 = 'Trend of ' + str(round(trend_x_der_D,2)) + ' m/yr \n($r^{2}$ = ' + str(round(rsquared_x_der_D,2)) + ', n = ' + n_x_der_D + ')' 

i = 3
axs[i] = create_subplot(i, mean_per_year_x_der_D, trend_array_x_der_D, trend_x_der_D, rsquared_x_der_D, ci95_x_der_D, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

fig.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.subplots_adjust(wspace = 0.25, hspace = 0.25)


filename = 'Trends_H_D_x_y_der'
plt.savefig(DirFigures + filename + '.png')
# plt.savefig(DirFigures + filename + '.eps', format='eps')
plt.savefig(DirFigures + filename + '.pdf', format='pdf')
print('saved figure second derivative method')

#%% Plot trends for the dune toe position based on the second derivative method

fig, axs = plt.subplots(2,2, figsize=(25, 27), facecolor='w', edgecolor='k')
axs = axs.ravel()

begin_yr = 1980
end_yr = 2017
unit = 'm'
label2 = 'Spatial average'

## LEFT SIDE
        
region = 'Holland Coast'   
color = '#136306'
variable = 'Dune toe elevation'  
ylimit = [2.4,4.0]    
label1 = 'Trend of ' + str(round(trend_y_pyb_H*1000,1)) + ' mm/yr \n($r^{2}$ = ' + str(round(rsquared_y_pyb_H,2)) + ', n = ' + n_y_pyb_H + ')'

i = 0
axs[0] = create_subplot(i, mean_per_year_y_pyb_H, trend_array_y_pyb_H, trend_y_pyb_H, rsquared_y_pyb_H, ci95_y_pyb_H, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

variable = 'Dune toe cross shore location'
ylimit = [-35,45] 
label1 = 'Trend of ' + str(round(trend_x_pyb_H,2)) + ' m/yr \n($r^{2}$ = ' + str(round(rsquared_x_pyb_H,2)) + ', n = ' + n_x_pyb_H + ')'

i = 2
axs[i] = create_subplot(i, mean_per_year_x_pyb_H, trend_array_x_pyb_H, trend_x_pyb_H, rsquared_x_pyb_H, ci95_x_pyb_H, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

## RIGHT SIDE
region = 'Delta Coast'
color = '#4169E1'
variable = 'Dune toe elevation'
ylimit = [2.4,4.0]    
label1 = 'Trend of ' + str(round(trend_y_pyb_D*1000,1)) + ' mm/yr \n($r^{2}$ = ' + str(round(rsquared_y_pyb_D,2)) + ', n = ' + n_y_pyb_D + ')'

i = 1
axs[i] = create_subplot(i, mean_per_year_y_pyb_D, trend_array_y_pyb_D, trend_y_pyb_D, rsquared_y_pyb_D, ci95_y_pyb_D, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

variable = 'Dune toe cross shore location'
ylimit = [-35,45] 
label1 = 'Trend of ' + str(round(trend_x_pyb_D,2)) + ' m/yr \n($r^{2}$ = ' + str(round(rsquared_x_pyb_D,2)) + ', n = ' + n_x_pyb_D + ')' 

i = 3
axs[i] = create_subplot(i, mean_per_year_x_pyb_D, trend_array_x_pyb_D, trend_x_pyb_D, rsquared_x_pyb_D, ci95_x_pyb_D, color, ylimit, label1, variable, unit, region, begin_yr, end_yr)

fig.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.subplots_adjust(wspace = 0.25, hspace = 0.25)


filename = 'Trends_H_D_x_y_pyb'
plt.savefig(DirFigures + filename + '.png')
# plt.savefig(DirFigures + filename + '.eps', format='eps')
plt.savefig(DirFigures + filename + '.pdf', format='pdf')
print('saved figure pybeach method')

