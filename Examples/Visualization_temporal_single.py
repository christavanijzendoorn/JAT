# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:11:20 2020

@author: cijzendoornvan
"""


#%%
import statsmodels.api as sm
def fit_line2(x, y):
    """Return slope, intercept of best fit line."""
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop') # ignores entires where x or y is NaN
    fit = model.fit()
    rsquared = fit.rsquared
    return fit.params[1], fit.params[0], rsquared # could also return stderr in each via fit.bse

def get_overall_trend(variable_dataframe):
    mean_per_transect = variable_dataframe.mean(axis = 0)
    median_per_transect = variable_dataframe.median(axis = 0)
    mean_per_year = variable_dataframe.mean(axis = 1)
    median_per_year = variable_dataframe.median(axis = 1)
    
    mean_trend, intercept, rsquared = fit_line2(mean_per_year.index, mean_per_year)    
    
    return mean_per_transect, median_per_transect, mean_per_year, median_per_year, mean_trend, intercept, rsquared

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
    mean_per_transect_filt, median_per_transect_filt, mean_per_year_filt, median_per_year_filt, trend_of_yearly_mean_filt, intercept_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_overall_trend(variable_dataframe_filt)
    
    # Calculate trend array
    mean_of_trends_filt_array = mean_of_trends_filt*mean_per_year_filt.index + mean_of_intercepts_filt
    trend_of_yearly_mean_filt_array = trend_of_yearly_mean_filt*mean_per_year_filt.index + intercept_of_yearly_mean_filt
    
    return mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt





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

def plot_overview_types(variable_dataframe, filter_file, begin_yr, end_yr, variable, method, DirVarPlots): 

    DF_filtered = bad_locations_filter(variable_dataframe, filter_file)
    DF_filtered = bad_yrs_filter(DF_filtered, begin_yr, end_yr)
    DF_nourished, DF_not_nourished = nourishment_filter(DF_filtered)
    
    # Set new column ids based on alongshore values
    DF_type = 'filtered'
    plot_overview(DF_filtered, variable, method, DF_type, DirVarPlots)
    DF_type = 'nourished'
    plot_overview(DF_nourished, variable, method, DF_type, DirVarPlots)
    DF_type = 'not_nourished'
    plot_overview(DF_not_nourished, variable, method, DF_type, DirVarPlots)

    return DF_filtered, DF_nourished, DF_not_nourished #, not_nourished_transects     

def plot_trend(DF, variable, method, unit, DF_type, begin_yr, end_yr, ylimit):
    
    threshold = 0
    area_bounds = [2000000, 7000000, 10000000]        
    
    conversion_alongshore2ids, conversion_ids2alongshore = get_conversion_dicts(DF.columns) 

    for i, a in enumerate(area_bounds):
        if i == len(area_bounds) - 1:
            DF_area = DF.loc[:, area_bounds[i] <= DF.columns]
        else:
            DF_area = DF.loc[:, (area_bounds[i] <= DF.columns) & (DF.columns < area_bounds[i+1])]
        
        DF_area.rename(columns = conversion_ids2alongshore, inplace = True)
        DF_area = DF_area.sort_index(axis=1) # Flip transects based on alongshore order instead of ids order.    
        
        ids_block = [conversion_alongshore2ids[col] for col in DF_area.columns]    
        
        missing_trscts = 0
        DF_noNans = DF_area.isna().sum()
        for i in DF_noNans:
            if i == len(DF_area.index):
                missing_trscts += 1
        print('There are ' + str(len(ids_block) - missing_trscts) + 'transects in ' + variable + ' (' + method + ', ' + DF_type + ') ' + 'between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
        
        if DF_area.empty == False and DF_area.dropna(how='all').empty == False:
    
            mean_per_year_filt, median_per_transect_filt, mean_of_trends_filt_array, trend_of_yearly_mean_filt_array, trend_of_yearly_mean_filt, rsquared_of_yearly_mean_filt = get_filtered_trends(DF_area, threshold)
            
            plt.figure(figsize=(15,10))
            
            plt.plot(mean_per_year_filt)
            plt.plot(mean_per_year_filt.index, trend_of_yearly_mean_filt_array)
            plt.plot(mean_per_year_filt.index, mean_of_trends_filt_array)
            plt.ylim(ylimit)
            plt.xlim([begin_yr, end_yr])
            
            plotloc2 = (ylimit[1] - ylimit[0])*0.2 + ylimit[0]
            plotloc1 = (ylimit[1] - ylimit[0])*0.25 + ylimit[0]
            
            if variable == 'dune foot elevation':
                plt.text(begin_yr, plotloc1, str(round(trend_of_yearly_mean_filt*1000,2)) + ' mm/yr')
            elif variable == 'dune foot location':
                plt.text(begin_yr, plotloc1, str(round(trend_of_yearly_mean_filt,2)) + ' m/yr')
            
            plt.text(begin_yr, plotloc2, str(round(rsquared_of_yearly_mean_filt,3)) + ' r^2')
                
            plt.title('Trend in ' + variable + ' (' + method + ', ' + DF_type + ') ' + 'between transect ' + str(min(ids_block)) + ' and ' + str(max(ids_block)))
            
            plt.xlabel('Time (yr)')
            plt.ylabel(variable + '(' + unit + ')')
        
            variable_str = variable.replace(' ','')
            DF_type_str = DF_type.replace(' ','')
            filename = 'Trend_' + variable_str + '_' + DF_type_str + '_' + method + '_' + str(min(ids_block)) + '.png'
            plt.savefig(DirDFAnalysis + filename)
            print('saved figure')
            
            plt.close()
            


# Set new column ids based on alongshore values



# DF = DF_x_Dia_norm
# figure_title = 'Alongshore and temporal variation of the cross shore dune toe location (m)'
# colorbar_label = 'Crossshore dune toe location (m)'
# colormap_var = "Greens"
# file_name = 'dune_foot_location'

