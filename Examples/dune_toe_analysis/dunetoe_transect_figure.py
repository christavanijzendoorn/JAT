

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:19:29 2020

@author: cijzendoornvan
"""
######################################
# THIS SCRIPT CREATES PART OF FIGURE 1
######################################

######################
# PACKAGES
######################
import yaml
import pickle
from JAT.Jarkus_Analysis_Toolbox import Transects, Extraction

from JAT.visualisation import multilineplot
import numpy as np
from scipy.interpolate import griddata
from IPython import get_ipython
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'auto')

from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/dune_toe_analysis/jarkus_03.yml"))

####################
#### LOADING  ######
####################

Jarkus = Dataset(config['inputdir'] + config['data locations']['Jarkus'])
crossshore = Jarkus.variables['cross_shore'][:]
altitude = Jarkus.variables['altitude'][:,:]
ids_Jk = Jarkus.variables['id'][:]

# ##################################
# ####  LOAD DIMENSIONS FILE    ####
# ##################################
# Dir_pickles = settings['Dir_B']

# pickle_file = Dir_pickles + 'Transect_' + trsct + '_dataframe.pickle'
# Dimensions = pickle.load(open(pickle_file, 'rb'))

# variable_x = 'Dunefoot_x_der'
# variable_y = 'Dunefoot_y_der'
# dune_toes_x = pickle.load(open(config['outputdir'] + config['save locations']['DirD'] + variable_x + '_dataframe.pickle','rb'))   
# dune_toes_y = pickle.load(open(config['outputdir'] + config['save locations']['DirD'] + variable_y + '_dataframe.pickle','rb'))  

dunefoots = Dataset(config['inputdir'] + config['data locations']['DuneFoot'])
             
years_requested = list(range(1965, 2017, 5))
years_requested_all = list(range(1965, 2017, 1))

time = dunefoots.variables['time'][:]
years = num2date(time, dunefoots.variables['time'].units)
years = [yr.year for yr in years]                    # convert to purely integers indicating the measurement year
years_filter =  np.isin(years, years_requested)
years_filter_all =  np.isin(years, years_requested_all)
years_req_idxs = np.where(years_filter)[0]
years_req = np.array(years)[years_req_idxs]
years_req_idxs_all = np.where(years_filter_all)[0]
years_req_all = np.array(years)[years_req_idxs_all]

ids = dunefoots.variables['id'][:]
#%%%
for trsct_idx, trsct in enumerate(ids):                
    dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, trsct_idx]
    dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, trsct_idx]
    
    elevation = altitude[:,trsct_idx,:]
    transect = str(trsct)
    # elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
    # Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
    xlimit = [-200,500] # EXAMPLE: [-400,1000]
    ylimit = [-2.1, 25] # EXAMPLE: [-10,22]

   	# Set figure layout
    fig = plt.figure(figsize=(25,12.5))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

    # Load and plot data per year   
    for i, yr in enumerate(years_requested_all):
        
        elev = elevation[i,:]
        mask = elev.mask
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)
    
        colorVal = scalarMap.to_rgba(yr)
        plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
            
    plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')
    
    plt.legend(loc='upper right', fontsize = 22)
    
    ax.set_xlim(xlimit)
    ax.set_ylim(ylimit)
    
    # Show the figure    
    plt.show()
    
    # Save figure as png in predefined directory
    plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
    pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))
    
    plt.close()




#%% NW Ameland
###### SINGLE
# years_requested_all = list(range(1965, 2017, 1))

dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, 176]
dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, 176]

elevation = altitude[:,177,:]
transect = str(trsct)
# elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [250,500] # EXAMPLE: [-400,1000]
ylimit = [-2.1, 14] # EXAMPLE: [-10,22]

   	# Set figure layout
fig = plt.figure(figsize=(25,12.5))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

# Load and plot data per year   
for i, yr in enumerate(years_requested_all):
    
    colorVal = scalarMap.to_rgba(yr)
    elev = elevation[i,:]
    mask = elev.mask
    if yr in years_requested:
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)

    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
    
plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

plt.legend(loc='upper right', fontsize = 22)

ax.set_xlim(xlimit)
ax.set_ylim(ylimit)

# Show the figure    
plt.show()

# Save figure as png in predefined directory
plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))

# plt.close()

#%% GROOTE KEETEN
###### SINGLE
# years_requested_all = list(range(1965, 2017, 1))

dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, 937]
dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, 937]

elevation = altitude[:,954,:]
transect = str(trsct)
# elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-350,-100] # EXAMPLE: [-400,1000]
ylimit = [-2.1, 18] # EXAMPLE: [-10,22]

   	# Set figure layout
fig = plt.figure(figsize=(25,12.5))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

# Load and plot data per year   
for i, yr in enumerate(years_requested_all):
    
    colorVal = scalarMap.to_rgba(yr)
    elev = elevation[i,:]
    mask = elev.mask
    if yr in years_requested:
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)

    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
    
plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

plt.legend(loc='upper right', fontsize = 22)

ax.set_xlim(xlimit)
ax.set_ylim(ylimit)

# Show the figure    
plt.show()

# Save figure as png in predefined directory
plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))

# plt.close()

#%% GROOTE KEETEN
###### SINGLE
# years_requested_all = list(range(1965, 2017, 1))

dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, 938]
dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, 938]

elevation = altitude[:,955,:]
transect = str(trsct)
# elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-350,-100] # EXAMPLE: [-400,1000]
ylimit = [-2.1, 18] # EXAMPLE: [-10,22]

   	# Set figure layout
fig = plt.figure(figsize=(25,12.5))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

# Load and plot data per year   
for i, yr in enumerate(years_requested_all):
    
    colorVal = scalarMap.to_rgba(yr)
    elev = elevation[i,:]
    mask = elev.mask
    if yr in years_requested:
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)

    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
    
plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

plt.legend(loc='upper right', fontsize = 22)

ax.set_xlim(xlimit)
ax.set_ylim(ylimit)

# Show the figure    
plt.show()

# Save figure as png in predefined directory
plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))

# plt.close()

#%% IJMUIDEN
###### SINGLE
# years_requested_all = list(range(1965, 2017, 1))

dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, 1191]
dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, 1191]

elevation = altitude[:,1208,:]
transect = str(trsct)
# elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-150,200] # EXAMPLE: [-400,1000]
ylimit = [-2.1, 23] # EXAMPLE: [-10,22]

   	# Set figure layout
fig = plt.figure(figsize=(25,12.5))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

# Load and plot data per year   
for i, yr in enumerate(years_requested_all):
    
    colorVal = scalarMap.to_rgba(yr)
    elev = elevation[i,:]
    mask = elev.mask
    if yr in years_requested:
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)

    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
    
plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

plt.legend(loc='upper right', fontsize = 22)

ax.set_xlim(xlimit)
ax.set_ylim(ylimit)

# Show the figure    
plt.show()

# Save figure as png in predefined directory
plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))

# plt.close()
#%% Noordwijk north
###### SINGLE
# years_requested_all = list(range(1965, 2017, 1))

dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, 1275]
dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, 1275]

elevation = altitude[:,1292,:]
trsct = 8007975
transect = str(trsct)
# elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-40,120] # EXAMPLE: [-400,1000]
ylimit = [-2.1, 20] # EXAMPLE: [-10,22]

   	# Set figure layout
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

# Load and plot data per year   
for i, yr in enumerate(years_requested_all):
    
    colorVal = scalarMap.to_rgba(yr)
    elev = elevation[i,:]
    mask = elev.mask
    if yr in years_requested:
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)

for i, yr in enumerate(years_requested_all):
    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
    
plt.legend(loc='upper right', fontsize = 22)

ax.set_xlim(xlimit)
ax.set_ylim(ylimit)

xlabel = 'Cross-shore distance [m]'
ylabel = 'Elevation [m to datum]'
ax.set_xlabel(xlabel, fontsize = 24)
ax.set_ylabel(ylabel, fontsize = 24)

# Show the figure    
plt.show()

# Save figure as png in predefined directory
plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))

# plt.close()


#%% Noordwijk south
###### SINGLE
# years_requested_all = list(range(1965, 2017, 1))

dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, 1289]
dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, 1289]

elevation = altitude[:,1306,:]
transect = str(trsct)
# elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [0,150] # EXAMPLE: [-400,1000]
ylimit = [-2.1, 18] # EXAMPLE: [-10,22]

   	# Set figure layout
fig = plt.figure(figsize=(25,12.5))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

# Load and plot data per year   
for i, yr in enumerate(years_requested_all):
    
    colorVal = scalarMap.to_rgba(yr)
    elev = elevation[i,:]
    mask = elev.mask
    if yr in years_requested:
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)

    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
    
plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

plt.legend(loc='upper right', fontsize = 22)

ax.set_xlim(xlimit)
ax.set_ylim(ylimit)

# Show the figure    
plt.show()

# Save figure as png in predefined directory
plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))

# plt.close()

#%% Scheveningen south
###### SINGLE
# years_requested_all = list(range(1965, 2017, 1))

dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, 1376]
dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, 1376]

elevation = altitude[:,1393,:]
trsct = 9010235
transect = str(trsct)
# elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-125,105] # EXAMPLE: [-400,1000]
ylimit = [-2.1, 14.1] # EXAMPLE: [-10,22]

   	# Set figure layout
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)

jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

# Load and plot data per year   
for i, yr in enumerate(years_requested_all):
    
    colorVal = scalarMap.to_rgba(yr)
    elev = elevation[i,:]
    mask = elev.mask
    if yr in years_requested:
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)

for i, yr in enumerate(years_requested_all):
    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        


plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

plt.legend(loc='upper right', fontsize = 22)

ax.set_xlim(xlimit)
ax.set_ylim(ylimit)

xlabel = 'Cross-shore distance [m]'
ylabel = 'Elevation [m to datum]'
ax.set_xlabel(xlabel, fontsize = 26)
ax.set_ylabel(ylabel, fontsize = 26)

# Show the figure    
plt.show()

# Save figure as png in predefined directory
plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))

# plt.close()

#%% Vlugtenburg
###### SINGLE
# years_requested_all = list(range(1965, 2017, 1))

dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, 1477]
dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, 1477]

elevation = altitude[:,1494,:]
transect = str(trsct)
# elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

##################################
####   CREATE AND SAVE PLOTS  ####
##################################
# Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
xlimit = [-100,400] # EXAMPLE: [-400,1000]
ylimit = [-4, 15] # EXAMPLE: [-10,22]

   	# Set figure layout
fig = plt.figure(figsize=(25,12.5))
ax = fig.add_subplot(111)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

jet = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

# Load and plot data per year   
for i, yr in enumerate(years_requested_all):
    
    colorVal = scalarMap.to_rgba(yr)
    elev = elevation[i,:]
    mask = elev.mask
    if yr in years_requested:
        plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)

    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
    
plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

plt.legend(loc='upper right', fontsize = 22)

ax.set_xlim(xlimit)
ax.set_ylim(ylimit)

# Show the figure    
plt.show()

# Save figure as png in predefined directory
plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))

# plt.close()

