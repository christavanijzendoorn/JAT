# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:19:29 2020

@author: cijzendoornvan
"""

######################
# PACKAGES
######################
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt

import matplotlib.colors as colors
import matplotlib.cm as cm

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/04_dune_toe_analysis/jarkus_04.yml"))

####################
#### LOADING  ######
####################
trsct = 9010235
transect = str(trsct)

DirFigures = config['outputdir'] + config['save locations']['DirFig']

elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + str(transect) + '_elevation' + '.pickle','rb'))    
crossshore = elevation.columns

# Load 2nd derivative method version
var = 'Dunetoe_y_der' # Dune toe elevation
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_y_der = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension     

var = 'Dunetoe_x_der' # Dune toe cross-shore location
pickle_file = config['outputdir'] + config['save locations']['DirD'] + var + '_dataframe.pickle'    
DF_x_der = pickle.load(open(pickle_file, 'rb')) #load pickle of dimension  
             
years_requested = list(range(1965, 2017, 5))
years_requested_all = list(range(1965, 2017, 1))

#%% Scheveningen south
dunetoes_y = DF_y_der.iloc[:, 1376]
dunetoes_x = DF_x_der.iloc[:, 1376]

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
for yr in years_requested:
    colorVal = scalarMap.to_rgba(yr)
    
    elevation = elevation.replace(-9999, np.nan)
    elev_ind = np.where(elevation.loc[yr].notnull())[0]
    elev = elevation.loc[yr].iloc[elev_ind]
    crossshore = elev.index
    
    plt.plot(crossshore, elev, color=colorVal, label = str(yr), linewidth = 2.5)

for yr in years_requested_all:
    colorVal = scalarMap.to_rgba(yr)
    plt.plot(dunetoes_x.loc[yr], dunetoes_y.loc[yr], 'o', markersize=13, color = colorVal, mew=3)        

# Plot marker for in legend
plt.plot(dunetoes_x.iloc[0], dunetoes_y.iloc[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')

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
plt.savefig(config['outputdir'] + config['save locations']['DirFig'] + 'Transect_dunetoes_' + transect + '.png')

#%%% Plot all transects with dune toes
# Note, code below does not work currently, but can be used as example

# for trsct_idx, trsct in enumerate(ids):                
#     dunefoots_y = dunefoots.variables['dune_foot_2nd_deriv'][:, trsct_idx]
#     dunefoots_x = dunefoots.variables['dune_foot_2nd_deriv_cross'][:, trsct_idx]
    
#     elevation = altitude[:,trsct_idx,:]
#     transect = str(trsct)
#     # elevation = pickle.load(open(config['outputdir'] + config['save locations']['DirA'] + transect + '_elevation' + '.pickle','rb'))    

# ##################################
# ####   CREATE AND SAVE PLOTS  ####
# ##################################
#     # Set x and y limit of plots - Leave lists empty (i.e. []) for automatic axis limits
#     xlimit = [-200,500] # EXAMPLE: [-400,1000]
#     ylimit = [-2.1, 25] # EXAMPLE: [-10,22]

#    	# Set figure layout
#     fig = plt.figure(figsize=(25,12.5))
#     ax = fig.add_subplot(111)
#     ax.tick_params(axis='x', labelsize=20)
#     ax.tick_params(axis='y', labelsize=20)
    
#     jet = plt.get_cmap('jet') 
#     cNorm  = colors.Normalize(vmin=min(years_requested), vmax=max(years_requested))
#     scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)   

#     # Load and plot data per year   
#     for i, yr in enumerate(years_requested_all):
        
#         elev = elevation[i,:]
#         mask = elev.mask
#         plt.plot(crossshore[~mask], elev[~mask], color=colorVal, label = str(yr), linewidth = 2.5)
    
#         colorVal = scalarMap.to_rgba(yr)
#         plt.plot(dunefoots_x[i], dunefoots_y[i], 'o', markersize=13, color = colorVal, mew=3)        
            
#     plt.plot(dunefoots_x[0], dunefoots_y[0], 'o', markersize=13, color = 'grey', mew=3, label = 'Dune toe')
    
#     plt.legend(loc='upper right', fontsize = 22)
    
#     ax.set_xlim(xlimit)
#     ax.set_ylim(ylimit)
    
#     # Show the figure    
#     plt.show()
    
#     # Save figure as png in predefined directory
#     plt.savefig(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.png')
#     pickle.dump(fig, open(config['outputdir'] + config['save locations']['DirB2'] + transect + '_DFplot' + '.fig.pickle', 'wb'))
    
#     plt.close()

