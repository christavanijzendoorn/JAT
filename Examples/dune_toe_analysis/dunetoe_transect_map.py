# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:14:22 2020

@author: amton
"""

####
# Basemap version 1.2.0 and basemap-data-hires need specific versions of matplotlib to work
# https://anaconda.org/conda-forge/basemap-data-hires
# https://anaconda.org/conda-forge/basemap 
# matplotlib=3.0.0

# use anaconda prompt and go to directory where environment file is located
# conda env create -f dune_transect_map.yml
# conda activate map
# python dunetoe_transect_map.py
####

import yaml
import mpl_toolkits

import matplotlib.pyplot as plt
import os
os.environ['PROJ_IB'] = r'C:\Users\cijzendoornvan\AppData\Local\Continuum\anaconda3\envs\map\Lib\site-packages' 
mpl_toolkits.__path__.append(r'C:\Users\cijzendoornvan\AppData\Local\Continuum\anaconda3\envs\map\Lib\site-packages\mpl_toolkits')
from mpl_toolkits.basemap import Basemap

config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/dune_toe_analysis/jarkus_04.yml"))
DirFigures = config['outputdir'] + config['save locations']['DirFig']
if os.path.isdir(DirFigures) == False:
            os.mkdir(DirFigures)

#%% Plot zoom

fig = plt.figure(figsize=(12,9))

m = Basemap(projection='merc',
            llcrnrlat=50.7,
            urcrnrlat=53.7,
            llcrnrlon=3.25,
            urcrnrlon=7.4,
            lat_ts=20,
            resolution='h') #c = crude, h = high, f = full, i = 

m.drawcoastlines()
m.drawcountries(color='black',linewidth=1)

m.drawmapboundary(color='white', linewidth=1, fill_color='white')
m.fillcontinents(color='white') #lake_color='white'

m.drawlsmask(land_color='lightgreen', ocean_color='white', lakes=True)
m.drawmapscale(3.6, 53.5, 4, 52, 50)

plt.show()

plt.savefig(DirFigures + 'map_nederland', bbox_inches = 'tight', dpi=300)


