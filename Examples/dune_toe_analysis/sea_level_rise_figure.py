# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:13:53 2020

@author: cijzendoornvan
"""

######################################
# THIS SCRIPT CREATES FIGURE 4
######################################

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

######################
# LOAD SETTINGS + DATA
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/dune_toe_analysis/jarkus_04.yml"))

DirSLR = config['inputdir'] + config['data locations']['SLR']
DirFigures = config['outputdir'] + config['save locations']['DirFig']
if os.path.isdir(DirFigures) == False:
            os.mkdir(DirFigures)
            
SLR = pd.read_excel(DirSLR)

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(SLR['Jaren'], SLR['Trend'], color = '#4169E1')#, marker = 'o', markersize = 8)
ax.fill_between(SLR['Jaren'], SLR['Onzekerheid trend Bandbreedte min'], SLR['Onzekerheid trend Bandbreedte max'], color = '#4169E1', alpha = 0.15)
ax.scatter(SLR['Jaren'], SLR['Jaargemiddelde 6 kuststations'], color = 'grey', alpha = 0.6)#, marker = 'o', markersize = 8)

ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='x', which='major', rotation=30)
ax.set_ylabel("Sea level (cm above NAP)", fontsize=34)
ax.set_xlabel("Year", fontsize=34)
legend = ax.legend(loc='upper left', title = 'Sea level rise', labels = ['Trend of 1.9 mm/yr', 'Trend uncertainty', 'Yearly average of \n6 coastal stations'], fontsize=30)
plt.setp(legend.get_title(),fontsize=34)
legend._legend_box.align = "left"

DirDFAnalysis = "C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/ANALYSIS/DF_analysis//"    

file_name = 'SLR'
filename_pdf = DirFigures + file_name + '.pdf'
# filename_eps = DirFigures + file_name + '.eps'
filename_png = DirFigures + file_name + '.png'
plt.savefig(filename_png, bbox_inches='tight')
# plt.savefig(filename_eps, format='eps', bbox_inches='tight')
plt.savefig(filename_pdf, bbox_inches='tight')
print('saved figure')