# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:13:53 2020

@author: cijzendoornvan
"""

get_ipython().run_line_magic('matplotlib', 'auto') ## %matplotlib auto TO GET WINDOW FIGURE

import pandas as pd
import matplotlib.pyplot as plt

SLR = pd.read_excel("C:/Users/cijzendoornvan/Documents/DuneForce/JARKUS/zeespiegelstijging_clo.xlsx")

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(SLR['Jaren'], SLR['Trend'], color = '#4169E1')#, marker = 'o', markersize = 8)
ax.fill_between(SLR['Jaren'], SLR['Onzekerheid trend Bandbreedte min'], SLR['Onzekerheid trend Bandbreedte max'], color = '#4169E1', alpha = 0.15)
ax.scatter(SLR['Jaren'], SLR['Jaargemiddelde 6 kuststations'], color = 'grey', alpha = 0.6)#, marker = 'o', markersize = 8)

ax.tick_params(axis='both', which='major', labelsize=22)
ax.tick_params(axis='x', which='major', rotation=30)
ax.set_ylabel("Sea level (cm above NAP)", fontsize=26)
ax.set_xlabel("Year", fontsize=26)
legend = ax.legend(loc='upper left', title = 'Sea level rise', labels = ['Trend of 1.9 mm/yr', 'Trend uncertainty', 'Yearly average of \n6 coastal stations'], fontsize=24)
plt.setp(legend.get_title(),fontsize=26)
legend._legend_box.align = "left"

