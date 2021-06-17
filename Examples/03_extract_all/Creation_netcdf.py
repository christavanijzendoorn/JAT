# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:41:58 2021

@author: cijzendoornvan
"""
######################
# PACKAGES
######################
import yaml
import pickle
import numpy as np
from datetime import datetime
from netCDF4 import Dataset, stringtochar
from JAT.Jarkus_Analysis_Toolbox import Transects, Extraction

######################
# LOAD SETTINGS
######################
config = yaml.safe_load(open("C:/Users/cijzendoornvan/OneDrive - Delft University of Technology/Documents/DuneForce/JARKUS/JAT/Examples/03_extract_all/jarkus_03.yml"))
areanames_dict = yaml.safe_load(open(config['inputdir'] + config['data locations']['AreaNames'])) 
metadata = yaml.safe_load(open(config['inputdir'] + config['data locations']['MetaData'])) 

# Load jarkus dataset
data = Transects(config)

#%%##############################
####      EXECUTE            ####
#################################
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")

cx = list(range(-3000, 9315,5))

version = 0.1

extract = Extraction(data, config) # initalize the extra class 
variables = extract.get_requested_variables() # get all variables that were requested (based on jarkus.yml file)

# get one variable to extract shape
variable = variables[0]
dimension = pickle.load(open(config['outputdir'] + config['save locations']['DirD'] + variable + '_dataframe.pickle','rb')) 

# create netcdf file
netcdf_dir = config['inputdir'] + 'extracted_parameters.nc'
ds = Dataset(netcdf_dir, 'w', format='NETCDF4')

# assign global metadata
ds.naming_authority = 'tudelft'
ds.title = 'characteristic parameters extracted from the JarKus dataset using the JAT'
ds.summary =  'The Jarkus Analysis Toolbox (JAT) was used to extract a range of characteristic parameters from the JarKus dataset. This file provides the locations of characteristic parameters in all coastal profiles measured since 1965'
ds.keywords = 'coastal profile, Jarkus, python'
ds.history = 'Data extracted with version ' + str(version) + ' of the JAT. NetCDF created on ' + current_time + ' by Christa van IJzendoorn with Creation_netcdf.py'
ds.institution = 'Delft University of Technology'
ds.source = 'https://opendap.deltares.nl/thredds/catalog/opendap/rijkswaterstaat/jarkus/profiles/catalog.html?dataset=varopendap/rijkswaterstaat/jarkus/profiles/transect.nc (Rijkswaterstaat)'
ds.references = '4TU storage: https://doi.org/10.4121/c.5335433, software: https://github.com/christavanijzendoorn/JAT, documentation: https://jarkus-analysis-toolbox.readthedocs.io/, original source: http://www.rijkswaterstaat.nl' 
ds.creator_name = 'Christa van IJzendoorn'
ds.creater_url = 'https://www.tudelft.nl/citg/over-faculteit/afdelingen/hydraulic-engineering/sections/coastal-engineering/staff/co-christa-van-ijzendoorn-msc'
ds.creator_email = 'c.o.vanijzendoorn@tudelft.nl'
ds.processing_level = 'preliminary'
ds.version = version
ds.conventions = 'CF-1.6'
ds.license: 'These data and the software that was used to create them can be used freely for research purposes (GPL-3.0 License) provided that the source of the JarKus data (RIJKSWATERSTAAT) is acknowledged. disclaimer: This data is made available in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.'

# assign dimensions
time = ds.createDimension("time", dimension.shape[0]) 
alongshore = ds.createDimension("alongshore", dimension.shape[1]) 
cross_shore = ds.createDimension("cross_shore", len(cx)) 
bounds2 = ds.createDimension("bounds2", 2) 
stringsize = ds.createDimension("stringsize", 100) 

# set general variables

# set ids
trscts_int = [int(i) for i in dimension.columns]
ids = ds.createVariable('id', 'i4', ('alongshore'))
ids[:] = trscts_int
ids.long_name = 'identifier'
ids.comment = 'sum of area code (x1000000) and alongshore coordinate'

# set areacode
codes = np.around(np.array(trscts_int) / 1000000)
areacode = ds.createVariable('areacode', 'i4', ('alongshore'))
areacode[:] = codes
areacode.long_name = 'area code'
areacode.comment = 'codes for the 15 coastal areas as defined by rijkswaterstaat'

# set areaname
areanames = []
for i in codes:
    areanames.append(areanames_dict[i])
areaname = ds.createVariable('areaname', 'S1', ('alongshore', 'stringsize'))
areaname[:] = stringtochar(np.array(areanames, dtype='S100'))
areaname.long_name = 'area name'
areaname.comment = 'names for the 15 coastal areas as defined by rijkswaterstaat'

# alongshore coordinate
alongshore = ds.createVariable('alongshore', 'S1', ('alongshore'))
alongshore[:] = np.array(trscts_int) - np.array(codes)*1000000
alongshore.long_name = 'alongshore coordinate'
alongshore.unit = 'm'
alongshore.comment = 'alongshore coordinate relative to the rsp (rijks strand paal)'

# crossshore 
cross_shore = ds.createVariable('cross_shore', 'i4', ('cross_shore'))
cross_shore.long_name = 'cross-shore coordinate'
cross_shore.unit = 'm'
cross_shore.comment = 'cross-shore coordinate relative to the rsp (rijks strand paal)'

# time
times = []
start_dates = []
end_dates = []
for t in dimension.index:
    then = datetime(t, 7, 1)
    begin = datetime(t, 1, 1)
    end = datetime(t, 12, 31)
    ref = datetime(1970, 1, 1)
    times.append(((then - ref).days))
    start_dates.append(((begin - ref).days))
    end_dates.append(((end - ref).days))
time = ds.createVariable('time', 'i4', ('time'))  
time[:] = times
time.standard_name = 'time'
time.units = 'days since 1970-01-01'

# time_bounds
time_bounds = ds.createVariable('time_bounds', 'i4', ('bounds2', 'time')) 
timebounds = np.array([start_dates, end_dates]).T
time_bounds[:,:]  = timebounds
time_bounds.standard_name = 'time'
time_bounds.units = 'days since 1970-01-01'


# set variable for each characteristic parameter
for variable in variables:
    dimension = pickle.load(open(config['outputdir'] + config['save locations']['DirD'] + variable + '_dataframe.pickle','rb')) 

    value = ds.createVariable(variable, 'f4', ('time', 'alongshore'))
    value.units = metadata[variable][1]
    value.long_name = metadata[variable][0]
    
    value[:,:] = np.array(dimension.values)

ds.close()

#%%##############################
####      TEST LOADING       ####
#################################

# Load netcdf 
char_params = Dataset(netcdf_dir)

# Get global attributes
for name in char_params.ncattrs():
    print(name + ' = ' + str(getattr(char_params, name)))

# Get all dimensions
print(char_params.dimensions.values())

# Get all variables info
print(char_params.variables.values())

# Get info for one variable
print(char_params.variables['DuneTop_prim_x'].__dict__)

# Get values of one variable
dunetops = char_params.variables['DuneTop_prim_x'][:,:]

dunevols = char_params.variables['DuneVol_fix'][:,:]




