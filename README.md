# Jarkus Analysis Toolbox

The Jarkus Analysis Toolbox (JAT) is a Python-based open-source software, that can be used to analyze the Jarkus dataset. The Jarkus dataset is one of the most elaborate coastal datasets in the world and consists of coastal profiles of the entire Dutch coast, spaced about 250-500 m apart, which have been measured yearly since 1965. The main purpose of the JAT is to provide stakeholders (e.g. scientists, engineers and coastal managers) with the techniques that are necessary to study the spatial and temporal variations in characteristic parameters like dune height, dune volume, dune foot, beach width and closure depth. Different available definitions for extracting these characteristic parameters were collected and implemented in the JAT. 

<img src="https://github.com/christavanijzendoorn/JAT/blob/master/images/parameters.png" width="500">
*Example of characteristic parameters that can be extracted using the JAT*

The modular set-up of the JAT makes sure that new extraction techniques can be added in the future. For instance, we are currently working on the inclusion of:
- a closure depth method
- the python based version of the second derivative method
- the momentary coastline calculation

The output of the extraction methods for all years and transects produced with the JAT are available online (ref. TUdelft server). In the future, some examples of visualizing these data will be provided in this repository.

# Method

<img src="https://github.com/christavanijzendoorn/JAT/blob/master/images/flowchart.png" width="500">
*Flowchart of Jarkus Analysis Toolbox functionalities*

# Installation

'''python setup.py install'''

# Usage

'''import JAT

# Here include more info on how to use

'''

Refer to [examples](https://github.com/christavanijzendoorn/JAT/tree/master/Examples)

# Source data

[The Jarkus dataset](http://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/jarkus/profiles/transect.nc)

[The dune foots extracted using the second derivative method](http://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/DuneFoot/DF_2nd_deriv.nc)

You can save these files locally to be independent of internet access. Make sure to include their directory in the settings file (jarkus.yml).

# Settings

Use the jarkus.yml file to input the settings.
Create folder for the output documents and include their directory in the settings file.

The chosen years and jarkus transect numbers are filled in in the working files for ease of access.

## User-defined settings
- Filter1 = profiles to nan if data is missing between -1 and 5 m.
- Filter2 = years or profile locations are removed if too much data is missing
- Assumed elevations of various variables are included
- For landward derivative and seaward DoC threshold values are defined
- Normalization year = year used to normalize cross-shore varying variables, so they become comparable.

# Jarkus transect definitions

Vaknummer + raainummer = VNNNNNN: 
- always 6 transect (raai) related numbers
- 1 or 2 kustvak related numbers, 2 in case of kustvak of 10+

Example Sand Engine: Vak 9, raai 11109 = 9011109

Overview of transects: https://maps.rijkswaterstaat.nl/geoweb55/index.html?viewer=Kustlijnkaart 
Overview of transects and ‘kustvakken’: http://publicaties.minienm.nl/documenten/kustlijnkaarten-2020-resultaten-beoordeling-ligging-kustlijn-op-1-januari-2020

# Available characteristic parameters

### The use of characteristic parameters in the JAT

Parameter | Variable name | Variable output | Dependent on
------------ | ------------- | ------------- | ------------- 
Dune height - Primary | dune_height_and_location | DuneTop_prim_x, DuneTop_prim_y | -
Dune height - Secondary | dune_height_and_location | DuneTop_prim_x, DuneTop_prim_y, DuneTop_sec_x, DuneTop_sec_y | -
Mean Sea Level | mean_sea_level | MSL_x | -
&nbsp; | mean_sea_level_variable | MSL_x_var | MLW_x_var & MHW_x_var
Mean Low Water | mean_low_water_fixed | MLW_x_fix | -
&nbsp; | mean_low_water_variable | MLW_y_var, MLW_x_var | -
Mean High Water | mean_high_water_fixed | MHW_x_fix | -
&nbsp; | mean_high_water_variable | MHW_y_var, MHW_x_var | -
Intertidal area width | Intertidal_width_fix | MLW_x_var & MHW_x_var
&nbsp; | Intertidal_width_var | MLW_x_var & MHW_x_var
Landward boundary | landward_point_variance | Landward_x_variance | DuneTop_prim_x 
&nbsp; | landward_point_derivative | Landward_x_der | MHW_y_var 
&nbsp; | landward_point_bma | Landward_x_bma | - 
Seaward boundary | seaward_point_foreshore | Seaward_x_FS | -
&nbsp; | seaward_point_activeprofile | Seaward_x_AP | -
&nbsp;   | seaward_point_doc** | Seaward_x_mindepth, Seaward_x_DoC | -
Dune foot | dune_foot_fixed | Dunefoot_x_fix | -
&nbsp; | dune_foot_derivative | Dunefoot_y_der, Dunefoot_x_der | -
&nbsp;   | dune_foot_pybeach | Dunefoot_y_pybeach_mix, Dunefoot_x_pybeach_mix | MHW_x_var & Landward_x_der  
Beach width | beach_width_fix | Beach_width_fix | MSL_x & Dunefoot_x_fix
&nbsp; | beach_width_var | Beach_width_var | MSL_x_var & Dunefoot_x_fix
&nbsp;   | beach_width_der | Beach_width_der| MSL_x & Dunefoot_x_der
&nbsp;   | beach_width_der_var | Beach_width_der_var | MSL_x_var & Dunefoot_x_der 
Beach gradient | beach_gradient_fix | Beach_gradient_fix | MSL_x & Dunefoot_x_fix
&nbsp; | beach_gradient_var | Beach_gradient_var | MSL_x_var & Dunefoot_x_fix
&nbsp;   | beach_gradient_der | Beach_gradient_der| MSL_x & Dunefoot_x_der
Dune front width | dune_front_width_prim_fix | Dunefront_width_prim_fix | Dunefoot_x_fix & DuneTop_prim_x
&nbsp; | dune_front_width_prim_der | Dunefront_width_prim_der | Dunefoot_x_der & DuneTop_prim_x
&nbsp;   | dune_front_width_sec_fix | Dunefront_width_sec_fix| Dunefoot_x_fix & DuneTop_sec_x
&nbsp;   | dune_front_width_sec_der | Dunefront_width_sec_der | Dunefoot_x_der & DuneTop_sec_x
Dune front gradient | dune_front_gradient_prim_fix | Dunefront_gradient_prim_fix | Dunefoot_x_fix & DuneTop_prim_x
&nbsp; | dune_front_gradient_prim_der | Dunefront_gradient_prim_der | Dunefoot_x_der & DuneTop_prim_x
&nbsp;   | dune_front_gradient_sec_fix | Dunefront_gradient_sec_fix| Dunefoot_x_fix & DuneTop_sec_x
&nbsp;   | dune_front_gradient_sec_der | Dunefront_gradient_sec_der | Dunefoot_x_der & DuneTop_sec_x
Dune volume | dune_volume_fix | DuneVol_fix | Dunefoot_x_fix & Landward_x_variance
&nbsp; | dune_volume_der | DuneVol_der | Dunefoot_x_der & Landward_x_variance
Intertidal area gradient | intertidal_gradient | Intertidal_gradient_fix | MLW_x_fix & MHW_x_fix
Intertidal area volume | intertidal_volume_fix | Intertidal_volume_fix | MLW_x_fix & MHW_x_fix
&nbsp; | intertidal_volume_var | Intertidal_volume_var | MLW_x_var & MHW_x_var
Foreshore gradient | foreshore_gradient | Foreshore_gradient | Seaward_x_FS & Landward_x_bma
Foreshore volume | foreshore_volume | Foreshore_volume | Seaward_x_FS & Landward_x_bma
Active profile gradient | active_profile_gradient | Active_profile_gradient | Seaward_x_AP & Landward_x_bma
Active profile volume | active_profile_volume | Active_profile_volume | Seaward_x_AP & Landward_x_bma

### Explanation of characteristic parameters

Parameter | Type | Explanation
------------ | ------------- | ------------ 
Dune height | Primary  | Cross-shore location and elevation of the primary dune peak with elevation and prominence above user-defined value
&nbsp; | Secondary | Cross-shore location and elevation of the secondary dune peak with elevation and prominence above user-defined value
Mean Sea Level | Fixed | Elevation and cross-shore location of Mean Sea level defined as user-defined value w.r.t. reference datum (default = 0)
&nbsp;| Variable | Elevation and cross-shore location of Mean Sea level defined as the cross-shore location between the variable MHW and MLW location
Mean Low Water | Fixed | Cross-shore location of Mean Low Water defined as user-defined value w.r.t. reference datum (default =  -1 m)
&nbsp; | Variable | Elevation and cross-shore location of Mean Low Water based on value w.r.t. reference datum provided per transect in the Jarkus dataset* based on tidal modeling
Mean High Water | Fixed | Cross-shore location of Mean High Water defined as user-defined value m w.r.t. reference datum (default =  +1 m)
&nbsp; | Variable | Elevation and cross-shore location of Mean High Water based on value w.r.t. reference datum provided per transect in the Jarkus dataset* based on tidal modeling
Intertidal area width | Fixed | Cross-shore width between the fixed Mean High Water and Mean Low Water MHW-MLW
&nbsp; | Variable | Cross-shore width between the variable Mean High Water and Mean Low Water MHW-MLW
Landward boundary | Variance | Landward boundary where variance of elevation through time is below a user-defined threshold (default = 0.1)
&nbsp; | Derivative | Landward boundary defined as dune peak above a fixed threshold (default = +2.4 m) and with a maximum elevation (defulat = +6.0m) used for 2nd derivative method (Diamantidou, 2019)
&nbsp; | BMA | Cross-shore location of value w.r.t. reference datum that approximates the Boundary between the Marine and Aeolian zone (BMA) (De Vries et al., 2010) (default =  +2 m)
Seaward boundary | Foreshore | Cross-shore location of value w.r.t. reference datum that approximates the seaward boundary of the foreshore (default = -4.0m)
&nbsp; | Active Profile | Cross-shore location of value w.r.t. reference datum that approximates the seaward boundary of the active profile (default = -8.0m)
&nbsp;   | Depth of Closure** | Approximation of the depth of closure below a user-defined minimum  (default = -5.0m) where the standard deviation of the elevation through time is below a user-defined value (default = 0.25) for at least a user-defined length (default = 200m) (Hinton, 2000)
Dune foot | Fixed | Cross-shore location of the dune foot based on user-defined value (default = +3.0, assumed value for Dutch coast)
&nbsp; | Derivative | Elevation and cross-shore location based on the second derivative method by Diamantidou (2019)
&nbsp;   | Pybeach | Elevation and  cross-shore location based on the pybeach method by Beuzen (2019) uses mixed classifier
Beach width | Fixed | Beach width calculated as cross-shore distance between the fixed mean sea level and fixed dune foot
&nbsp; | Variable | Beach width calculated as cross-shore distance between the varibale mean sea level and fixed dune foot
&nbsp;   | Derivative | Beach width calculated as cross-shore distance between the fixed mean sea level and dune foot based on the second derivative method
&nbsp;   | Variable Derivative | Beach width calculated as cross-shore distance between the varaible mean sea level and dune foot based on the second derivative method
Beach gradient | Fixed | Beach gradient calculated as the slope between the fixed mean sea level and fixed dune foot
&nbsp; | Variable | Beach gradient calculated as the slope between the varibale mean sea level and fixed dune foot
&nbsp;   | Derivative | Beach gradient calculated as the slope between the fixed mean sea level and dune foot based on the second derivative method   
Dune front width | Primary Fixed | Dune front width calculated as cross-shore distance between the primary dune peak and fixed dune foot
&nbsp; | Primary Derivative | Dune front width calculated as cross-shore distance between the primary dune peak and dune foot based on the second derivative method  
&nbsp;   | Secondary Fixed | Dune front width calculated as cross-shore distance between the secondary dune peak and fixed dune foot
&nbsp;   | Secondary Derivative | Dune front width calculated as cross-shore distance between the secondary dune peak and dune foot based on the second derivative method  
Dune front gradient | Primary Fixed | Dune front gradient calculated as cross-shore distance between the primary dune peak and fixed dune foot
&nbsp; | Primary Derivative | Dune front gradient calculated as cross-shore distance between the primary dune peak and dune foot based on the second derivative method  
&nbsp;   | Secondary Fixed | Dune front gradient calculated as cross-shore distance between the secondary dune peak and fixed dune foot
&nbsp;   | Secondary Derivative | Dune front gradient calculated as cross-shore distance between the secondary dune peak and dune foot based on the second derivative method  
Dune volume | Fixed | Dune volume calculated as the volume under the coastal profile between location of the fixed dune foot and the landward boundary based on the variance
&nbsp; | Derivative | Dune volume calculated as the volume under the coastal profile between location of the dune foot based on the second derivative method  and the landward boundary based on the variance
Intertidal area gradient | Fixed | Gradient of the profile between the fixed Mean High Water and Mean Low Water
Intertidal area volume | Fixed | Volume of the intertidal area calculated as the volume under the profile between the location of the fixed Mean High Water and Mean Low Water
&nbsp; | Variable | Volume of the intertidal area calculated as the volume under the profile between the location of the variable Mean High Water and Mean Low Water
Foreshore gradient | BMA | Gradient of the foreshore calculated as the slope between the BMA and the seaward boundary of the foreshore
Foreshore volume | BMA | Volume of the foreshore calculated as the volume under the profile between the BMA and the seaward boundary of the foreshore
Active profile gradient | BMA | Gradient of the active profile calculated as the slope between the BMA and the seaward boundary of the active profile
Active profile volume | BMA | Volume of the active profile calculated as the volume under the profile BMA and the seaward boundary of the active profile


\* These values vary alongshore (per transect), but are constant through time (per year)

** It should be checked whether this method corresponds to the way it was implemented by Nicha Zwarenstein Tutunji in his MSc work




