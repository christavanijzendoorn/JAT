# Jarkus Analysis Toolbox

The Jarkus Analysis Toolbox (JAT) is a Python-based open-source software, that can be used to analyze the Jarkus dataset. The Jarkus dataset is one of the most elaborate coastal datasets in the world and consists of coastal profiles of the entire Dutch coast, spaced about 250-500 m apart, which have been measured yearly since 1965. The main purpose of the JAT is to provide stakeholders (e.g. scientists, engineers and coastal managers) with the techniques that are necessary to study the spatial and temporal variations in characteristic parameters like dune height, dune volume, dune foot, beach width and closure depth. Different available definitions for extracting these characteristic parameters were collected and implemented in the JAT. 

![Example of characteristic parameters that can be extracted using the JAT](https://github.com/christavanijzendoorn/JAT/blob/master/images/parameters.png)

The modular set-up of the JAT makes sure that new extraction techniques can be added in the future. For instance, we are currently working on the inclusion of:
- a closure depth method
- the python based version of the second derivative method
- the momentary coastline calculation

The output of the extraction methods for all years and transects produced with the JAT are available online (ref. TUdelft server). In the future, some examples of visualizing these data will be provided in this repository.

# Method

![Flowchart of Jarkus Analysis Toolbox functionalities](https://github.com/christavanijzendoorn/JAT/blob/master/images/flowchart.png)


# Available characteristic parameters

Parameter | Variable name | Explanation | Variable output | Dependent on
------------ | ------------- | ------------ | ------------- | ------------- 
Dune height - Primary | dune_height_and_location | Cross-shore location and elevation of the primary dune peak with elevation and prominence above user-defined value | DuneTop_prim_x, DuneTop_prim_y | -
Dune height - Secondary | dune_height_and_location | Cross-shore location and elevation of the secondary dune peak with elevation and prominence above user-defined value | DuneTop_prim_x, DuneTop_prim_y, DuneTop_sec_x, DuneTop_sec_y | -
Mean Sea Level | mean_sea_level | Elevation and cross-shore location of Mean Sea level defined as user-defined value w.r.t. reference datum (default = 0) | MSL_x | -
&nbsp; | mean_sea_level_variable | Elevation and cross-shore location of Mean Sea level defined as the cross-shore location between the variable MHW and MLW location (MLW+MHW)/2 | MSL_x_var | MLW_x_var & MHW_x_var
Mean Low Water | mean_low_water_fixed | Cross-shore location of Mean Low Water defined as user-defined value w.r.t. reference datum (default =  -1 m) | MLW_x_fix | -
&nbsp; | mean_low_water_variable | Elevation and cross-shore location of Mean Low Water based on value w.r.t. reference datum provided per transect in the Jarkus dataset* based on tidal modeling | MLW_y_var, MLW_x_var | -
Mean High Water | mean_high_water_fixed | Cross-shore location of Mean High Water defined as user-defined value m w.r.t. reference datum (default =  +1 m) | MHW_x_fix | -
&nbsp; | mean_high_water_variable | Elevation and cross-shore location of Mean High Water based on value w.r.t. reference datum provided per transect in the Jarkus dataset* based on tidal modeling | MHW_y_var, MHW_x_var | -
Intertidal area width | Intertidal_width_fix | Cross-shore width between the fixed Mean High Water and Mean Low Water MHW-MLW | MLW_x_var & MHW_x_var
&nbsp; | Intertidal_width_var | Cross-shore width between the variable Mean High Water and Mean Low Water MHW-MLW | MLW_x_var & MHW_x_var
Landward boundary | landward_point_variance | Landward boundary where variance of elevation through time is below a user-defined threshold (default = 0.1) | Landward_x_variance | DuneTop_prim_x 
&nbsp; | landward_point_derivative | Landward boundary defined as dune peak above a fixed threshold (default = +2.4 m) and with a maximum elevation (defulat = +6.0m) used for 2nd derivative method (Diamantidou, 2019) | Landward_x_der | MHW_y_var 
&nbsp; | landward_point_bma | Cross-shore location of value w.r.t. reference datum that approximates the Boundary between the Marine and Aeolian zone (BMA) (De Vries et al., 2010) (default =  +2 m) | Landward_x_bma | - 
Seaward boundary | seaward_point_foreshore | Cross-shore location of value w.r.t. reference datum that approximates the seaward boundary of the foreshore (default = -4.0m) | Seaward_x_FS | -
&nbsp; | seaward_point_activeprofile | Cross-shore location of value w.r.t. reference datum that approximates the seaward boundary of the active profile (default = -8.0m)| Seaward_x_AP | -
&nbsp;   | seaward_point_doc** | Approximation of the depth of closure below a user-defined minimum  (default = -5.0m) where the standard deviation of the elevation through time is below a user-defined value (default = 0.25) for at least a user-defined length (default = 200m) (Hinton, 2000) | Seaward_x_mindepth, Seaward_x_DoC | -
   

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
   
      
   

* These values vary alongshore (per transect), but are constant through time (per year). 
** It should be checked whether this method corresponds to the way it was implemented by Nicha Zwarenstein Tutunji in his MSc work

        
        dune_foot_fixed             : ['Dunefoot_x_fix']
        dune_foot_derivative        : ['Dunefoot_y_der', 'Dunefoot_x_der']
        dune_foot_pybeach           : ['Dunefoot_y_pybeach_mix', 'Dunefoot_x_pybeach_mix']
        
        beach_width_fix             : ['Beach_width_fix']
        beach_width_var             : ['Beach_width_var']
        beach_width_der             : ['Beach_width_der']
        beach_width_der_var         : ['Beach_width_der_var']
        
        beach_gradient_fix          : ['Beach_gradient_fix']
        beach_gradient_var          : ['Beach_gradient_var']
        beach_gradient_der          : ['Beach_gradient_der']

        dune_front_width_prim_fix   : ['Dunefront_width_prim_fix']
        dune_front_width_prim_der   : ['Dunefront_width_prim_der']
        dune_front_width_sec_fix    : ['Dunefront_width_sec_fix']
        dune_front_width_sec_der    : ['Dunefront_width_sec_der']
        
        dune_front_gradient_prim_fix: ['Dunefront_gradient_prim_fix']
        dune_front_gradient_prim_der: ['Dunefront_gradient_prim_der']
        dune_front_gradient_sec_fix : ['Dunefront_gradient_sec_fix']
        dune_front_gradient_sec_der : ['Dunefront_gradient_sec_der']
        
        dune_volume_fix             : ['DuneVol_fix']
        dune_volume_der             : ['DuneVol_der']
        
        intertidal_gradient         : ['Intertidal_gradient_fix']
        intertidal_volume_fix       : ['Intertidal_volume_fix']
        intertidal_volume_var       : ['Intertidal_volume_var']
        
        foreshore_gradient          : ['Foreshore_gradient']
        foreshore_volume            : ['Foreshore_volume']
        
        active_profile_gradient     : ['Active_profile_gradient']
        active_profile_volume       : ['Active_profile_volume']

# Source data

The Jarkus dataset:
http://opendap.deltares.nl/thredds/catalog/opendap/rijkswaterstaat/jarkus/profiles/catalog.html?dataset=varopendap/rijkswaterstaat/jarkus/profiles/transect.nc  

The dune foots extracted using the second derivative method:
http://opendap.deltares.nl/thredds/catalog/opendap/rijkswaterstaat/DuneFoot/catalog.html?dataset=varopendap/rijkswaterstaat/DuneFoot/DF_2nd_deriv.nc 

you can save these files locally to be independent of internet access. Make sure to include their directory in the settings file (jarkus.yml).

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




