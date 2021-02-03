# Jarkus Analysis Toolbox

The Jarkus Analysis Toolbox (JAT) is a Python-based open-source software, that can be used to analyze the Jarkus dataset. The Jarkus dataset is one of the most elaborate coastal datasets in the world and consists of coastal profiles of the entire Dutch coast, spaced about 250-500 m apart, which have been measured yearly since 1965. The main purpose of the JAT is to provide stakeholders (e.g. scientists, engineers and coastal managers) with the techniques that are necessary to study the spatial and temporal variations in characteristic parameters like dune height, dune volume, dune foot, beach width and closure depth. Different available definitions for extracting these characteristic parameters were collected and implemented in the JAT. 

![Example of characteristic parameters that can be extracted using the JAT](https://github.com/christavanijzendoorn/JAT/images/parameters.png)

The modular set-up of the JAT makes sure that new extraction techniques can be added in the future. For instance, we are currently working on the inclusion of:
- a closure depth method
- the python based version of the second derivative method
- the momentary coastline calculation

The output of the extraction methods for all years and transects produced with the JAT are available online (ref. TUdelft server). In the future, some examples of visualizing these data will be provided in this repository.

# Method

![Flowchart of Jarkus Analysis Toolbox functionalities](https://github.com/christavanijzendoorn/JAT/images/flowchart.png)


# Available characteristic parameters



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




