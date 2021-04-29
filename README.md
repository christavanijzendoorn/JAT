# Jarkus Analysis Toolbox

The Jarkus Analysis Toolbox (JAT) is a Python-based open-source software, that can be used to analyze the Jarkus dataset. The Jarkus dataset is one of the most elaborate coastal datasets in the world and consists of coastal profiles of the entire Dutch coast, spaced about 250-500 m apart, which have been measured yearly since 1965. The main purpose of the JAT is to provide stakeholders (e.g. scientists, engineers and coastal managers) with the techniques that are necessary to study the spatial and temporal variations in characteristic parameters like dune height, dune volume, dune foot, beach width and closure depth. Different available definitions for extracting these characteristic parameters were collected and implemented in the JAT. 

<img src="https://github.com/christavanijzendoorn/JAT/blob/master/images/parameters.png" width="500">
Example of characteristic parameters that can be extracted using the JAT


The modular set-up of the JAT makes sure that new extraction techniques can be added in the future. For instance, expected extraction methods are:
- a closure depth method
- the python based version of the second derivative method
- the momentary coastline calculation

The output of the extraction methods for all years and transects produced with the JAT are available on the [4TU repository](https://doi.org/10.4121/14514213). Example 5 in this repository provides a script that can be used to load and visualize these data.

For a complete guide on how to use the Jarkus Analysis Toolbox, with examples and explanation of the available characterstic parameters, go to the [Documentation](https://jarkus-analysis-toolbox.readthedocs.io/). 


