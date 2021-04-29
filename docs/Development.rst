Development
------------


Suggested improvements and additions
=====================================

* Include Momentary Coastline Calculation and BKL
* Include Depth of Closure - both cross-shore and elevation - check with Nicha's work
* Include 2nd derivative method python version - based on current matlab-based method
* Include example of nourishment filter use - should work with .nc file on opendap
* Add active profile calculation based on landward variance boundary and DoC
* Add extraction of lon and lat - currently only cross-shore values are available after extraction
* Add ... many other extraction methods that are available

Adding new extraction methods
==============================

The JAT currently provides a large range of extraction methods, but many more could be introduced. Below you find the most important steps for adding a new method to the JAT:

* Add the extraction method in the `Jarkus_analysis_toolbox.py` in class `Extraction` as `def get_parameter` (fill in appropriate name `for parameter`). It is best to use the other available extraction methods as inspiration so the method is comparable. For the assigment of the output into the dimensions dataframe use an appropriate variable name.
* Add an `if statement` for the extraction of the parameter in `get_all_dimensions` of `Extraction` class in `Jarkus_analysis_toolbox.py`
* Add the parameter name in the configuration file so the `if statement` (reference above) will work. This means the parameter name should be added in both dimensions→setting→variablename with a True/False statement, and in dimensions→variables→ variablename with the variable names as assigned in the `get_parameter` function, in the jarkus.yml file.
* Add these assigned variable names to the plot_tiles.yml file, so the distribution figures can be generated automatically. Note, that for cross-shore variables it is crucial to add an 'x' in the variable name, so it is picked up in the automatic normalization (Extraction → normalize_dimensions). Then, add a suitable title and label name for in the figure.
* Make sure to update the documentation!