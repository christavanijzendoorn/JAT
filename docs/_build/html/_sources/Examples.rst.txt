Examples
----------

After going through the :doc:`Gettingstarted`: section you can try out the Jarkus Analysis Toolbox with th following examples.

1. Single transect
====================
This example provides the code necessary to extract all characteristic parameters from one single transect. The default settings in the `jarkus_01.yml` file  look at the years from 1980 to 2020 and at transect location 8009325. It is adviced to play around with these settings and extract the characteristic parameters for different periods and location. To extract the characteristic parameters open `JAT_use_single_transect.py` in the Python IDE of your choice (I use Spyder) and run the commands step by step. This should show you the steps necessary to extract the characteristic parameters for one transect and gives examples how these data can be visualized.


2. Regional analysis
=====================
Example 2 shows how to extract the characteristic parameters from multiple transects at once.


3. Extract all
===============
This Example shows how to extract all characteristic parameters from all transect locations. It provides the scripts that were used to create the dataset of characteristic parameters in the 4TU repository.
Note, this analysis can take a long time, around 10 hours. The `Filtering_execution.py` file provides an example of how the filtering functionalities of the JAT can be used. These result in filtered dataframes with all characteristic parameters that are then visualized for all locations through time.

.. _4TU repository: https://github.com/christavanijzendoorn/JAT

4. Dune toe analysis
====================
The Jarkus Analysis Toolbox was developed during the research that led to the publication of Van IJzendoorn et al. (2021) [#IJz]_;. This example shows how the toolbox was used for the dune toe analysis and provides the scripts that produce the figures that are included in the paper.

* dunetoe_transect_figure.py - Figure 1
* dunetoe_transect_map.py - Figure 1
* dunetoe_trend_figure.py - Figure 2 and Supl. Figure 1
* dunetoe_alongshore_figure.py - Figure 3
* sea_level_rise_figure.py - Figure 4

The mapping executed in the dunetoe_transect_map.py uses the package basemap which is dependent on a specific version of matplotlib and is therefore not compatible with the jarkus dependencies. Thus, it is best to create a new environment to run this script. This can be done by using the dune_transect_map.yml file which includes all the dependencies necessary to run the mapping script. Use the anaconda prompt and go to directory where environment file (dune_transect_map.yml) is located, use the followind commands::
$ conda env create -f dune_transect_map.yml
$ conda activate map
$ python dunetoe_transect_map.py

It should be noted that for the sea level rise figure, a specific dataset is used that can be found here. 

The Figures folder includes all figures for reference so you can check whether your output matches the expectations.

.. _here: https://www.clo.nl/indicatoren/nl022910-zeespiegelstand-nederland-en-mondiaal

.. [#IJz] Van IJzendoorn, C.O., De Vries, S., Hallin, C. & Hesp, P.A. (2021). Sea level outpaced by coastal dune toe translation. `In review`


5. Working with nc-file
=========================
Upcoming!!