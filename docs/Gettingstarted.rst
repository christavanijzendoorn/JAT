Getting Started
================

Installation 
-----------------------------
Download the JAT from https://github.com/christavanijzendoorn/JAT.git and save the JAT to a convenient location on your computer.

Or use git and navigate to a convenient location and clone the repository::

  $ git clone https://github.com/christavanijzendoorn/JAT.git

Open anaconda prompt and activate the environment you created or want to use (are you not able to follow? Go to :doc:`Help`). The JAT requires Python 3.7 and is not compatible with Python 3.8, so make sure to use the right version in your environment.

Navigate to the directory where the Jarkus Ananlysis Toolbox is located and the `setup.py` file is present. Use the following command to install the JAT::
 
  $  python setup.py install

Using the JAT
--------------
To use the JAT you will need to create two files (the names are suggestions based on the provided :doc:`Examples`:
	1. `jarkus.yml`
	2. `JAT_use.py`

The `jarkus.yml` file contains all the settings that are used to analyse the jarkus data. 

These settings include:
	* **years and transects** - Fill in the requested years and transects 
	* **inputdir** - Fill in where the input data is stored 
	* **outputdir** - Fill in where you want to store the JAT output
	* **data locations** - Specify the name of the input files or specify their online location
	* **save locations** -  Specify the names of the folders in which the JAT output is saved
	* **user defined** -  Specify the user defined values
	* **dimensions**:
		* **setting** - Specify the characteristic parameters that should be extracted
		* **variables** - No action needed, this is included to create a list of the requested parameters
		
The functionalities that you can use in the `JAT_use.py` file are explained in the :doc:`Functionalities` section. The best way to get an introduction into these functionalities is by using the :doc:`Examples`. These examples provide information on how to prepare transects, extract dimensions from these transects and show how to filter, analyse and visualize the extracted dimensions. 
Do not forget to change the directory of the `jarkus.yml` file in `JAT_use.py`.

Below you can find information that helps to understand (how to fill in) the settings in the `jarkus.yml` file.

Jarkus transect numbers
-----------------------------

To be able to decide what transects you want to analyse with the JAT, you need to know the way in which the transects are numbered.
The convention that is used in the JAT is as follows::

Vaknummer + raainummer = VNNNNNN:
	* always 6 transect (raai) related numbers
	* 1 or 2 coastal section (kustvak) related numbers, 2 in case of kustvak of 10+

Example Sand Engine: Vak 9, raai 11109 = 9011109
Example Meijendel: Vak 8, raai 9325 = 8009325
Example Westenschouwen: Vak 13, raai 1465 = 13001465


To check which transects are present in the area you want to analyse use the following sources:
	* Overview of transects: https://maps.rijkswaterstaat.nl/geoweb55/index.html?viewer=Kustlijnkaart 
	* Overview of transects and ‘kustvakken’: https://puc.overheid.nl/rijkswaterstaat/doc/PUC_629858_31/

	
In the `jarkus.yml` file you can choose how many transects you want to analyse. First, you choose the type of analysis:
	* **single** - analyse just one transect
	* **multiple** - analyse a selection of tansects, these do not have to be next to each other spatially
	* **range** - analyse transects between certain transect numbers. Especially around the boundaries of kustvakken, make sure to check whether the transects you want are indeed in increasing order
	* **all** - analyse al available transect in the Jarkus dataset

In all cases, the JAT will automatically filter transect numbers that do not exist.


Input files
--------------

**Jarkus**

The Jarkus Analysis Toolbox was developed to make the analysis of the Jarkus dataset more accessible.
To work with the JAT, the Jarkus data has to be accessed through this `link`_.

.. _link: https://opendap.deltares.nl/thredds/fileServer/opendap/rijkswaterstaat/jarkus/profiles/transect.nc

When you want to access large amounts of data (i.e. many transects and years) or want to be independent of internet access it is advisable to download the dataset (approx. 3 GB). Make sure to include their directory in the settings file (`jarkus.yml`).

**Dunetoe**

When you want to work with the *dune toes* that were extracted using the second derivative method. These can be found `here`_.

.. _here: https://opendap.deltares.nl/thredds/fileServer/opendap/rijkswaterstaat/DuneFoot/DF.nc

**Nourishment**

`This`_ is where the *nourishment* database can be found.

.. _This: https://opendap.tudelft.nl/thredds/dodsC/data2/deltares/rijkswaterstaat/suppleties/nourishments.nc.html

**LocFilter**

The `location_filter.yml` file is used to remove transects that contain, for instance, dams and dikes. It is used in Example 4 with :py:mod:`JAT.Filtering_functions.locations_filter`.
This file can be rewritten and used with the :py:mod:`JAT.Filtering_functions.locations_filter` to do other types of filtering.

**Titles**

This file is used to automatically create figures that show the distribution through time and space of all available characteristic parameters, see Example 3.

User-defined settings
----------------------
Below you can find a list of all user-defined settings that are included in the `jarkus.yml` file. For each setting a link to the documentation of the corresponding function is provided which explains how the setting is used.

	* filter1: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Transects.save_elevation_dataframes`
	* filter2: :py:mod:`JAT.Filtering_functions.availability_locations_filter`
	* primary dune: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_primary_dune_top`
	* secondary dune: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_secondary_dune_top`
	* mean sea level: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_sea_level`
	* mean high water: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_high_water_fixed`
	* mean low water: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_mean_low_water_fixed`
	* landward variance threshold: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_variance`
	* landward derivative: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_derivative`
	* landward bma: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_landward_point_bma`
	* seaward foreshore: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_seaward_point_foreshore`
	* seaward active profile: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_seaward_point_activeprofile`
	* seaward DoC: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_seaward_point_doc`
	* dune toe fixed: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_fixed`
	* dune toe classifier: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.get_dune_toe_derivative`
	* normalization: :py:mod:`JAT.Jarkus_Analysis_Toolbox.Extraction.normalize_dimensions(`
   
   
Dependencies
---------------
The JAT has specific dependencies that are managed through the `setup.py` file, the packages needed are as follows::

* numpy =1.17.2
* pandas = 0.25.1
* netCDF4
* scipy = 1.3.1
* matplotlib
* cftime = 1.0.3.4
* joblib = 0.13.2
* pybeach

License
---------

The JAT is free software made available under the GPL-3.0 License. For details see the license file_.

.. _file: https://github.com/christavanijzendoorn/JAT/blob/master/LICENSE.txt
