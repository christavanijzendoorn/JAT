Method
================

.. image:: ./_static/flowchart.png
..

    | Flowchart of Jarkus Analysis Toolbox functionalities.

The Jarkus Analysis Toolbox helps to analyse the Jarkus dataset. This dataset is stored on an online repository and made available by Rijkswaterstaat and Deltares.

.. _online repository: https://opendap.deltares.nl/thredds/fileServer/opendap/rijkswaterstaat/jarkus/profiles/transect.nc

Based on user input the necessary data is retrieved from this dataset by the JAT for certain years and locations. The JAT contains the option to save the elevation information of the requested coastal transects and to create a quickplot that shows all measured years per requested transect.

The core of the JAT is in the parameter extraction. This means that characteristic parameters are extracted from the elevation profile of each requested coastal transect. User input determines which characteristic parameters are extracted and it is expected that more extraction methods will be added to the JAT in the future. A guide on how to add a new method is provided in the :doc:`Development` section.

The raw output parameters that were extracted by using the currently available methods are made avaiable through `4TU repository`_. 

.. _4TU repository: https://doi.org/10.4121/14514213

Within the examples provided along with the JAT there are examples of filtering and visualisation that can be executed based on the raw output parameters. These examples provide suggestions which help to kick-start further analysis, but this is where the user can apply their own methods.

