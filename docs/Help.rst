Help
----------


New to Python
==============

If you are new to Python it might be that you are not able to follow the instructions in :doc:`Gettingstarted`. To help you I will tell about the set-up I use to work with Python. Note, there are many different options and everyone has different needs and preferences so this is just a suggestion.

First of all, install Anaconda! You can find an installation guide here: https://docs.anaconda.com/anaconda/install/windows/.

As you may or may not know Python works with packages. These packages provide certain functionalities, for instance, installing the package matplotlib provides a library of functions that help to create graphs and plots. Often a package is dependent on other packages to be able to work. For instance, the package numpy provides scientific computing with Python and you can imagine that you will need this package for matplotlib to function. The packages that are needed are called the `dependencies`. The JAT has several specific dependencies which are listed in :doc:`Gettingstarted`:. 

It might be that the specific packages that you need for the JAT do not go together with another project that you are working on that, for instance, works with a newer version of matplotlib. This can occur because the available packages for python are constantly evolving. To avoid these type of incompatibilities Anaconda works with environments. In this case, I advice you to make a jarkus specific environment (I called it `jarkus`). When you have created the environment make sure that you use python 3.7 because the JAT is not compatible with version 3.8. The best guide to follow is provided here: https://docs.anaconda.com/anaconda/navigator/getting-started/. 

The explanation on this web page relates to the Anaconda Navigator which is a nice visual way of looking at the environments. However, what you will often see is that people install packages using conda in the Anaconda prompt. That is also what we will need to do to install the JAT as explained in :doc:`Gettingstarted`:.

You can find information on managing environments in the Anaconda prompt here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html. But don't get too lost in this! The only things you need to do is open the Anaconda Prompt and type the following::

	$  activate jarkus
	$  cd C:/JAT/setup.py
	$  python setup.py install

First, this activate the environment that you just created (in this case `jarkus`). Note, that the dollar sign represents the blinking cursor in the Anaconda Prompt. Subsequently, the 'cd' command indicates that you want to change the folder that is open in the Anaconda prompt. When typing the 'cd' command follow it with the directory (in this case C:\\JAT\\setup.py) where the `setup.py` file is located. For the JAT this directory is dependent on where you have saved the JAT files, but it should end in ...\\JAT\\setup.py.

These commands will install the Jarkus Analysis Toolbox and all its dependencies so it works in one go.
	
To work with the examples, I recommend Spyder. To make sure it is installed open the Anaconda Navigator and browse to the jarkus environment (Applications on ...). Here, install Spyder and then launch it. In Spyder load, from example 1, `jarkus_01.yml` to see what settings are used and then open `JAT_use_single_transect.py` to see the steps necessary for the analysis. I recommend Spyder's walkthrough to get a feeling for how it works: https://docs.spyder-ide.org/current/index.html.

What I use most often:
	* F5 for running all the code in one script (green play button)
	* crtl+Return for running current sections denoted by `%##` statements (green play button in yellow/white marking)
	* F9 for running selected code
	
This is the point where you can start using the :doc:`Gettingstarted` and :doc:`Examples` sections again. Hopefully, this helped and I always recommend using Google/Stack Overflow because the internet knows a lot and other people often have experienced the same issues that you are running into.


Adding new extraction methods
==============================

* Add the extraction method in the `Jarkus_analysis_toolbox.py` in class `Extraction` as `def get_parameter` (fill in appropriate name `for parameter`). It is best to use the other available extraction methods as inspiration so the method is comparable. For the assigment of the output into the dimensions dataframe use an appropriate variable name.
* Add an `if statement` for the extraction of the parameter in `get_all_dimensions` of `Extraction` class in `Jarkus_analysis_toolbox.py`
* Add the parameter name in the configuration file so the `if statement` (reference above) will work. This means the parameter name should be added in both dimensions→setting→variablename with a True/False statement, and in dimensions→variables→ variablename with the variable names as assigned in the `get_parameter` function, in the jarkus.yml file.
* Add these assigned variable names to the plot_tiles.yml file, so the distribution figures can be generated automatically. Note, that for cross-shore variables it is crucial to add an 'x' in the variable name, so it is picked up in the automatic normalization (Extraction → normalize_dimensions). Then, add a suitable title and label name for in the figure.
* Make sure to update the documentation!