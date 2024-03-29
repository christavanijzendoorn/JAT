Help
----------

New to Python
==============

If you are new to Python it might be that you are not able to follow the instructions in :doc:`Gettingstarted`. Here, a recommened set-up is presented, however, there are many different options and everyone has different needs and preferences so this is just a suggestion.

First of all, install Anaconda! You can find an installation guide here: https://docs.anaconda.com/anaconda/install/windows/.

As you may or may not know Python works with packages. These packages provide certain functionalities, for instance, installing the package matplotlib provides a library of functions that help to create graphs and plots. Often a package is dependent on other packages to be able to work. For instance, the package numpy provides scientific computing with Python and you can imagine that you will need this package for matplotlib to function. The packages that are needed are called the `dependencies`. The JAT has several specific dependencies which are listed in :doc:`Gettingstarted`. 

It might be that the specific packages that you need for the JAT do not go together with another project that you are working on that, for instance, works with a newer version of matplotlib. This can occur because the available packages for python are constantly evolving. To avoid these type of incompatibilities Anaconda works with environments. In this case, it is recommended to make a jarkus specific environment (for instance with the name `jarkus`). When you have created the environment make sure that you use python 3.7 because the JAT is not compatible with version 3.8. The best guide to follow is provided here: https://docs.anaconda.com/anaconda/navigator/getting-started/. 

The explanation on this web page relates to the Anaconda Navigator which is a nice visual way of looking at the environments. However, what you will often see is that people install packages using conda in the Anaconda prompt. That is also what we will need to do to install the JAT as explained in :doc:`Gettingstarted`:.

You can find information on managing environments in the Anaconda prompt here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html. But don't get too lost in this! The only things you need to do is open the Anaconda Prompt and type the following::

	$  activate jarkus
	$  cd C:/JAT/setup.py
	$  python setup.py install

First, this activate the environment that you just created (in this case `jarkus`). Note, that the dollar sign represents the blinking cursor in the Anaconda Prompt. Subsequently, the 'cd' command indicates that you want to change the folder that is open in the Anaconda prompt. When typing the 'cd' command follow it with the directory (in this case C:\\JAT\\setup.py) where the `setup.py` file is located. For the JAT this directory is dependent on where you have saved the JAT files, but it should end in ...\\JAT\\setup.py.

These commands will install the Jarkus Analysis Toolbox and all its dependencies so it works in one go.
	
To work with the examples, Spyder is recommended. To make sure it is installed open the Anaconda Navigator and browse to the jarkus environment (Applications on ...). Here, install Spyder and then launch it. In Spyder load, from example 1, `jarkus_01.yml` to see what settings are used and then open `JAT_use_single_transect.py` to see the steps necessary for the analysis. Spyder's walkthrough to get a feeling for how it works is recommended: https://docs.spyder-ide.org/current/index.html.

Often used short-cuts and features are:
	* F5 for running all the code in one script (green play button)
	* crtl+Return for running current sections denoted by `%##` statements (green play button in yellow/white marking)
	* F9 for running one line of code or selected code
	* The variable explorer
	
This is the point where you can start using the :doc:`Gettingstarted` and :doc:`Examples` sections again. If you run into other issues, try using Google/Stack Overflow because the internet knows a lot and other people often have experienced the same issues that you are running into.

Contact
========
If you find problems in the code of the Jarkus Analysis Toolbox please create an `issue on Github`_. There, you can also ask questions, indicate that you want to contribute to the code or share ideas on the improvement and application of the JAT.

.. _issue on Github: https://github.com/christavanijzendoorn/JAT/issues