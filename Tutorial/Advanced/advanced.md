# Advanced Tutorial

Welcome to the advanced tutorial for detex. At this point you should:

1. have a working knowledge of python

2. have some experience with the python package pandas or have gone through the [10 minute pandas tutorial](http://pandas.pydata.org/pandas-docs/stable/10min.html)

3. have completed (or at least looked through) the [introductory tutorial](intro.md)

4. have read, or skimmed through, [Dave Harris' subspace paper](https://e-reports-ext.llnl.gov/pdf/335299.pdf)

5. have read, or skimmed through, [my subspace paper](http://gji.oxfordjournals.org/content/203/2/1388.full?keytype=ref&ijkey=5HUaTUw3o0Xikhs) which explains what detex is doing under the hood

The purpose of this tutorial is to guide you through the workflow of detex with much greater detail than the introductory tutorial. 

Note that each of the segments of this tutorial, as well as the introductory tutorial, where created using [ipython notebook](http://ipython.org/notebook.html). I have left the notebook files for each segment so that you can run them directly. 

# Detex Organization
First, a word about the organization of detex. There are several important modules and classes that merit some discussion. 

## The modules
Here are the modules in detex and a brief description of what they do as of version 1.0.4. More modules may be added in the future. 

* **Init** - called when importing detex. Imports the other modules explicitly and is used to start the logger. Controls several parameters for verbosity and log file length.

* **getdata** - home of the DataFetcher class which serves all data to other detex functions and classes. Also has functions to download data to create local directories and index the directories with an SQLite database. 

* **construct** - contains the functions for creating the cluster and subspace objects. Performs waveform correlation and alignment, as well as various data checks.

* **subspace** - home of the cluster classes (ClusterStream, Cluster) and subspace classes (SubSpace) which are the main objects used to control the waveform clustering and subspace detection.

* **fas** - module to handle the estimation of false alarm statistics. Is called from the SubSpace class to parse random continuous data and try to fit a distribution to the null space in order to set detection statistic thresholds. 

* **detect** - module called by SubSpace class to preform the subspace detection, magnitude estimation, etc. 

* **results** - used to associate potential detections together from various stations. Home of the SSResults class.

* **util** - module with various functions that can be useful in data prep, setting up files, visualization, etc. but are not part of the main work flow.

* **streamPick** - a light GUI used for making manual phase picks for both individual events and whole subspaces. The GUI was created by [miili](miili) and the original project can be found [here](https://github.com/miili/StreamPick). In order to use streamPick in the context of detex, however, it required several modification that would hinder its use in the application it was designed for, so the modified code included in detex.

* **pandasdbms** - a script for writing pandas DataFrames to SQLite databases. The original script can be found [here](https://gist.github.com/catawbasam/3164289).

* **qualitycheck** - A module still in the works for checking on quality of available data. Ideally it will inform the user of data gaps, missing responses, etc. and include some visualization methods. Currently it is not complete. 

## The classes
There are few important classes worth mentioning:

* **DataFetcher** - used to serve data to all other detex functions. Can get data from obspy clients (FDSN, NEIC, Earthworm, etc.) or local data directories. 

* **ClusterStream** - container for Cluster instances.

* **Cluster** - the waveform similarity for a single station. Contains the lag times, correlation coefficients, etc. 

* **SubSpace** - controller for handling subspace operations. Takes a ClusterStream as input. 

* **SSResults** - results class, used for associating detections together. 


# Detex workflow

## [Logging](Logging/logging.md)
Introduces logging and explains how to check the version number.

## [Required Files](RequiredFiles/required_files.md)
reviews the format of the two required files: the station key and the template key. Also highlights some methods for creating the keys (from station inventories and catalog objects for example). 

##  [Getdata](GetData/get_data.md)
Covers the various methods of getting seismic data in detex using the DataFetcher class. 


