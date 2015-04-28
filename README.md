# Detex
Detex is a python program that performs subspace detection and waveform correlation detection for seismic data sets. I developed the code to perform research for my MS degree but several others have also used it.
Detex relies heavily on [Obspy](https://github.com/obspy/obspy/wiki)
and [Pandas](http://pandas.pydata.org/)
Detex, has not been extensively tested. If you would like to help test/develop detex you are welcome. 

# Installation
To install detex, first make sure that you have a working distribution of python 2.7. I recommend the [Anaconda distribution] (http://continuum.io/downloads).
Next, clone (or download) Detex and run the setupy.py by typing into the terminal (or command line if using windows)
``` bash
Python setup.py install
```
The install script will make sure you have the required packages and install detex in the appropriate place. 
Note: If the process fails to install basemap, which it has been known to do, you can find manual installation instructions [here]( http://matplotlib.org/basemap/users/installing.html). Most of detex functionality does not require basemap so if you can’t get it working you can still use the majority of detex. 

# Tutorial
A tutorial can be found [here](http://d-chambers.github.io/Detex/), all of the required files are found in the tutorial directory.

