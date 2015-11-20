# Advanced Tutorial

Welcome to the advanced tutorial for detex. At this point you should:

1.  have a working knowledge of python
2. you have worked with the python package pandas before or have gone through the [10 minute pandas tutorial](http://pandas.pydata.org/pandas-docs/stable/10min.html)
3. have completed (or at least looked through) the [introductory tutorial](intro.md)
4. have read, or skimmed through, the [Harris subspace paper](https://e-reports-ext.llnl.gov/pdf/335299.pdf)
5. have read, or skimmed through, [my  subspace paper](http://gji.oxfordjournals.org/content/203/2/1388.full?keytype=ref&ijkey=5HUaTUw3o0Xikhs) which explains a bit more of the methods detex employs

Once you are ready this tutorial will guide you through the workflow of detex with much greater detail than the introductory tutorial. This document also serves as the main form of documentation for detex (other than the doc strings of course).

Note that each of the segments of this tutorial, in addition to the introductory tutorial where created using [ipython notebook](http://ipython.org/notebook.html). I have left the notebook files for each segment so that you can run them directly if you desire. 

# Detex workflow

## (Logging)[Advanced/Logging/configuring_detex.md]
Introduces logging and explains how to check the version number

## (Required Files)[Advanced/RequiredFiles/required_files.md]
reviews the format of the two required files: the station key and the template key. Also highlights some methods for creating the keys (from station inventories and catalog objects for example). 

##  (Getdata)[Advanced/GetData/get_data.md]
Covers the various methods of getting seismic data in detex. The focus is on the DataFetcher class and it's methods. 


