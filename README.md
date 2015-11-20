# Introduction

Detex is a python package for performing waveform similarity clustering and [subspace detection](https://e-reports-ext.llnl.gov/pdf/335299.pdf) on seismic data. The main goal of such analyses is to determine if a group of seismic events are similar, meaning they have comparable source mechanisms and hypocenters, and to use such information to identify small, often difficult to detect, earthquakes.

Detex is written in python (currently only tested on 2.7) and relies heavily on Obspy, Numpy, Scipy, Matplotlib, and Pandas. 

Special thanks to Tex Kubacki (whose work inspired Detex), Jared Stein, Kris Pandow, Lisa Linvile, Shawn Blotz, Chase Batchelor and the many other faculty and students of the University of Utah that have help develop and test detex.

## Tutorial 

There are two tutorials avaliable:

[The introductory tutorial](ReadMe/intro.md) - Serves to illustrate what detex does on a high level by introducing a suggested workflow. I recommend you start here. 

[The advanced tutorial](ReadMe/advanced.md) - Provides details on the important classes in detex and highlights key features. Apart from the doc strings, this tutorial serves as the main form of documentation for detex. 

## Contributing to detex

If you would like to contribute to detex your help is much appreciated. This could include adding/requesting a feature, fixing/reporting a bug, or simply asking a question. [Here](ReadMe/ContributeToDetex/contributing.md) is a basic guide to doing so, everything should be fairly standard. 

# References

The following publication provides additional details on the methods employed by detex :

Chambers, D. J., Koper, K. D., Pankow, K. L., & McCarter, M. K. (2015). Detecting and characterizing coal mine related seismicity in the Western US using subspace methods. Geophysical Journal International, 203(2), 1388-1399. doi: 10.1093/gji/ggv383

If you use detex in your research please consider citing it. You can access the article [here](http://gji.oxfordjournals.org/content/203/2/1388.full?keytype=ref&ijkey=5HUaTUw3o0Xikhs). 

# Recent changes
The [changelog](ChangeLog.txt) should be up-to-date with major changes made to detex


