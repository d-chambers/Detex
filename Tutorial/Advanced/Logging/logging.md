
# Configuring Detex
This section is intended to introduce the detex logger and how to check the version of detex, 

## The log
Detex can create a log of nearly everything it does, but this feature needs to be enabled. 

First, we import detex


```python
import detex
```


```python
detex.setLogger()
```




    (<logging.Logger at 0xe3c6160>, 'C:\\Chambers\\Detex\\detex_log.log')



Notice that, in your current working directory, a new file has been created called detex_log.log. Alternatively we could pass a relative or absolute path to the setLogger call to make the file in a different location and call it by a different name, but we will stick with the default for now. 

For each logged event, that is each entry in the log, there are four fields separated by tabs ("\t"). They are:
* The date and time the entry was made
* The module that made the entry
* The level of the log, they are:
    1. info - nothing is wrong, detex is simply letting you know what it is doing.
    1. debug - there might be some minor problems, info is logged to help you figure it out.
    1. warning - something went wrong but detex can press forward. This happens, for example when detex finds corrupted data; it will issue a warning and try to move on to the next usable data block.
    1. critical - something is very wrong but detex will try to keep going (not used very often).
    1. error - detex encountered a major problem that cannot be ignored. At this point an exception is raised and everything stops.
* The message passed to the logger

Many times the logger will also print to the screen in addition to making an entry in the log file. If you prefer your screen not to be cluttered but detex you can silence it by setting the verbose varaible to False.


```python
detex.verbose = False
```

Detex also has a max file size the log can be. If it finds a log that has exceeded this size it will simply delete it and create a new one. Ideally, a [rotating file handler](https://docs.python.org/2/library/logging.handlers.html#rotatingfilehandler) will be used in the future to keep the log under the specified limit without deleting the whole log but as of version 1.0.4 it has not been implimented. The log size limit is controlled by the maxSize parameter which, by default is 10 mb. 

One way to read a log, and process it in a systemic way is to use a function in the detex utilities called readLog, which will load the log into a pandas DataFrame (if you don't know what that is stop and do the [10 minute tutorial](http://pandas.pydata.org/pandas-docs/stable/10min.html) on pandas). Once in a DataFrame you could search the log for certain times, levels, or strings in the messages. Of course, you could do this with commands like grep from a terminal as well, but the less you have to leave python the better.  




```python
log = detex.util.readLog()
log
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Mod</th>
      <th>Level</th>
      <th>Msg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-11-27 10:33:11,323</td>
      <td>detex</td>
      <td>INFO</td>
      <td>Starting logging, path to log file: C:\Chamber...</td>
    </tr>
  </tbody>
</table>
</div>



## Version
You can check the detex version with the version attribute of detex


```python
detex.__version__
```




    '1.0.4'



# Next Section
The next section covers the [getdata module](../GetData/get_data.md)
