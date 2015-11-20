
# Configuring Detex
This section is intended to introduce the detex logger and how to check the version of detex, 

## The log
By default detex creates a log of nearly everything it does. The log has a few components to it so let's simply create on to see what it looks like. 

First, we import detex


```python
import detex
```

Notice that, in your current working directory, a new file has been created called detex_log.log. Let's read that file and see what is inside.


```python
with open('detex_log.log', 'r') as log:
    for line in log:
        print line
```

    2015-11-19 09:05:42,526	detex	INFO	Imported Detex
    
    2015-11-19 09:13:59,818	detex	INFO	Imported Detex
    



```python

```

For each logged event, that is each entry in the log, there are four fields separated by tabs ("\t"). They are:
* The date and time the entry was made
* The module that made the entry
* The level of the log, they are:
    1. info - nothing is wrong, detex is simply letting you know what it is doing.
    1. debug - there might be some minor problems, info is logged to help you figure it out.
    1. warning - something went wrong but detex can press forward. This happens, for example when detex finds corrupted data; it will issue a warning and try to move on to the next usable data block.
    1. critical - something is very wrong but detex will try to keep going (not used very often).
    1. error - detex encountered an unrecoverable error. At this point an exception is raised and everything stops.
* The message passed to the logger

Many times the logger will also print to the screen in addition to making an entry in the log file. If you prefer your screen not to be cluttered but detex you can silence it by setting its verbosity to False.


```python
detex.verbose = False
```

Detex also has a max file size the log can be. If it finds a log that has exceeded this size it will simply delete it and create a new one. Ideally, a [rotating file handler](https://docs.python.org/2/library/logging.handlers.html#rotatingfilehandler) will be used in the future to keep the log under the specified limit without deleting the whole log but as of version 1.0.5 it has not been implimented. The log size limit is controlled by the maxSize parameter which, by default is 10 mb. If you don't want detex to make a log set the makeLog parameter to False.


```python
detex.makeLog = False
```

In most cases, however, we do want detex to produce a log and we want it to print to screen


```python
detex.makeLog = True
detex.verbose = True
```

Another useful way to read a log, and process it in a more systemic way is to use a function in the detex utilities called readLog, which will load the log into a pandas DataFrame (if you don't know what that is stop and do the [10 minute tutorial](http://pandas.pydata.org/pandas-docs/stable/10min.html) on pandas). Once in a DataFrame you could search the log certain times, levels, or strings in the messages. Of course, you could do this with commands like grep from the command line as well, but the less you have to leave python the better.  




```python
log = detex.util.readLog()
log
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
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
      <td> 2015-11-19 09:05:42,526</td>
      <td> detex</td>
      <td> INFO</td>
      <td> Imported Detex</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 2015-11-19 09:13:59,818</td>
      <td> detex</td>
      <td> INFO</td>
      <td> Imported Detex</td>
    </tr>
  </tbody>
</table>
</div>



## Version
You can check the detex version with the version attribute of detex


```python
detex.version
```




    '1.0.3b'



# Next Section
The next section covers the (getdata module)[../GetData/get_data.md]
