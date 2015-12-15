
# Required Files

Detex requires two files: the station key and the template key (referred to as "keys"). Both are usually saved as csv files for ease of editing/viewing.

## Station Key

The station key is used to tell detex which stations and channels to use. The location of the station and a desired time frame are also included. 

Here is an example from the introductory tutorial:


| NETWORK | STATION | STARTTIME | ENDTIME | LAT | LON | ELEVATION | CHANNELS |
|:-------:|:-------:|:---------:| :-----: | :-: | :-: | :-------: | :------: |
| TA | M18A	| 2009-04-01T00:00:00 | 2009-04-04T00:00:00	| 41.4272 | -110.0674 | 2103 | BHE-BHN-BHZ |
| TA | M17A	| 2009-04-01T00:00:00 | 2009-04-04T00:00:00	| 41.4729 | - 110.6664 | 2101 | BHE-BHN-BHZ |


The STARTTIME and ENDTIME fields indicate the time range of the continuous data and can be in any format readable by the [obspy.UTCDateTime class](http://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html) including a time stamp (ie epoch time). If you do use a time stamp be careful when editing outside of python because some programs, like excel, tend to silently round large numbers. See the [obspy.UTCDateTime docs](https://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html) for more info on readable formats. 

The CHANNELS field lists the channels that will be used for each station. If multiple channels are used they are separated by a dash (-).

The LAT, LON, and ELEVATION fields give the stations location in global coordinates (elevation is from sea-level, in meters).

The order of the headers is not important. Extra fields can be added without affecting Detex's ability to read the file. If you need to keep track of location for example, simply add a location field.

## Template Key

The template key is usually saved as TemplateKey.csv. It contains information on each of the events that will be used by detex. 

Here is an example from the introductory tutorial:

| CONTRIBUTOR | NAME | TIME | LAT | LON | DEPTH | MTYPE | MAG |
| :---------: | :--: | :--: | :-: | :-: |:----: | :---: |:--: |
| ANF | 2007-12-19T17-56-18 | 2007-12-19T17-56-18 | 41.7205	| -110.6486	| 4.07 | ML | 2.36 |
| ANF | 2007-12-21T18-30-09	| 2007-12-21T18-30-09 | 41.7669	| -110.6122	| 8.97 | ML | 2.17 |
| ANF | 2007-12-21T18-30-09	| 2007-12-21T18-30-09 | 41.7669	| -110.6122	| 8.97 | ML	| 2.17 |

The NAME field can be any string that can also be used as a file name by your OS. Windows does not allow ":" in a file path so the ":" between the hour and minute, and between the minute and seconds, have been replaced with a "-".

The TIME field, just like the STARTTIME and ENDTIME fields in the station key, can be in any obspy UTCDateTime readable format. 

The MAG field is used in estimating magnitudes of newly detected events. 

The LAT, LON, and DEPTH fields are used in some visualization methods. 

The CONTRIBUTOR and MTYPE fields are not required by detex but can be useful for record keeping. Additionally, just as with the station key, any extra fields can be added in any order. 

# Generating Keys

As long as the comma separated format shown above is followed you can use any method you like to create the keys. For small data sets it may be suitable to create the keys by hand in a text editor or in a program like open office. For larger data sets, however, it is better to either use some of the built in functions to generate the keys or create your own script to do so. 

The following shows a few of the built in methods for generating the keys but it is an good learning exercise in python, especially for those new in the language, to generate these files yourself. If you do write a script or function that uses some data source detex currently cannot read consider contributing it to detex as others will probably find it useful.

The following examples follows the [obspy FDSN tutorial](https://docs.obspy.org/packages/obspy.fdsn.html) closely. 



## Generating Station Keys

The format for the station key is very similar to that produced by the [IRIS station query](https://ds.iris.edu/SeismiQuery/station.htm). If you elect to have the results emailed to you it becomes a trivial to make a station key from the data in the email. Currently there is only one method to make the station key which uses an instance of the [obspy Inventory class](https://docs.obspy.org/packages/autogen/obspy.station.inventory.Inventory.html) as an input argument. 

### Station key from obspy inventory object



```python
import detex
import obspy
from obspy.fdsn import Client
import obspy

client = Client("IRIS") # use IRIS client

starttime = obspy.UTCDateTime('2009-01-01')
endtime = obspy.UTCDateTime('2010-01-01')

lat = 41.4272
lon = -110.0674

inv = client.get_stations(network="TA", starttime=starttime, endtime=endtime, 
                          channel='BH*', latitude=lat, longitude=lon, maxradius=1,
                         level='channel')

```


```python
stakey = detex.util.inventory2StationKey(inv, starttime, endtime)
stakey
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NETWORK</th>
      <th>STATION</th>
      <th>STARTTIME</th>
      <th>ENDTIME</th>
      <th>LAT</th>
      <th>LON</th>
      <th>ELEVATION</th>
      <th>CHANNELS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TA</td>
      <td>L17A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>42.0995</td>
      <td>-110.8727</td>
      <td>1996.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TA</td>
      <td>L18A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>41.9243</td>
      <td>-110.0364</td>
      <td>2051.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TA</td>
      <td>L19A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>42.1012</td>
      <td>-109.3575</td>
      <td>2034.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TA</td>
      <td>M17A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>41.4729</td>
      <td>-110.6664</td>
      <td>2101.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TA</td>
      <td>M18A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>41.4272</td>
      <td>-110.0674</td>
      <td>2103.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TA</td>
      <td>M19A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>41.5047</td>
      <td>-109.1569</td>
      <td>2080.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TA</td>
      <td>N17A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>40.9425</td>
      <td>-110.8335</td>
      <td>2500.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TA</td>
      <td>N18A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>40.9763</td>
      <td>-109.6731</td>
      <td>1893.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TA</td>
      <td>N19A</td>
      <td>2009-01-01T00-00-00</td>
      <td>2010-01-01T00-00-00</td>
      <td>40.8936</td>
      <td>-109.1772</td>
      <td>1703.0</td>
      <td>BHE-BHN-BHZ</td>
    </tr>
  </tbody>
</table>
</div>



## Generating Template Keys

There are two methods for generating template keys. The first uses an obspy catalog object as input and the second uses the output from the University of Utah Seismograph Stations (UUSS) code EQsearch. 

### Template key from obspy catalog object


```python
cat = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=2.5, catalog='ANF', 
                        latitude=lat, longitude=lon, maxradius=1)
```

to use this catalog as a template key we simply need to call the catalog2TemplateKey function of detex.util


```python
temkey = detex.util.catalog2Templatekey(cat) # get template key as DataFrame
temkey.to_csv('TemplateKey.csv', index=False) # save as csv
```


```python
temkey
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>TIME</th>
      <th>LAT</th>
      <th>LON</th>
      <th>DEPTH</th>
      <th>MAG</th>
      <th>MTYPE</th>
      <th>CONTRIBUTOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-11-23T19-14-08</td>
      <td>2009-11-23T19-14-08.060</td>
      <td>41.5691</td>
      <td>-108.8264</td>
      <td>11.1</td>
      <td>2.5</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-09-08T17-55-04</td>
      <td>2009-09-08T17-55-04.190</td>
      <td>41.4931</td>
      <td>-108.803</td>
      <td>1.3</td>
      <td>2.6</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-06-01T17-22-47</td>
      <td>2009-06-01T17-22-47.000</td>
      <td>41.609</td>
      <td>-108.774</td>
      <td>0.4</td>
      <td>3.0</td>
      <td>ML</td>
      <td>PDE-Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009-05-05T13-40-43</td>
      <td>2009-05-05T13-40-43.940</td>
      <td>41.588</td>
      <td>-109.3144</td>
      <td>26.0</td>
      <td>3.0</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009-04-20T17-53-38</td>
      <td>2009-04-20T17-53-38.440</td>
      <td>41.6998</td>
      <td>-110.6275</td>
      <td>3.0</td>
      <td>2.5</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2009-04-17T18-35-23</td>
      <td>2009-04-17T18-35-23.070</td>
      <td>41.7977</td>
      <td>-110.6184</td>
      <td>8.9</td>
      <td>2.5</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2009-04-16T18-23-21</td>
      <td>2009-04-16T18-23-21.320</td>
      <td>41.6856</td>
      <td>-110.6516</td>
      <td>7.9</td>
      <td>2.6</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2009-04-14T16-56-38</td>
      <td>2009-04-14T16-56-38.140</td>
      <td>41.7042</td>
      <td>-110.6316</td>
      <td>9.0</td>
      <td>2.6</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2009-04-13T17-28-07</td>
      <td>2009-04-13T17-28-07.410</td>
      <td>41.721</td>
      <td>-110.687</td>
      <td>0.0</td>
      <td>2.5</td>
      <td>ML</td>
      <td>NEIC</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2009-04-08T17-54-04</td>
      <td>2009-04-08T17-54-04.030</td>
      <td>41.1525</td>
      <td>-109.553</td>
      <td>10.0</td>
      <td>2.5</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2009-04-01T17-36-58</td>
      <td>2009-04-01T17-36-58.980</td>
      <td>41.6824</td>
      <td>-110.6313</td>
      <td>2.1</td>
      <td>2.5</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2009-03-19T19-06-07</td>
      <td>2009-03-19T19-06-07.760</td>
      <td>41.8112</td>
      <td>-110.6174</td>
      <td>8.2</td>
      <td>2.8</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2009-03-13T17-39-47</td>
      <td>2009-03-13T17-39-47.460</td>
      <td>41.8099</td>
      <td>-110.6039</td>
      <td>1.3</td>
      <td>2.5</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2009-03-07T02-45-10</td>
      <td>2009-03-07T02-45-10.000</td>
      <td>41.67</td>
      <td>-109.92</td>
      <td>4.5</td>
      <td>3.4</td>
      <td>ML</td>
      <td>PDE-Q</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2009-02-19T16-53-08</td>
      <td>2009-02-19T16-53-08.740</td>
      <td>41.506</td>
      <td>-108.8102</td>
      <td>6.4</td>
      <td>2.5</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2009-02-10T19-08-26</td>
      <td>2009-02-10T19-08-26.700</td>
      <td>41.7022</td>
      <td>-110.6354</td>
      <td>8.9</td>
      <td>2.7</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2009-01-29T18-02-07</td>
      <td>2009-01-29T18-02-07.560</td>
      <td>41.6945</td>
      <td>-110.6229</td>
      <td>8.8</td>
      <td>2.6</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2009-01-26T21-11-20</td>
      <td>2009-01-26T21-11-20.420</td>
      <td>41.704</td>
      <td>-110.613</td>
      <td>0.0</td>
      <td>2.6</td>
      <td>ML</td>
      <td>NEIC</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2009-01-23T21-59-33</td>
      <td>2009-01-23T21-59-33.000</td>
      <td>41.669</td>
      <td>-108.822</td>
      <td>1.1</td>
      <td>3.1</td>
      <td>ML</td>
      <td>PDE-Q</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2009-01-23T20-47-06</td>
      <td>2009-01-23T20-47-06.180</td>
      <td>41.8063</td>
      <td>-110.607</td>
      <td>11.4</td>
      <td>2.5</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2009-01-16T18-17-56</td>
      <td>2009-01-16T18-17-56.550</td>
      <td>41.809</td>
      <td>-110.594</td>
      <td>9.1</td>
      <td>2.7</td>
      <td>ML</td>
      <td>ANF</td>
    </tr>
  </tbody>
</table>
</div>



The other function used to make a template key is only useful at the University of Utah where a program called EQsearch is used to query the UUSS catalog. EQsearch produces a file, by default, called eqsrchsum. The function EQSearch2TemplateKey is a parser that takes the information from this file and converts it to a template key.

# Reading Keys

All detex functions and classes that use a key file call the detex.util.readKey function to read in the key file (either template key, station key, or phase picks) or to validate a key that is already in memory (in the form of a DataFrame). This function makes sure all the required fields exist and have legal values. If you want to verify that a key file you have created is valid simply try and read it in with the readKey function. Alternatively, we can pass a DataFrame to the function to see if it is a valid key. 


```python
temkey2 = detex.util.readKey(temkey, key_type='template')
stakey2 = detex.util.readKey(stakey, key_type='station')
```

Since no errors were raised the station key and the template key we created are valid.
