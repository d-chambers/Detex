
# Clustering
This segment of the tutorial will teach you how to preform waveform similarity clustering in detex. The function used to preform clustering is createCluster of the construct module. The results are then stored in an instance of the ClusterStream class. Let's start by looking at the createCluster function and some important parameters that go into it, then we will look at the ClusterStream class its methods.

## CreateCluster



```python
import detex
version = detex.__version__
print ("Detex version is %s\n" % version)
print (detex.construct.createCluster.__doc__)
```

    Detex version is 1.0.5
    
     
        Function to create an instance of the ClusterStream class 
        
        Parameters
        -------
        CCreq : float, between 0 and 1
            The minimum correlation coefficient for grouping waveforms. 
            0.0 results in all waveforms grouping together and 1.0 will not
            form any groups.
        fetch_arg : str or detex.getdata.DataFetcher instance
            Fetch_arg of detex.getdata.quickFetch, see docs for details.
        filt : list
            A list of the required input parameters for the obspy bandpass 
            filter [freqmin, freqmax, corners, zerophase].
        stationKey : str or pd.DataFrame
            Path to the station key or DataFrame of station key.
        templateKey : str or pd.DataFrame
            Path to the template key or loaded template key in DataFrame.
        trim : list 
            A list with seconds to trim from events with respect to the origin 
            time reported in the template key. The default value of [10, 120] 
            means each event will be trimmed to only contain 10 seconds before 
            its origin time and 120 seconds after. The larger the values of this 
            argument the longer the computation time and chance of misalignment,
            but values that are too small may trim out desired phases of the
            waveform. 
        saveClust : bool
            If true save the cluster object in the current working 
            directory. The name is controlled by the fileName parameter. 
        fileName : str
            Path (or name) to save the clustering instance, only used 
            if saveClust is True.
        decimate : int or None
            A decimation factor to apply to all data in order to decrease run 
            time. Can be very useful if the the data are oversampled. For 
            example, if the data are sampled at 200 Hz but a 1 to 10 Hz 
            bandpass filter is applied it may be appropriate to apply a 
            decimation factor of 5 to bring the sampling rate down to 40 hz. 
        dytpe : str
            An option to recast data type of the seismic data. Options are:
                double- numpy float 64
                single- numpy float 32, much faster and amenable to cuda GPU 
                processing, sacrifices precision.
        eventsOnAllStations : bool
            If True only use the events that occur on all stations, if 
            false let each station have an independent event list.
        enforceOrigin : bool
            If True make sure each trace starts at the reported origin time in 
            the template key. If not trim or merge with zeros. Required  for 
            lag times to be meaningful for hypoDD input.
        fillZeros : bool
            If True fill zeros from trim[0] to trim[1]. Suggested for older 
            data or if only triggered data are available.
            
        Returns
        ---------
            An instance of the detex SSClustering class
        


As you can see there are a lot of input arguments, and a lot to think about when creating a cluster object. Let me elaborate on some of the arguments you should pay special attention to. 

* fet_arg - make sure to look at the detex.getdata.quickFetch docs for this one. Basically, if you want to use a custom DataFetcher be sure to pass it to the createCluster call here or else detex will try to use a local directory with the default name of EventWaveForms.

* filt - parameters to apply a bandpass filter to the waveform similarity clustering and ALL all detex downstream operations. Make sure to think about this carefully before simply using the default, as the default values are not appropriate for all data sets.

* fillZeros - a parameter for handling data with gaps. If data are not avaliable for the entire range (defined by template key and trim parameter) detex will simply fill zeros so that each trace will have the length defined by the trim parameter. The created cluster instance can then be used later on by detex, although you should be careful going forward to no include a bunch of the zero data in your detector, more on that later. 

* trim - a two element list that defines the length of each waveform. The first element is the time before the origin (as reported in the station key) and the second element is the number of seconds after the reported origin time. 

### Dealing with gaps

In order to see how some of these parameters affect the clustering process we will look at an early UUSS dataset that has some issues with gaps. Here are the stations and templates:




```python
stakey = detex.util.readKey('StationKey.csv', key_type='station')
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
      <td>UU</td>
      <td>IMU</td>
      <td>2003-01-03T00:00:00</td>
      <td>2004-04-04T00:00:00</td>
      <td>38.6332</td>
      <td>-113.158</td>
      <td>1833</td>
      <td>EHZ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>UU</td>
      <td>MSU</td>
      <td>2003-01-01T00:00:00</td>
      <td>2004-04-04T00:00:00</td>
      <td>38.5123</td>
      <td>-112.177</td>
      <td>2105</td>
      <td>EHZ</td>
    </tr>
  </tbody>
</table>
</div>




```python
temkey = detex.util.readKey('TemplateKey.csv', key_type='template')
temkey
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TIME</th>
      <th>NAME</th>
      <th>LAT</th>
      <th>LON</th>
      <th>MAG</th>
      <th>DEPTH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2002-02-01T04-37-29.94</td>
      <td>2002-02-01T04-37-29.94</td>
      <td>38.616833</td>
      <td>-112.463833</td>
      <td>1.40</td>
      <td>2.82</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002-02-02T03-38-36.21</td>
      <td>2002-02-02T03-38-36.21</td>
      <td>38.575667</td>
      <td>-112.714500</td>
      <td>1.19</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002-02-22T22-59-28.69</td>
      <td>2002-02-22T22-59-28.69</td>
      <td>38.558667</td>
      <td>-112.460500</td>
      <td>2.17</td>
      <td>1.72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2002-02-22T23-10-36.58</td>
      <td>2002-02-22T23-10-36.58</td>
      <td>38.538667</td>
      <td>-112.455500</td>
      <td>0.99</td>
      <td>5.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002-02-26T11-45-48.87</td>
      <td>2002-02-26T11-45-48.87</td>
      <td>38.527333</td>
      <td>-112.467667</td>
      <td>1.98</td>
      <td>4.01</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2002-02-26T12-09-47.92</td>
      <td>2002-02-26T12-09-47.92</td>
      <td>38.558500</td>
      <td>-112.456500</td>
      <td>2.39</td>
      <td>-0.28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2002-03-04T02-53-36.92</td>
      <td>2002-03-04T02-53-36.92</td>
      <td>38.708000</td>
      <td>-112.535500</td>
      <td>1.43</td>
      <td>-0.14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2002-03-04T17-13-19.61</td>
      <td>2002-03-04T17-13-19.61</td>
      <td>38.542333</td>
      <td>-112.455333</td>
      <td>1.33</td>
      <td>1.84</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2002-03-05T00-32-02.35</td>
      <td>2002-03-05T00-32-02.35</td>
      <td>38.544667</td>
      <td>-112.457667</td>
      <td>1.34</td>
      <td>2.80</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2002-03-10T16-03-15.42</td>
      <td>2002-03-10T16-03-15.42</td>
      <td>38.722000</td>
      <td>-112.545000</td>
      <td>2.50</td>
      <td>-2.97</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2002-03-21T20-16-14.74</td>
      <td>2002-03-21T20-16-14.74</td>
      <td>38.711333</td>
      <td>-112.541167</td>
      <td>2.04</td>
      <td>1.81</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2002-03-23T16-39-08.76</td>
      <td>2002-03-23T16-39-08.76</td>
      <td>38.701667</td>
      <td>-112.542333</td>
      <td>1.32</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2002-05-12T18-32-19.43</td>
      <td>2002-05-12T18-32-19.43</td>
      <td>38.724333</td>
      <td>-112.539500</td>
      <td>0.71</td>
      <td>1.42</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2002-05-15T15-35-00.17</td>
      <td>2002-05-15T15-35-00.17</td>
      <td>38.588667</td>
      <td>-112.586833</td>
      <td>0.73</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002-07-06T22-05-16.73</td>
      <td>2002-07-06T22-05-16.73</td>
      <td>38.512000</td>
      <td>-112.460833</td>
      <td>0.35</td>
      <td>5.78</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2002-08-01T20-19-21.00</td>
      <td>2002-08-01T20-19-21.00</td>
      <td>38.525833</td>
      <td>-112.455500</td>
      <td>1.03</td>
      <td>1.84</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2002-08-19T19-35-07.44</td>
      <td>2002-08-19T19-35-07.44</td>
      <td>38.689000</td>
      <td>-112.543500</td>
      <td>0.72</td>
      <td>5.96</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2002-08-19T21-09-35.83</td>
      <td>2002-08-19T21-09-35.83</td>
      <td>38.685833</td>
      <td>-112.541833</td>
      <td>1.17</td>
      <td>4.54</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2002-09-07T19-55-27.85</td>
      <td>2002-09-07T19-55-27.85</td>
      <td>38.713000</td>
      <td>-112.647500</td>
      <td>1.02</td>
      <td>2.79</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2002-09-17T14-50-53.10</td>
      <td>2002-09-17T14-50-53.10</td>
      <td>38.663167</td>
      <td>-112.630000</td>
      <td>0.71</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2002-10-06T11-57-35.19</td>
      <td>2002-10-06T11-57-35.19</td>
      <td>38.645667</td>
      <td>-112.461667</td>
      <td>1.31</td>
      <td>3.64</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2002-11-05T23-12-44.72</td>
      <td>2002-11-05T23-12-44.72</td>
      <td>38.532333</td>
      <td>-112.466167</td>
      <td>0.95</td>
      <td>6.02</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2002-11-05T23-13-27.56</td>
      <td>2002-11-05T23-13-27.56</td>
      <td>38.555167</td>
      <td>-112.468167</td>
      <td>1.37</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2002-11-06T00-44-00.38</td>
      <td>2002-11-06T00-44-00.38</td>
      <td>38.527833</td>
      <td>-112.482167</td>
      <td>2.05</td>
      <td>4.09</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2002-11-06T01-48-04.90</td>
      <td>2002-11-06T01-48-04.90</td>
      <td>38.532167</td>
      <td>-112.463833</td>
      <td>1.10</td>
      <td>3.15</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2002-11-06T02-24-03.05</td>
      <td>2002-11-06T02-24-03.05</td>
      <td>38.515833</td>
      <td>-112.463833</td>
      <td>0.61</td>
      <td>6.02</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2002-11-08T18-27-28.27</td>
      <td>2002-11-08T18-27-28.27</td>
      <td>38.534667</td>
      <td>-112.468500</td>
      <td>1.29</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2002-11-08T21-48-47.69</td>
      <td>2002-11-08T21-48-47.69</td>
      <td>38.530833</td>
      <td>-112.470500</td>
      <td>2.26</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2002-11-08T22-00-02.41</td>
      <td>2002-11-08T22-00-02.41</td>
      <td>38.519500</td>
      <td>-112.468000</td>
      <td>1.12</td>
      <td>2.07</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2002-11-09T21-38-33.05</td>
      <td>2002-11-09T21-38-33.05</td>
      <td>38.548167</td>
      <td>-112.468333</td>
      <td>1.22</td>
      <td>7.83</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>192</th>
      <td>2013-10-10T16-10-37.49</td>
      <td>2013-10-10T16-10-37.49</td>
      <td>38.597167</td>
      <td>-112.598167</td>
      <td>1.96</td>
      <td>5.21</td>
    </tr>
    <tr>
      <th>193</th>
      <td>2013-10-10T23-12-49.39</td>
      <td>2013-10-10T23-12-49.39</td>
      <td>38.606500</td>
      <td>-112.612833</td>
      <td>2.48</td>
      <td>3.32</td>
    </tr>
    <tr>
      <th>194</th>
      <td>2013-10-10T23-21-10.54</td>
      <td>2013-10-10T23-21-10.54</td>
      <td>38.601833</td>
      <td>-112.606333</td>
      <td>2.36</td>
      <td>3.91</td>
    </tr>
    <tr>
      <th>195</th>
      <td>2013-10-11T00-26-32.78</td>
      <td>2013-10-11T00-26-32.78</td>
      <td>38.603833</td>
      <td>-112.606833</td>
      <td>1.34</td>
      <td>-0.16</td>
    </tr>
    <tr>
      <th>196</th>
      <td>2013-10-11T10-36-01.05</td>
      <td>2013-10-11T10-36-01.05</td>
      <td>38.613667</td>
      <td>-112.607667</td>
      <td>0.86</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>197</th>
      <td>2013-10-11T15-19-47.68</td>
      <td>2013-10-11T15-19-47.68</td>
      <td>38.598167</td>
      <td>-112.608333</td>
      <td>1.96</td>
      <td>4.14</td>
    </tr>
    <tr>
      <th>198</th>
      <td>2013-10-24T09-57-46.49</td>
      <td>2013-10-24T09-57-46.49</td>
      <td>38.589167</td>
      <td>-112.606333</td>
      <td>1.54</td>
      <td>1.48</td>
    </tr>
    <tr>
      <th>199</th>
      <td>2013-11-01T02-56-52.97</td>
      <td>2013-11-01T02-56-52.97</td>
      <td>38.734500</td>
      <td>-112.567167</td>
      <td>1.31</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>200</th>
      <td>2013-11-10T15-36-35.26</td>
      <td>2013-11-10T15-36-35.26</td>
      <td>38.714667</td>
      <td>-112.526333</td>
      <td>0.70</td>
      <td>9.28</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2013-12-25T05-32-32.15</td>
      <td>2013-12-25T05-32-32.15</td>
      <td>38.633833</td>
      <td>-112.539500</td>
      <td>0.78</td>
      <td>-0.10</td>
    </tr>
    <tr>
      <th>202</th>
      <td>2014-01-09T00-10-03.16</td>
      <td>2014-01-09T00-10-03.16</td>
      <td>38.554833</td>
      <td>-112.728833</td>
      <td>1.93</td>
      <td>-0.24</td>
    </tr>
    <tr>
      <th>203</th>
      <td>2014-01-17T12-52-52.03</td>
      <td>2014-01-17T12-52-52.03</td>
      <td>38.637500</td>
      <td>-112.612000</td>
      <td>-9.99</td>
      <td>1.49</td>
    </tr>
    <tr>
      <th>204</th>
      <td>2014-01-17T12-52-55.73</td>
      <td>2014-01-17T12-52-55.73</td>
      <td>38.609667</td>
      <td>-112.613000</td>
      <td>1.84</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>205</th>
      <td>2014-01-17T16-16-23.71</td>
      <td>2014-01-17T16-16-23.71</td>
      <td>38.607667</td>
      <td>-112.603667</td>
      <td>0.96</td>
      <td>1.93</td>
    </tr>
    <tr>
      <th>206</th>
      <td>2014-02-08T07-14-59.47</td>
      <td>2014-02-08T07-14-59.47</td>
      <td>38.664500</td>
      <td>-112.535667</td>
      <td>0.86</td>
      <td>4.32</td>
    </tr>
    <tr>
      <th>207</th>
      <td>2014-02-10T00-32-14.92</td>
      <td>2014-02-10T00-32-14.92</td>
      <td>38.734333</td>
      <td>-112.521500</td>
      <td>1.24</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th>208</th>
      <td>2014-03-08T20-37-13.43</td>
      <td>2014-03-08T20-37-13.43</td>
      <td>38.722000</td>
      <td>-112.552667</td>
      <td>0.94</td>
      <td>4.96</td>
    </tr>
    <tr>
      <th>209</th>
      <td>2014-03-23T01-26-16.64</td>
      <td>2014-03-23T01-26-16.64</td>
      <td>38.674167</td>
      <td>-112.585833</td>
      <td>1.92</td>
      <td>-3.09</td>
    </tr>
    <tr>
      <th>210</th>
      <td>2014-04-06T08-19-55.40</td>
      <td>2014-04-06T08-19-55.40</td>
      <td>38.563000</td>
      <td>-112.690167</td>
      <td>1.92</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>211</th>
      <td>2014-05-27T19-04-32.17</td>
      <td>2014-05-27T19-04-32.17</td>
      <td>38.594833</td>
      <td>-112.593833</td>
      <td>1.20</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>212</th>
      <td>2014-06-05T03-14-06.76</td>
      <td>2014-06-05T03-14-06.76</td>
      <td>38.683667</td>
      <td>-112.533167</td>
      <td>2.01</td>
      <td>3.51</td>
    </tr>
    <tr>
      <th>213</th>
      <td>2014-06-09T11-00-38.18</td>
      <td>2014-06-09T11-00-38.18</td>
      <td>38.727000</td>
      <td>-112.503167</td>
      <td>0.57</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>214</th>
      <td>2014-06-12T06-03-10.42</td>
      <td>2014-06-12T06-03-10.42</td>
      <td>38.678167</td>
      <td>-112.539500</td>
      <td>1.70</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>215</th>
      <td>2014-07-11T03-47-35.36</td>
      <td>2014-07-11T03-47-35.36</td>
      <td>38.689333</td>
      <td>-112.579500</td>
      <td>0.78</td>
      <td>3.96</td>
    </tr>
    <tr>
      <th>216</th>
      <td>2014-10-29T10-58-38.16</td>
      <td>2014-10-29T10-58-38.16</td>
      <td>38.574833</td>
      <td>-112.587333</td>
      <td>1.64</td>
      <td>3.12</td>
    </tr>
    <tr>
      <th>217</th>
      <td>2014-10-29T11-01-00.32</td>
      <td>2014-10-29T11-01-00.32</td>
      <td>38.592167</td>
      <td>-112.594500</td>
      <td>0.94</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>218</th>
      <td>2014-11-15T16-47-50.40</td>
      <td>2014-11-15T16-47-50.40</td>
      <td>38.586167</td>
      <td>-112.600333</td>
      <td>0.79</td>
      <td>2.93</td>
    </tr>
    <tr>
      <th>219</th>
      <td>2014-11-29T14-18-04.87</td>
      <td>2014-11-29T14-18-04.87</td>
      <td>38.698167</td>
      <td>-112.544500</td>
      <td>0.89</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>220</th>
      <td>2014-12-17T08-52-09.85</td>
      <td>2014-12-17T08-52-09.85</td>
      <td>38.579500</td>
      <td>-112.589833</td>
      <td>1.47</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>221</th>
      <td>2014-12-21T17-01-29.91</td>
      <td>2014-12-21T17-01-29.91</td>
      <td>38.586500</td>
      <td>-112.587167</td>
      <td>0.56</td>
      <td>0.69</td>
    </tr>
  </tbody>
</table>
<p>222 rows × 6 columns</p>
</div>



Because there are so many events it may take some time to get the data. We will skip getting the continuous data because it is not needed for this section of the tutorial. We should probably also start the logger in case we need more info than what is printed to the screen. We will also delete an old logger if there is one.


```python
import os
if os.path.exists("detex_log.log"):
    os.remove("detex_log.log")
detex.setLogger()
detex.getdata.makeDataDirectories(getContinuous=False)
```

    Getting template waveforms
    EventWaveForms is not indexed, indexing now
    finished makeDataDirectories call


    /home/derrick/anaconda/lib/python2.7/site-packages/obspy/mseed/core.py:610: UserWarning: The encoding specified in trace.stats.mseed.encoding does not match the dtype of the data.
    A suitable encoding will be chosen.
      warnings.warn(msg, UserWarning)


Now we will cluster these events while varying the input arguments. Let's start by using the defaults.


```python
%time cl = detex.createCluster() # notice we can call createCluster from the detex level
```

    Cannot remove response without a valid inventoryArg, setting removeResponse to False
    Starting IO operations and data checks
    2004-07-29T23-09-58.55 on UU.IMU is out of length tolerance, removing
    2005-01-02T06-35-22.46 on UU.IMU is out of length tolerance, removing
    2005-09-30T13-47-54.79 on UU.IMU is out of length tolerance, removing
    2005-11-28T16-18-36.82 on UU.IMU is out of length tolerance, removing
    2005-12-21T04-48-05.24 on UU.IMU is out of length tolerance, removing
    2004-07-29T23-09-58.55 is fractured or missing data, removing
    2004-05-19T20-44-17.41 on UU.MSU is out of length tolerance, removing
    2005-01-02T06-35-22.46 on UU.MSU is out of length tolerance, removing
    2005-01-14T01-30-48.97 on UU.MSU is out of length tolerance, removing
    2005-08-09T06-35-29.44 on UU.MSU is out of length tolerance, removing
    2005-09-30T13-47-54.79 on UU.MSU is out of length tolerance, removing
    2005-12-21T04-48-05.24 on UU.MSU is out of length tolerance, removing
    2006-02-05T23-33-43.00 on UU.MSU is out of length tolerance, removing
    performing cluster analysis on UU.IMU
    performing cluster analysis on UU.MSU
    ccReq for station UU.IMU updated to ccReq=0.500
    ccReq for station UU.MSU updated to ccReq=0.500
    writing ClusterStream instance as clust.pkl
    CPU times: user 2min 36s, sys: 5min 50s, total: 8min 27s
    Wall time: 2min 36s


    No handlers could be found for logger "detex.getdata.__init__"


We see the wall time for the createCluster call was around 2 minutes (on my computer). Let's make a function to see how many of the original 220 events were actually used


```python
def check_cluster(cl):
    for c in cl:
        sta = c.station
        num_events = len(c.key)
        print '%s had %d events used in the analysis' % (sta, num_events)
    print '\n'
def get_unused_events(cl, temkey):
    for c in cl:
        sta = c.station
        unused = list(set(temkey.NAME) - set(c.key))
        print 'Unused events on %s are:\n %s\n' % (sta, unused)

def get_info(cl, temkey_in='TemplateKey.csv'):
    temkey = detex.util.readKey(temkey_in, 'template')
    print 'There are %d events in the template key' % len(temkey) 
    check_cluster(cl)
    get_unused_events(cl, temkey)

get_info(cl)
```

    There are 222 events in the template key
    UU.IMU had 213 events used in the analysis
    UU.MSU had 214 events used in the analysis
    
    
    Unused events on UU.IMU are:
     ['2005-07-28T16-14-45.72', '2004-07-29T23-09-58.55', '2005-01-02T06-35-22.46', '2010-03-04T00-02-53.67', '2005-07-30T08-58-00.45', '2005-12-21T04-48-05.24', '2005-09-30T13-47-54.79', '2005-11-28T16-18-36.82', '2009-10-07T18-44-39.86']
    
    Unused events on UU.MSU are:
     ['2005-09-30T13-47-54.79', '2004-05-19T20-44-17.41', '2004-07-29T23-09-58.55', '2005-01-02T06-35-22.46', '2005-12-21T04-48-05.24', '2005-01-14T01-30-48.97', '2005-08-09T06-35-29.44', '2006-02-05T23-33-43.00']
    


Now let's try using fillZeros as True rather than the default of False. This will force each event waveform to be exactly the length defined by the trim parameter by filling with zeros where necessary.  


```python
%time cl2 = detex.createCluster(fillZeros=True)
```

    Cannot remove response without a valid inventoryArg, setting removeResponse to False
    Starting IO operations and data checks
    performing cluster analysis on UU.IMU
    performing cluster analysis on UU.MSU
    ccReq for station UU.IMU updated to ccReq=0.500
    ccReq for station UU.MSU updated to ccReq=0.500
    writing ClusterStream instance as clust.pkl
    CPU times: user 2min 43s, sys: 5min 36s, total: 8min 20s
    Wall time: 2min 24s



```python
get_info(cl2)
```

    There are 222 events in the template key
    UU.IMU had 218 events used in the analysis
    UU.MSU had 222 events used in the analysis
    
    
    Unused events on UU.IMU are:
     ['2005-07-28T16-14-45.72', '2009-10-07T18-44-39.86', '2010-03-04T00-02-53.67', '2005-07-30T08-58-00.45']
    
    Unused events on UU.MSU are:
     []
    


So setting fill_zeros to True caused detex to use all the events on MSU and all but four on IMU. The four IMU events that went unused were probably due to missing waveforms. We can verify this by looking in the log for indications the that the data were not available to download.


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
      <td>2015-12-14 22:26:20,753</td>
      <td>detex</td>
      <td>INFO</td>
      <td>Starting logging, path to log file: /home/derr...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-12-14 22:26:21,050</td>
      <td>detex.getdata.makeDataDirectories</td>
      <td>INFO</td>
      <td>Getting template waveforms</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-12-14 22:26:22,396</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-12-14 22:26:22,402</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-12-14 22:26:22,407</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-12-14 22:26:22,413</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-12-14 22:26:22,419</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-12-14 22:26:22,424</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-12-14 22:26:22,430</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-12-14 22:26:23,903</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2015-12-14 22:26:23,909</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2015-12-14 22:26:23,915</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2015-12-14 22:26:23,921</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2015-12-14 22:26:23,926</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2015-12-14 22:26:25,190</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2015-12-14 22:26:25,196</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2015-12-14 22:26:25,202</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2015-12-14 22:26:25,208</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2015-12-14 22:26:25,213</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2015-12-14 22:26:25,219</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2015-12-14 22:26:25,224</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2015-12-14 22:26:25,230</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2015-12-14 22:26:25,235</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2015-12-14 22:26:26,237</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2015-12-14 22:26:26,244</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2015-12-14 22:26:26,250</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2015-12-14 22:26:26,255</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015-12-14 22:26:26,261</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2015-12-14 22:26:26,267</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2015-12-14 22:26:26,272</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>174</th>
      <td>2015-12-14 22:30:21,939</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>175</th>
      <td>2015-12-14 22:30:21,945</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>176</th>
      <td>2015-12-14 22:30:21,951</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>177</th>
      <td>2015-12-14 22:30:21,956</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2015-12-14 22:30:21,962</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2015-12-14 22:30:21,968</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2015-12-14 22:30:21,973</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2015-12-14 22:30:21,979</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>182</th>
      <td>2015-12-14 22:30:21,984</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>183</th>
      <td>2015-12-14 22:30:21,990</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>184</th>
      <td>2015-12-14 22:30:23,109</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>185</th>
      <td>2015-12-14 22:30:24,314</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>186</th>
      <td>2015-12-14 22:30:24,319</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>187</th>
      <td>2015-12-14 22:30:24,325</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>188</th>
      <td>2015-12-14 22:30:24,331</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>189</th>
      <td>2015-12-14 22:30:24,336</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>190</th>
      <td>2015-12-14 22:30:24,342</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>191</th>
      <td>2015-12-14 22:30:24,347</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>192</th>
      <td>2015-12-14 22:30:25,736</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>193</th>
      <td>2015-12-14 22:30:25,742</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>194</th>
      <td>2015-12-14 22:30:25,748</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>2015-12-14 22:30:25,754</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>196</th>
      <td>2015-12-14 22:30:25,760</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>197</th>
      <td>2015-12-14 22:30:26,965</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>198</th>
      <td>2015-12-14 22:30:26,972</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>199</th>
      <td>2015-12-14 22:30:26,979</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>200</th>
      <td>2015-12-14 22:30:26,986</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2015-12-14 22:30:26,993</td>
      <td>detex.getdata._dataCheck</td>
      <td>WARNING</td>
      <td>Found non-int sampling_rates, rounded to neare...</td>
    </tr>
    <tr>
      <th>202</th>
      <td>2015-12-14 22:33:46,231</td>
      <td>detex.getdata.indexDirectory</td>
      <td>INFO</td>
      <td>EventWaveForms is not indexed, indexing now</td>
    </tr>
    <tr>
      <th>203</th>
      <td>2015-12-14 22:33:48,172</td>
      <td>detex.getdata.makeDataDirectories</td>
      <td>INFO</td>
      <td>finished makeDataDirectories call</td>
    </tr>
  </tbody>
</table>
<p>204 rows × 4 columns</p>
</div>



### Time Trials
If you are trying to perform waveform clustering on a large data set it may be worth your time to understand how varying certain parameters can affect runtimes. Let's isolate a few variables and compare run times from the default values. If you are running this on your computer at home it may take some time, skip ahead if you aren't interested. 


```python
# Setup code for time trials
import time
def timeit(func): # decorator for timing function calls
    def wraper(*args, **kwargs):
        t = time.time()
        out = func(*args, **kwargs)
        return (time.time() - t, out)
    return wraper

@timeit
def time_cluster(*args, **kwargs):
    detex.createCluster(*args, **kwargs)

```


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
detex.verbose = False # silence detex

cols = ['waveform_duration', 'run_time']
df = pd.DataFrame(columns=cols)

trims = [(10, 120), (5, 60), (2, 30), (1, 15)]

for trim in trims:
    rt = time_cluster(trim=trim)[0]
    ser = pd.Series([sum(trim), rt], index=cols)
    df.loc[len(df)] = ser
    
plt.plot(df.waveform_duration, df.run_time)
plt.title("Waveform Length vs Run Times")
plt.ylabel("run times (seconds)")
plt.xlabel("waveform lengths (seconds)")

    
    
```




    <matplotlib.text.Text at 0x7f0ea9cc1290>




![png](output_18_1.png)



```python
cols = ['num_events', 'run_time']
df = pd.DataFrame(columns=cols)

temkey = detex.util.readKey("TemplateKey.csv", "template")

temkey_lengths = [10, 20, 50, 100, 150, 200]

for tkl in temkey_lengths:
    temkey2 = temkey.copy()
    
    rt = time_cluster(templateKey=temkey2[:tkl+1])[0]
    ser = pd.Series([tkl, rt], index=cols)
    df.loc[len(df)] = ser
    
plt.plot(df.num_events, df.run_time)
plt.title("Number of Events vs Runtimes")
plt.xlabel("Number of Events")
plt.ylabel("Runtimes (seconds)")

```




    <matplotlib.text.Text at 0x7f0e2c93a5d0>




![png](output_19_1.png)




Although a bit more complicated than this, we could qualitatively estimate that changing the waveform length scales the runtime by approximately N (linearly with time) whereas the number of events scales the runtime by approximately N<sup>2</sup> (quadratic with time). Let's see how decimating the data changes the runtimes. 



```python
# Test various decimation factors
rt_base = time_cluster()[0]
rt_decimate = time_cluster(decimate=2)[0]
print("Base run time: %.02f, Decimated run time: %.02f" % (rt_base, rt_decimate))

```

    Base run time: 136.88, Decimated run time: 101.74


Interestingly, this didn't seem to make much of a difference. The original data were sampled at 100 Hz so using a decimation factor of 2 would have reduced the sampling rate to 50 Hz. Since we left the default bandpass filter (1.0 to 10.0 Hz) it might make sense to use a decimation factor of 4 in order to bring the sampling rate down to 25 Hz. 

## ClusterStream and Cluster Classes

The ClusterStream and Cluster classes are used to control and visualize waveform similarity clustering. These classes are required to define the subspaces used in the detection process.

The ClusterStream is a container for one or more Cluster instances. There is a cluster instance for each station, although most attributes are accessible from the ClusterStream level. Let's take create a ClusterStream instance and take a closer look.  


```python
import detex # reimport so we can start here
detex.verbose = False
cl = detex.createCluster()
```

The bulk of the information for the ClusterStream is stored in the trdf attribute, which, of course, is a pandas DataFrame.


```python
cl.trdf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Station</th>
      <th>Link</th>
      <th>CCs</th>
      <th>Lags</th>
      <th>Subsamp</th>
      <th>Events</th>
      <th>Stats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UU.IMU</td>
      <td>[[194.0, 195.0, 0.00121253182149, 2.0], [105.0...</td>
      <td>1         2         3         4     ...</td>
      <td>1     2     3    4    5     6     7    8...</td>
      <td>1         2         3         4     ...</td>
      <td>[2002-02-01T04-37-29.94, 2002-02-02T03-38-36.2...</td>
      <td>{u'2002-11-09T21-38-33.05': {u'Nc': 1, u'sampl...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>UU.MSU</td>
      <td>[[195.0, 196.0, 0.0035685379758, 2.0], [158.0,...</td>
      <td>1         2         3         4     ...</td>
      <td>1     2     3      4     5     6     7   ...</td>
      <td>1         2         3         4    ...</td>
      <td>[2002-02-01T04-37-29.94, 2002-02-02T03-38-36.2...</td>
      <td>{u'2002-11-09T21-38-33.05': {u'Nc': 1, u'sampl...</td>
    </tr>
  </tbody>
</table>
</div>



In this DataFrame there is a row for each station. The columns are:

| Column | Description |
|:-----:| :---------: |
| CCs | A matrix of max correlation coef for each station pair |
| Lags | A matrix of lag samples corresponding to the highest correlation coef |
| Subsamp | The decimal fraction determined by subsample extrapolation |
| Events | The name of the events used |
| Stats | Selected stats of the events |

The CCs and Lags are DataFrames that have indices and rows that correspond to an element in the Events list. This is probably best illustrated by an example. Let's say we want to find the max correlation ceof. between two events and the corresponding number of samples that would be required to shift the first event to line up with the second. First, we need to find where the events we want to find occur in the events list, then we can index them in the lags and ccs.



```python
# Here are two events in the list
ev1 = '2010-07-10T08-57-51.25'
ev2 = '2014-11-29T14-18-04.87'
events = list(cl.trdf.loc[0, 'Events']) # cast from np array to list
# Find the index where each event occurs in the list
ev1_ind = events.index(ev1)
ev2_ind = events.index(ev2)
print ("%s index is %d, %s index is %d" % (ev1, ev1_ind, ev2, ev2_ind))


```

    2010-07-10T08-57-51.25 index is 125, 2014-11-29T14-18-04.87 index is 210



```python
cc = cl.trdf.loc[0, 'CCs']
lags = cl.trdf.loc[0, 'Lags']
coef = cc.loc[ev1_ind, ev2_ind]
lag = lags.loc[ev1_ind, ev2_ind]
print (coef, lag)
# events
```

    (0.12697600464301972, 263)


### Visualization Methods
The ClusterStream has several methods for visualizing. We can create a simple similarity matrix.


```python
cl.simMatrix()
```


![png](output_31_0.png)



![png](output_31_1.png)


By default the events (x and y axis) are ordered based on origin time. We can also plot them based on the groups the events best fit in. 


```python
cl.simMatrix(groupClusts=True)
```


![png](output_33_0.png)



![png](output_33_1.png)


We can visualize and change the clustering structure for each station with the dendro and updateReqCC methods, just as in the intro tutorial. 


```python
cl.dendro()
```


![png](output_35_0.png)



![png](output_35_1.png)



```python
cl[0].updateReqCC(.6)
cl[0].dendro()
```


![png](output_36_0.png)


We can plot the spatial relations of the events with the plotEvents method. This is used to get a quick and dirty idea of event locations and depths; it still needs a lot of work before it will produce presentable plots. The following is not the best example of a meaningful plot because there are so many colors and different groups but plotEvents can be useful, especially on smaller datasets. 


```python
cl[0].plotEvents()
```


![png](output_38_0.png)



![png](output_38_1.png)



![png](output_38_2.png)


# Next section
The [next section](../SubspaceDetection/subspace_detection1.md) covers subspace detection.
