
#  Detex Tutorial

Detex is a python package for performing waveform similarity clustering and [subspace detection](https://e-reports-ext.llnl.gov/pdf/335299.pdf). It is written in python and relies heavily on Obspy, Numpy, Scipy, Matplotlib, and Pandas. If you are not familiar with python I recommend you install the [anaconda distribution](http://continuum.io/downloads) for your platform and spend a few hours learning the basics of the language before attempting to use Detex. Here are some great tutorials:

http://www.stavros.io/tutorials/python/
http://www.tutorialspoint.com/python/python_quick_guide.htm

Also, any time spent learning [obspy](http://docs.obspy.org/tutorial/) is a good investment as it is a very powerful tool for processing geophysical data.

Some knowledge of [pandas](http://pandas.pydata.org/pandas-docs/stable/tutorials.html) will also be useful, as the pandas DataFrame is used extensively in detex.

Special thanks to Tex Kubacki (whose work inspired Detex), Jared Stein, Lisa Linvile, Shawn Blotz, and Chase Batchelor.

## Installation 

Detex can be installed by running the setup.py script in the distribution directory. 

## Workflow

The basic Detex workflow has five steps:

1. Prepare required files
2. Data acquisition
3. Clustering
4. Subspace detection
5. Detection association



##1. Prepare required files
There are two required files: the station key and the template key.

The station key is generally saved as StationKey.csv. The following is an example:


| NEWTORK | STATION | STARTTIME | ENDTIME | LAT | LON | ELEVATION | CHANNELS |
|:----------:|:-----------:|:------------:| :-: | :-: | :-: | :-: | :-: |
| TA | M18A	| 2009-04-01T00:00:00 | 2009-04-04T00:00:00	| 41.4272 | -110.0674 | 2103 | BHE-BHN-BHZ |
| TA | M17A	| 2009-04-01T00:00:00 | 2009-04-04T00:00:00	| 41.4729 | - 110.6664 | 2101 | BHE-BHN-BHZ |

The STARTTIME and ENDTIME fields indicated the time range of the continuous data and can be in any format readable by the obspy.UTCDateTime class (including a time stamp). See the [obspy.UTCDateTime docs](http://docs.obspy.org/packages/autogen/obspy.core.utcdatetime.UTCDateTime.html) for more info 

The CHANNELS field should list the channels that will be used for each station separated by a dash (-). Additionally, any extra fields can be added without affecting Detex's ability to read the file.

The template key by default is saved as TemplateKey.csv. It contains information on each of the events that will be used to scan the continuous data for previously undetected events. Here is a few lines of the template key included in this tutorial:

| CONTRIBUTOR | NAME | TIME | LAT | LON | DEPTH | MTYPE | MAG |
| :--: | :--: | :--: | :--: | :--: |:--: | :--: |:--: |
| ANF | 2007-12-19T17-56-18 | 2007-12-19T17-56-18 | 41.7205	| -110.6486	| 4.07 | ML | 2.36 |
| ANF | 2007-12-21T18-30-09	| 2007-12-21T18-30-09 | 41.7669	| -110.6122	| 8.97 | ML | 2.17 |
| ANF | 2007-12-21T18-30-09	| 2007-12-21T18-30-09 | 41.7669	| -110.6122	| 8.97 | ML	| 2.17 |

The CONTRIBUTOR, MTYPE, and DEPTH fields are not required but can be useful for record keeping. Additionally, any extra fields can be added in any order in order to better keep track of the events. 

The NAME field can be any string that can also be used as a file name by your OS. Windows does not allow ":" in a file path so the ":" between the hour and minute, and between the minute and seconds, have been replaced with a "-". Again, the time field can be in any format understood by obspy.UTCDateTime.

The LAT and LON fields are not strictly required for basic Detex functionality, but are used in some visualization methods. 


##2. Data aquisition
Detex uses obspy's fdsn module to download seismic data from a variety of data centers. For supported options see the [obspy documentation](http://docs.obspy.org/packages/obspy.fdsn.html)

The getdata module is used for acquisition. Once the template key and station key have been created the data can be downloaded by using the getAllData function. Progress will periodically print to screen.



    import detex #import detex module
    # the next line is only needed in ipython notebook to make sure all figures show inline rather than opening seperate windows
    %pylab inline 
    detex.getdata.getAllData() #download all data from iris

    Populating the interactive namespace from numpy and matplotlib
    Getting template waveforms
    --------------------------
    25 events downloaded out of 81 for Station M18A
    25 events downloaded out of 81 for Station M17A
    50 events downloaded out of 81 for Station M18A
    50 events downloaded out of 81 for Station M17A
    75 events downloaded out of 81 for Station M18A
    75 events downloaded out of 81 for Station M17A
    Getting continuous data, this may take several hours depending on request size
    ------------------------------------------------------------------------------
    

##  3. Clustering
The next step is to cross correlate every event with every other event in order to form waveform similarity groupings on each station. A single link-algorithm is used to perform the clustering up to a determined dissimilarity level.   

In order to do this a clusterStream object is created, which is essentially a container for cluster objects created using data from each station independently. The main input parameter is the required correlation coefficient, below which clustering will not occur. If you want to run each waveform as a 1D subspace (IE in waveform correlation detection) you can simply set the required correlation coefficient to 1. Conversely, if you want to include all events in the subspace regardless of similarity then set this parameter to 0. The default value is 0.5 can be easily changed without re-running the correlations. For example:


    cl=detex.subspace.createCluster() # Create a clusters stream object

    <generator object <genexpr> at 0x0000000003F1CE10>
    getting CCs and lags on TA.M17A
    getting CCs and lags on TA.M18A
    CCreq for station TA.M17A updated to CCreq=0.500
    CCreq for station TA.M18A updated to CCreq=0.500
    writing cluster object as clust.pkl
    writing cluster object as clust.pkl
    


    cl.dendro() #create a dendrogram to visualize grouping structure on each station


![png](ReadMe/Images/output_7_0.png)



![png](ReadMe/Images/output_7_1.png)


Now if we wanted to form strictly 4 groups on each station we can modify the required correlation coefficient for grouping. This can be done for all stations at once or for each station individually. 


    cl.updateReqCC(.55)
    cl.dendro()

    CCreq for station TA.M17A updated to CCreq=0.550
    CCreq for station TA.M18A updated to CCreq=0.550
    writing cluster object as clust.pkl
    


![png](ReadMe/Images/output_9_1.png)



![png](ReadMe/Images/output_9_2.png)



    cl['TA.M17A'].updateReqCC(.38) # set required correlation coef. for only station TA.M17A
    cl['TA.M17A'].dendro() # visualize grouping

    CCreq for station TA.M17A updated to CCreq=0.380
    


![png](ReadMe/Images/output_10_1.png)



    # Similarity matricies can also be generated
    cl.simMatrix()


![png](ReadMe/Images/output_11_0.png)



![png](ReadMe/Images/output_11_1.png)


There are several other functions of the ClusterStream class. Notably, input for [hypoDD](http://www.ldeo.columbia.edu/~felixw/hypoDD.html]) (a well-established double difference relocation program) can be created using the writeHypoDDEventInput, writeHypoDDStationInput, and writeSimpleHypoDDInput class methods; although as of version 0.1.0 they have not been fully tested. I hope to develop other methods for locating detected events in the future. 

## 4. Subspace detection
The subspace creation process is applied to each waveform similarity group. The process involves 1) aligning the waveforms to optimize similarity, 2) performing a singular value decomposition, 3) determining a required dimension of representation, and 4) setting a significant detection statistic threshold. As a final step 5) the subspace detectors are run on each station and saved to an SQLite database.


    # First, the creation of the SubSpaceStream
    ss= detex.subspace.createSubSpace() 

    Starting Subspace Construction
    <generator object <genexpr> at 0x000000001902E1F8>
    

### 4.1 Trim waveforms
The next step is to set the beginning trim time for each subspace group and un-clustered singleton on each station. This can be done by calling GUI based pick functions built into the SubSpaceStream class (here it would be ss.detex.pickTimes()),or by attaching a csv or pickled pandas data frame with the following populated fields populated for at least one phase of each event-station pair:
TimeStamp, Station, Event, Phase. 

The EventPicks.csv file included in the tutorial is such a file (you can create this file by calling detex.util.pickPhases or load the dataframe by calling pandas.read_csv). 

Detex will then find the first arriving phase for each waveform (event-station pair) and average for the entire aligned group. From the average first arrival sample the waveforms will be trimmed to some duration ( 30 seconds default) or to the last arriving phase. See the function docs for further details.



    ss.attachPickTimes()

### 4.2, 4.3, 4.4 Perform SVD, set dimension of representation, and set threshold
Next a singular value decomposition is performed on the  waveform groups that have been aligned and trimmed. A dimension of representation (IE the number of left singular vectors used to describe the waveform family) is calculated based on the fractional energy capture of 90% (by default). A detection statistic (DS) threshold for each subspace and singleton is then determined by calculating the detection statistic of the subspace with random continuous data that contains no high amplitude signals, fitting a beta PDF to the distribution, and finding the DS corresponding to the selected probability of false detection $10^{-12}$ by default). 


    ss.SVD()

###4.5 Run detectors
Detex will scan the continuous data for each station-subspace pair and declare a detection whenever any subspace's threshold is exceeded. 


    ss.detex(useSingles=True) # run subspace detections and also run unclustered events as 1D subspaces (IE waveform correlation)

    Subspaces on TA.M17A completed, 23 potential detection(s) recorded
    Subspaces on TA.M18A completed, 55 potential detection(s) recorded
    Singletons on TA.M17A completed, 181 potential detection(s) recorded
    Singletons on TA.M18A completed, 12 potential detection(s) recorded
    

The results are saved to an sqlite database. The following tables are saved in the database (named SubSpace.db by default):

| Table | Description |
|:-----:| :---------: |
| ss_df | Results of the detections for the subspaces |
| sg_df | Detection results for the singletons (un-clustered events) |
| filt_params | Filter parameters used for the detections |
| ss_info | General information about each of the subspaces (such as station, comprising events, thresholds, etc.) |
| sg_info | General information about each singleton |
| ss_hist | Binned counts of all detection statistic values for subspaces |
| sg_hist | Binned counts of all detection statistic values for singletons |

Any of these tables can be loaded into a dataframe using the detex.util.loadSQLite function. For example, if we wanted to make an ugly plot of all of the detection statistic values for the subspaces:


    import matplotlib.pyplot as plt #import matplotlib for visualization
    import json # used to convert string arrays as loaded from sql to numpy arrays
    
    hist=detex.util.loadSQLite('SubSpace.db','ss_hist') # load the ss_hist table of the SubSpace.db sqlite database
    
    for ind,row in hist.iterrows(): #loop over datarame
        hist.loc[ind,'Value']=np.array(json.loads(row.Value)) #convert string arrays into numpy arrays 
    
    avbins=(hist.iloc[0].Value[:-1]+hist.iloc[0].Value[1:])/2.0 # middle of bin values for histograms
    
    
    ## Plot each histogram
    for ind, row in hist.iterrows(): # loop again through dataframe
        if ind==0: # skip if index of dataframe is 0 (these are the bin values)
            continue 
        plt.plot(avbins,row.Value,label=row.Sta+':'+row.Name) #plot
    plt.xlabel('Detection Statistic') #label x
    plt.ylabel('Occurrence Rate') # label y
    plt.title('Detection Statistic') # lable title
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5)) #add legend outside of plot
    plt.semilogy() #use semilog on y axis
    plt.xlim([0,.5]) #set x lim
    
    plt.show() #show plot
        


![png](ReadMe/Images/output_22_0.png)


## 5 Associate detections

The detex module "results" is used to associate all of the detections (DS that exceeded the determined threshold) on various stations together into coherent events. The association requirement is an overlap in predicted origin times. If a verification data set (IE event ground-truth) is available it can be used to assess detector performance. 

Note: If possible, it is very important to use at least 2 stations separated in space in order to reduce false detections.



    # associate events together, only require events to occur on 2 stations (that's all we have), count detections verified if they
    # occure within 10 minutes (5 minutes on either side) of reported origin times in blasting catalog. 
    res=detex.results.detResults(requiredNumStations=2,veriBuffer=60*10,veriFile='veriFile.csv') 

The associations may take a while for large data sets, there are still some optimizations that need to be implemented.
The verified detections, new detections, and auto detections (detection of training events) are stored in the form of pandas DataFrames that can be accessed by the Vers, Dets, and Autos attributes of the DetResults object. For example:



    res.Dets




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event</th>
      <th>DSav</th>
      <th>DSmax</th>
      <th>DS_STALTA</th>
      <th>MSTAMPmin</th>
      <th>MSTAMPmax</th>
      <th>Mag</th>
      <th>ProEnMag</th>
      <th>Verified</th>
      <th>Dets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 2009-04-01T20-25-40</td>
      <td> 0.678077</td>
      <td> 0.809730</td>
      <td> 4.453540</td>
      <td> 1.238618e+09</td>
      <td> 1.238618e+09</td>
      <td> 2.265334</td>
      <td> 2.193913</td>
      <td> True</td>
      <td>           DS  DS_STALTA          STMP Name    ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 2009-04-02T16-27-00</td>
      <td> 0.782322</td>
      <td> 0.839309</td>
      <td> 5.275199</td>
      <td> 1.238690e+09</td>
      <td> 1.238690e+09</td>
      <td> 2.252170</td>
      <td> 2.214971</td>
      <td> True</td>
      <td>            DS  DS_STALTA          STMP Name   ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 2009-04-02T19-16-51</td>
      <td> 0.505218</td>
      <td> 0.632107</td>
      <td> 5.026575</td>
      <td> 1.238700e+09</td>
      <td> 1.238700e+09</td>
      <td> 2.061220</td>
      <td> 1.922427</td>
      <td> True</td>
      <td>            DS  DS_STALTA          STMP Name   ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 2009-04-03T17-34-27</td>
      <td> 0.467678</td>
      <td> 0.530095</td>
      <td> 5.594161</td>
      <td> 1.238780e+09</td>
      <td> 1.238780e+09</td>
      <td> 1.902727</td>
      <td> 1.752857</td>
      <td> True</td>
      <td>            DS  DS_STALTA          STMP Name   ...</td>
    </tr>
  </tbody>
</table>
</div>




    res.Autos




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event</th>
      <th>DSav</th>
      <th>DSmax</th>
      <th>DS_STALTA</th>
      <th>MSTAMPmin</th>
      <th>MSTAMPmax</th>
      <th>Mag</th>
      <th>ProEnMag</th>
      <th>Verified</th>
      <th>Dets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 2009-04-01T17-36-58</td>
      <td> 0.997868</td>
      <td> 0.999010</td>
      <td> 9.732202</td>
      <td> 1.238607e+09</td>
      <td> 1.238607e+09</td>
      <td> 2.513489</td>
      <td> 2.528214</td>
      <td> True</td>
      <td>          DS  DS_STALTA          STMP Name     ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 2009-04-03T15-39-27</td>
      <td> 0.912501</td>
      <td> 0.931663</td>
      <td> 6.155092</td>
      <td> 1.238773e+09</td>
      <td> 1.238773e+09</td>
      <td> 2.323294</td>
      <td> 2.320163</td>
      <td> True</td>
      <td>            DS  DS_STALTA          STMP Name   ...</td>
    </tr>
  </tbody>
</table>
</div>




    res.Vers




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Event</th>
      <th>DSav</th>
      <th>DSmax</th>
      <th>DS_STALTA</th>
      <th>MSTAMPmin</th>
      <th>MSTAMPmax</th>
      <th>Mag</th>
      <th>ProEnMag</th>
      <th>Dets</th>
      <th>STMP</th>
      <th>MWD</th>
      <th>VerMag</th>
      <th>VerLat</th>
      <th>VerLon</th>
      <th>VerDepth</th>
      <th>VerName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 2009-04-01T17-36-58</td>
      <td> 0.997868</td>
      <td> 0.999010</td>
      <td> 9.732202</td>
      <td> 1.238607e+09</td>
      <td> 1.238607e+09</td>
      <td> 2.513489</td>
      <td> 2.528214</td>
      <td>          DS  DS_STALTA          STMP Name     ...</td>
      <td> 2009-04-01T17:35:00.000000Z</td>
      <td> 2943</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 9107</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 2009-04-01T20-25-40</td>
      <td> 0.678077</td>
      <td> 0.809730</td>
      <td> 4.453540</td>
      <td> 1.238618e+09</td>
      <td> 1.238618e+09</td>
      <td> 2.265334</td>
      <td> 2.193913</td>
      <td>           DS  DS_STALTA          STMP Name    ...</td>
      <td> 2009-04-01T20:22:00.000000Z</td>
      <td> 1713</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 9108</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 2009-04-02T16-27-00</td>
      <td> 0.782322</td>
      <td> 0.839309</td>
      <td> 5.275199</td>
      <td> 1.238690e+09</td>
      <td> 1.238690e+09</td>
      <td> 2.252170</td>
      <td> 2.214971</td>
      <td>            DS  DS_STALTA          STMP Name   ...</td>
      <td> 2009-04-02T16:24:00.000000Z</td>
      <td> 1542</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 9109</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 2009-04-02T19-16-51</td>
      <td> 0.505218</td>
      <td> 0.632107</td>
      <td> 5.026575</td>
      <td> 1.238700e+09</td>
      <td> 1.238700e+09</td>
      <td> 2.061220</td>
      <td> 1.922427</td>
      <td>            DS  DS_STALTA          STMP Name   ...</td>
      <td> 2009-04-02T19:20:00.000000Z</td>
      <td>  812</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 9110</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 2009-04-03T15-39-27</td>
      <td> 0.912501</td>
      <td> 0.931663</td>
      <td> 6.155092</td>
      <td> 1.238773e+09</td>
      <td> 1.238773e+09</td>
      <td> 2.323294</td>
      <td> 2.320163</td>
      <td>            DS  DS_STALTA          STMP Name   ...</td>
      <td> 2009-04-03T15:36:00.000000Z</td>
      <td> 2233</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 9111</td>
    </tr>
    <tr>
      <th>5</th>
      <td> 2009-04-03T17-34-27</td>
      <td> 0.467678</td>
      <td> 0.530095</td>
      <td> 5.594161</td>
      <td> 1.238780e+09</td>
      <td> 1.238780e+09</td>
      <td> 1.902727</td>
      <td> 1.752857</td>
      <td>            DS  DS_STALTA          STMP Name   ...</td>
      <td> 2009-04-03T17:30:00.000000Z</td>
      <td> 1022</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 9112</td>
    </tr>
  </tbody>
</table>
</div>



Loading the mine's blast log we can see that all six blasts over the four days were successfully detected with no false detections. Note: the magnitude, latitude, longitude, and depth were not known so I simply use dummy values here.


    import pandas as pd
    log=pd.read_csv('veriFile.csv')
    log




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>TIME</th>
      <th>MAG</th>
      <th>LAT</th>
      <th>LON</th>
      <th>DEPTH</th>
      <th>MWD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td> 9107</td>
      <td> 2009-04-01T17-35-00</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 2943</td>
    </tr>
    <tr>
      <th>1</th>
      <td> 9108</td>
      <td> 2009-04-01T20-22-00</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 1713</td>
    </tr>
    <tr>
      <th>2</th>
      <td> 9109</td>
      <td> 2009-04-02T16-24-00</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 1542</td>
    </tr>
    <tr>
      <th>3</th>
      <td> 9110</td>
      <td> 2009-04-02T19-20-00</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td>  812</td>
    </tr>
    <tr>
      <th>4</th>
      <td> 9111</td>
      <td> 2009-04-03T15-36-00</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 2233</td>
    </tr>
    <tr>
      <th>5</th>
      <td> 9112</td>
      <td> 2009-04-03T17-30-00</td>
      <td> 2</td>
      <td> n/a</td>
      <td> n/a</td>
      <td> 0</td>
      <td> 1022</td>
    </tr>
  </tbody>
</table>
</div>



If we only required the detections to occur on one station, however, (even with the default ultra-conservative acceptable probability of false detection of $10^{-12}$ the detector will return many false detections. For this reason it is important to use more than one station when possible.


    import detex
    res=detex.results.detResults(requiredNumStations=1,veriBuffer=60*10,veriFile='veriFile.csv')
    res




    SSResults instance with 2 autodections and 180 new detections, 6 are verified



Once we have detected new events we can instruct detex to extract the waveforms of the new detections. With the extracted waveforms phase picks can be made in order to located the newly-found events, cross correlation lag times can be calculated with the clustering and the detected events can be used to create a new detector to potentially find more events.


    res.writeDetections(eventDir='DetectedEvents',updateTemKey=False)

Now the waveforms of the newly detected events have been stored with the same directory structure as the TemplateWaveForms directory in a directory named DetectedEvents (because this is the argument we assigned to it). A new template key of the detected events can also be created, or by default the current template key csv will be updated with the newly detected events. The naming convention is the same but detected events will have a lowercase "d," for detected, at the start of the name string. The entire process can then be repeated to try and detect even more events, but in this case we know we have found all that there is to find. 

Note that detex has created a log (detex_log.log) that can be useful in debugging. 

## Detex utilities
There are several useful utilites in the detex.util module. Here I will highlight a few of them.

### KML generation
Using the [simple kml module](https://pypi.python.org/pypi/simplekml/1.2.8) several detex files can be converted to kml for easy viewing in google earth. They include the stations in the station key (detex.util.writeKMLFromStationKey), the template key (detex.util.writeKMLFromTemplateKey), outputs from hypoDD or hypoInverse, etc. 

### Loading arbitrary continuous data
The loadContinuousData function of detex.util can be used to load any data contained in the ContinuousWaveForms directory, trimmed to a user's specifications.

## [Contributing to detex](ReadMe/ContributeToDetex/contributing.md)

