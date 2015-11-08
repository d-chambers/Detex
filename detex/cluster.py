# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:01:31 2015

@author: derrick

Module to preform waveform similarity clustering

"""

import pandas as pd
import detex

def createCluster(CCreq=0.5, 
                  indir='EventWaveForms', 
                  templateDir=True, 
                  filt=[1, 10, 2, True], 
                  StationKey='StationKey.csv',
                  TemplateKey='TemplateKey.csv', 
                  trim=[100, 200], 
                  filelist=None, 
                  allram=True, 
                  masterStation=None, 
                  saveclust=True,
                  clustname='clust.pkl', 
                  decimate=None, 
                  dtype='double', 
                  consistentLength=True, 
                  eventsOnAllStations=False,
                  subSampleExtrapolate=True, 
                  enforceOrigin=False):
    """ 
    Function to initialize an instance of the cluster class which 
    contains the linkage matrix, event names, and a few visualization 
    methods
    
    Parameters
    -------
    CCreq : float, between 0 and 1
        The minimum correlation coefficient for a grouping waveforms. 
        0 means all waveforms grouped together, 1 will not form any 
        groups (in order to run each waveform as a correlation detector)
    indir : str
        Path to the directory containing the event waveforms
    templateDir : boolean
        If true indicates indir is formated in the way detex.getdata 
        organizes the directories
    filt : list
        A list of the required input parameters for the obspy bandpass 
        filter [freqmin,freqmax,corners,zerophase]
    StationKey : str
        Path to the station key used by the events 
    TemplateKey : boolean
        Path to the template key 
    trim : list 
        A list with seconds to trim from start of each stream in [0] 
        and the total duration in seconds to keep from trim point in 
        [1], the second parameter greatly influences the runtime and 
        if the first parameter is incorrectly selected the waveform 
        may be missed entirely
    filelist : list of str or None
        A list of paths to obspy readable seismic records. If none 
        use indir
    allram : boolean 
        If true then all the traces are read into the ram before 
        correlations, saves time but could potentially fill up ram 
        (Only True is currently supported)
    masterStation : str
        Allows user to set which station in StationKey should be used 
        for clustering, if the string of a single station name is passed 
        cluster analysis is only performed on that station. The event 
        groups will then be forced for all stations. If none is passed 
        then all stations are clustered independently (IE no master station
        to force event groups in subspace class)
    saveClust : boolean
        If true save the cluster object in the current working 
        directory as clustname
    clustname : str
        path (or name) to save the clustering instance, only used 
        if saveClust is True
    decimate : int or None
        A decimation factor to apply to all data (parameter is simply 
        passed to the obspy trace/stream method decimate). 
        Can greatly increase speed and is desirable if the data are 
        oversampled
    dytpe : str
        The data type to use for recasting both event waveforms and 
        continuous data arrays. If none the default of float 64 is 
        kept. Options include:
            double- numpy float 64
            single- numpy float 32, much faster and amenable with 
                cuda GPU processing, sacrifices precision
    consistentLength : boolean
        If true the data in the events files are more or less the 
        same length. Switch to false if the data are not, but can 
        greatly increase run times. 
    eventsOnAllStations : boolean
        If True only use the events that occur on all stations, if 
        false let each station have an independent event list
    subSampleExtrapolate : boolean
        If True subsample extrapolate lag times
    enforceOrigin : boolean
        If True make sure each traces starts at the reported origin time 
        for a give event (trim or merge with zeros if not). Required 
        for lag times to be meaningful for hypoDD input
    Returns
    ---------
        An instance of the detex SSClustering class
    """
    # Read in stationkey, delete blank rows, and set master station, 
    #if no master station selected us first station
    stakey = pd.read_csv(StationKey)
    legitRows = [isinstance(x, str) or abs(x) >= 0 for x in stakey.NETWORK]
    stakey = stakey[legitRows]
    # make sure station and network are strs
    stakey['STATION'] = [str(x) for x in stakey.STATION]
    stakey['NETWORK'] = [str(x) for x in stakey.NETWORK]
    if isinstance(masterStation, str):
        masterStation = [masterStation]
    if isinstance(masterStation, list):
        # if NETWORK.STATION format is used convert to just station
        if len(masterStation[0].split('.')) > 1:
            masterStation = [x.split('.')[1] for x in masterStation]
        stakey = stakey[stakey.STATION.isin(masterStation)] # trim stakey
    if len(stakey) == 0:
        msg = 'Master station is not in the station key, aborting clustering' 
        detex.log(__name__, msg, level='error')
    temkey = pd.read_csv(TemplateKey)
    
    # if template key is not sorted sort it and save over unsorted version
    if not temkey.equals(temkey.sort(columns='NAME')):
        temkey.sort(columns='NAME', inplace=True)
        temkey.reset_index(inplace=True, drop=True)
        temkey.to_csv(TemplateKey, index=False)
    else:
        temkey.reset_index(inplace=True, drop=True)
        
    # if station key is not sorted sort it and save over unsorted version
    if not stakey.equals(stakey.sort(columns=('NETWORK', 'STATION'))):
        stakey.sort(columns=('NETWORK', 'STATION'), inplace=True)
        stakey.reset_index(inplace=True, drop=True)
        stakey.to_csv(TemplateKey, index=False)
    else:
        stakey.reset_index(drop=True, inplace=True)

    # Intialize parts of DF that will be used to store cluster info
    if not templateDir:
        filelist = glob.glob(os.path.join(indir, '*'))
    TRDF = _loadEvents(filelist, indir, filt, trim, stakey, templateDir,
                       decimate, temkey, dtype, enforceOrigin=enforceOrigin)
    # deb(TRDF)
    TRDF.sort(columns='Station', inplace=True)
    TRDF.reset_index(drop=True, inplace=True)
    if consistentLength:
        TRDF = _testStreamLengths(TRDF)
    # deb([TRDF,ddf])
    TRDF['Link'] = None

    # Loop through stationkey performing cluster analysis only on stationkey
    if eventsOnAllStations:  # If only using events common to all stations
        # get list of events that occur on all required stations
        eventList = list(set.intersection(*[set(x) for x in TRDF.Events]))
        eventList.sort()
        if len(eventList) < 2:
            detex.log(
                __name__, 'less than 2 events in population have required stations', level='warning')
            raise Exception(
                'less than 2 events in population have required stations')
    for a in TRDF.iterrows():  # loop over master station(s)
        detex.log(__name__, 'getting CCs and lags on ' +
                  a[1].Station, pri=True)
        if not eventsOnAllStations:
            eventList = a[1].Events
        if len(a[1].Events) < 2:  # if only one event on this station skip it
            continue
        DFcc, DFlag = _makeDFcclags(
            eventList, a, consistentLength=consistentLength, subSampleExtrapolate=subSampleExtrapolate)
        TRDF.Lags[a[0]] = DFlag
        TRDF.CCs[a[0]] = DFcc
        # deb([DFlag,DFcc])
        cx = np.array([])
        lags = np.array([])
        cxdf = 1.0000001 - DFcc
        # flatten cxdf and index out nans
        cxx = [x[xnum:] for xnum, x in enumerate(cxdf.values)]
        cx = np.fromiter(itertools.chain.from_iterable(cxx), dtype=np.float64)
        # ensure x is unique in order to link correlation coefficients to lag
        # time in dictionary
        cx, cxdf = _ensureUnique(cx, cxdf)
        for b in DFlag.iterrows():  # TODO this is not efficient, consider rewriting without loops
            lags = np.append(lags, b[1].dropna().tolist())
        link = linkage(cx)  # get cluster linkage
        TRDF['Link'][a[0]] = link
        # TRDF.loc[a[0],'Link']=link
    # DFcc=pd.DataFrame(cxw,index=range(len(cxw)),columns=range(1,len(cxw)+1))
    # a truncated TRDF, only passing what is needed for clustering
    trdf = TRDF[['Station', 'Link', 'CCs', 'Lags', 'Events', 'Stats']]

    # try:
    clust = ClusterStream(trdf, temkey, eventList, CCreq, filt, decimate, trim, indir, saveclust, clustname,
                          templateDir, filelist, StationKey, saveclust, clustname, eventsOnAllStations, stakey, enforceOrigin)
#    except:
#        deb([trdf,TRDF,a])
    if saveclust:
        clust.write()
    return clust
