# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 18:39:50 2015

@author: derrick
"""

import pandas as pd
import numpy as np
import obspy
import detex
import scipy
import warnings
import sys

from scipy.cluster.hierarchy import linkage

pd.options.mode.chained_assignment = None #mute setting copy warning 

 
 

################ CLUSTERING FUNCTIONS AND CLASSES  ################

def createCluster(CCreq=0.5, 
                  fetch_arg='EventWaveForms',  
                  filt=[1, 10, 2, True], 
                  StationKey='StationKey.csv',
                  TemplateKey='TemplateKey.csv', 
                  trim=[100, 200], 
                  saveclust=True,
                  fileName='clust.pkl', 
                  decimate=None, 
                  dtype='double', 
                  consisLen=True, 
                  eventsOnAllStations=False,
                  subSamp=True, 
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
    fetch_arg : str or detex.getdata.DataFetcher object
        fetch_arg of detex.getdata.quick_fetch, see docs for details
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
    saveClust : boolean
        If true save the cluster object in the current working 
        directory as clustname
    fileName : str
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
    consisLen : boolean
        If true the data in the events files are more or less the 
        same length. Switch to false if the data are not, but can 
        greatly increase run times. 
    eventsOnAllStations : boolean
        If True only use the events that occur on all stations, if 
        false let each station have an independent event list
    subSamp : boolean
        If True subsample lag times with cosine extrapolation
    enforceOrigin : boolean
        If True make sure each traces starts at the reported origin time 
        for a give event (trim or merge with zeros if not). Required 
        for lag times to be meaningful for hypoDD input
    Returns
    ---------
        An instance of the detex SSClustering class
    """
    # Read in stationkey and template keys
    stakey = detex.util.read_key(StationKey, key_type='station')
    temkey = detex.util.read_key(TemplateKey, key_type='template')
    
    # get a data fetcher
    fetcher = detex.getdata.quick_fetch(fetch_arg)

    # Intialize object DF that will be used to store cluster info
    msg = 'Starting IO operations and data checks'
    detex.log(__name__, msg, level='info', pri=True)
    TRDF = _loadEvents(fetcher, filt, trim, stakey, temkey, decimate,
                       dtype, enforceOrigin=enforceOrigin)
    
    # make sure lengths are all equal else remove problem events
    if consisLen: 
        TRDF = _testStreamLengths(TRDF)

    # Prune event that do not occur on all stations if required
    if eventsOnAllStations:  
        # get list of events that occur on all required stations
        eventList = list(set.intersection(*[set(x) for x in TRDF.Events]))
        eventList.sort()
        if len(eventList) < 2:
            msg = 'less than 2 events in population have required stations'
            detex.log(__name__, msg, level='error')
    
    # Test number of events in each station
    if len(TRDF) < 1:
        msg = 'No events survived preprocessing, examin input args and data'
        detex.log(__name__, msg, level='error', pri=True)
    
    # Loop through entries for each station, perform clustering
    for ind, row in TRDF.iterrows():  
        msg = 'performing cluster analysis on ' + row.Station
        detex.log(__name__, msg, level='info', pri=True)
        if not eventsOnAllStations: 
            eventList = row.Events
        if len(row.Events) < 2:  # if only one event on this station skip it
            msg = 'Less than 2 valid events on station ' + row.station
            detex.log(__name__, msg, level='warning', pri=True)
            continue
        DFcc, DFlag = _makeDFcclags(eventList, row, consisLen=consisLen,
                                    subSamp=subSamp)
        TRDF.Lags[ind] = DFlag
        TRDF.CCs[ind] = DFcc
        cx = np.array([])
        cxdf = 1.0000001 - DFcc # get dissimilarities
        # flatten ccs and remove nans
        cx = _flatNoNan(cxdf)
        
        link = linkage(cx)  # get cluster linkage
        TRDF.loc[ind, 'Link'] = link
    trdf = TRDF[['Station', 'Link', 'CCs', 'Lags', 'Events', 'Stats']]

    #try:
    clust = detex.subspace.ClusterStream(trdf, temkey, stakey, fetcher, 
                                         eventList, CCreq, filt, decimate, 
                                         trim, fileName, eventsOnAllStations,
                                         enforceOrigin)
 
    if saveclust:
        clust.write()
    return clust


######################### SUBSPACE FUNCTIONS AND CLASSES #####################


def createSubSpace(Pf=10**-12, clust='clust.pkl', minEvents=2, dtype='double',
                   conDatFetcher=None):
    """
    Function to create subspaces on all available stations based on the 
    clusters in Clustering object which will either be passed directly as the 
    keyword clust or will be loaded from the path in clustFile

    Parameters
    -----------
    Pf : float
        The probability of false detection as modeled by the statistical 
        framework in Harris 2006 Theory (eq 20, not yet supported)
        Or by fitting a PDF to an empirical estimation of the null space 
        (similar to Wiechecki-Vergara 2001). Thresholds are not set until 
        calling the SVD function of the subspace stream class
    clust: str or instance of detex.subspace.ClusterStream
        The path to a pickled instance of ClusterStream or an instance of 
        ClusterStream. Used in defining the subspaces.
    minEvents : int
        The Min number of events that must be in a cluster in order for a 
        subspace to be created from that cluster
    dtype : str ('single' or 'double')
        The data type of the numpy arrays used in the detections. Options are:
            single- a np.float32, slightly faster (~30%) less precise 
            double- a np.float64 (default)
    conDatFetcher : None, str, or instance of detex.getdata.DataFetcher
        Parameter to indicate how continuous data will be fetched in the newly
        created instance of SubSpaceStream. 
        If None is passed detex will try to deduce the appropriate type of 
        DataFetcher from the event datafetcher attached to cluster instance
        If a str the str will be then be passed to detex.getdata.quickFetch 
        which returns a fetcher based on the str
        If an instance of detex.getdata.DataFetcher is passed then it will be
        used as the continuous data fetcher
        
        
    Returns
    -----------
    An instance of the SubSpaceStream class
    
        Note
    ----------
    Most of the parameters that define how to fetch seismic data, which events
    and stations to use in the analysis, filter parameters, etc. are already 
    defined in the cluster (ClusterStream) instance.     
    """
    # Read in cluster instance
    if isinstance (clust, str): #if no cluster object passed read a pickled one
        cl = detex.subspace.loadClusters(clust)
    elif isinstance(clust, detex.subspace.ClusterStream):
        cl = clust
    else:
        msg = 'Invalid clust type, must be a path or ClusterStream instance.'
        detex.log(__name__, msg, level='error')
    
    # Get info from cluster, load fetchers
    temkey = cl.temkey
    stakey = cl.stakey
    efetcher = cl.fetcher
    if isinstance(conDatFetcher, detex.getdata.DataFetcher):
        cfetcher = conDatFetcher
    elif isinstance(conDatFetcher, str):
        cfetcher = detex.getdata.quick_fetch(conDatFetcher)
    elif conDatFetcher is None:
         if efetcher.method == 'dir':
             cfetcher = detex.getdata.quick_fetch('ContinuousWaveForms')
         else: # if not directory assume obspy client used for both
             cfetcher = efetcher
        
    # Load events into main dataframe to create subspaces
    TRDF = _loadEvents(efetcher, cl.filt, cl.trim, stakey, temkey, cl.decimate, 
                       dtype) 
    for ind, row in TRDF.iterrows(): #Fill in cluster info from cluster object
        TRDF.loc[ind, 'Link'] = cl[row.Station].link
        TRDF.loc[ind, 'Clust'] = cl[row.Station].clusts   
   
    # Start subspace construction
    msg = 'Starting Subspace Construction'
    detex.log(__name__, msg, pri=True)   
    ssDict = {} # dict to store subspaces in
    for num, row in TRDF.iterrows(): # Loop through each cluster
        staSS = _makeSSDF(row, minEvents)
        if len(staSS) < 1: #if no clusters form on current station
            msg = 'No events survived processing on %s' % staSS.station
            detex.log(__name__, msg, level='warning', pri=True)
        for sind, srow in staSS.iterrows(): 
            eventList = srow.Events
            # get correlation values from cl object
            DFcc, DFlag = _getInfoFromClust(cl, srow)
            staSS['Lags'][sind] = DFlag
            staSS['CCs'][sind] = DFcc
            cxdf = 1.0000001 - DFcc 
            cx = _flatNoNan(cxdf)
            cx, cxdf = _ensureUnique(cx, cxdf) # ensure cc value is unique
            lags = _flatNoNan(DFlag)
            link = linkage(cx) #get cluster linkage
            staSS.loc[sind, 'Link'] = link
            # get lag times and align waveforms
            CCtoLag = _makeLagSeries(cx, lags) # a map from cc to lag times
            delays = _getDelays(link, CCtoLag, cx, lags, cxdf)
            delayNP = -1 * np.min(delays)
            delayDF = pd.DataFrame(delays + delayNP, columns=['SampleDelays'])
            delayDF['Events'] = [eventList[x] for x in delayDF.index]
            staSS['AlignedTD'][sind] = _alignTD(delayDF, srow) 
            ustimes = _updateStartTimes(srow, delayDF, temkey)
            staSS['Stats'][sind] = ustimes # update Start Times
            offsets = _getOffsetList(sind, srow, staSS)
            offsetAr = [np.min(offsets), np.median(offsets), np.max(offsets)]
            staSS['Offsets'][sind] = offsetAr
        # Put output into subspaceDict
        staOut = staSS.drop(['MPfd','MPtd','Link','Lags','CCs'],axis=1)
        ssDict[row.Station] = staOut
    # make a list of sngles to pass to subspace class
    singDic = _makeSingleEventDict(cl, TRDF, temkey) 
    
    substream = detex.subspace.SubSpaceStream(singDic, ssDict, cl, dtype, Pf, 
                                              cfetcher)
    return substream
def _getInfoFromClust(cl, srow):
    """
    get the DFcc dataframe and lags dataframe from values already stored in
    cluster object to avoid recalculating them
    """
    sta = srow.Station
    cll = cl.trdf[cl.trdf.Station==sta].iloc[0]
    odi = _makeEventListKey(srow.Events, cll.Events)
    inds = odi[:-1]
    cols = odi[1:]
    DFlag = cll.Lags.loc[inds, cols]
    DFlag.index = range(len(DFlag))
    DFlag.columns = range(1, len(DFlag) + 1)
    DFcc = cll.CCs.loc[inds, cols]
    DFcc.index = range(len(DFlag))
    DFcc.columns = range(1, len(DFlag) + 1)
    return DFcc, DFlag
    
    
    
def _makeEventListKey(evelist1, evelist2):
    """
    Make index key to make evelist1 to evelist2 
    """
    odi = [_fastWhere(x, evelist2) for num, x in enumerate(evelist1)]
    return odi
    
def _fastWhere(eve, objs):
    """
    like np.where but a bit faster because data checks are skipped
    """
    an = next(nu for nu, obj in enumerate(objs) if eve == obj)
    return an
    

def _getOffsetList(sind, srow, staSS):
    """
    get list of offsets and return it
    """
    return [staSS.loc[sind, 'Stats'][x]['offset'] for x in srow.Stats.keys()] 

def _updateStartTimes(srow, delayDF, temkey): 
    """
    Update the starttimes to reflect the values trimed in alignement
    """
    statsdict = srow.Stats
    sdo = srow.Stats
    for key in sdo.keys():
        temtemkey = temkey.loc[temkey.NAME == key].iloc[0]
        delaysamps = delayDF[delayDF.Events == key].iloc[0].SampleDelays
        Nc = sdo[key]['Nc']
        sr = sdo[key]['sampling_rate']
        stime = sdo[key]['starttime']
        
        stime_new = stime + delaysamps/(sr*Nc) # updated starttime to trim
        statsdict[key]['starttime'] = stime_new
        otime = obspy.UTCDateTime(temtemkey.TIME).timestamp #starttime 
        statsdict[key]['starttime'] = stime + delaysamps/(sr*Nc) 
        statsdict[key]['origintime'] = otime
        statsdict[key]['magnitude'] = temtemkey.MAG
        statsdict[key]['offset'] = stime_new - otime #predict offset time
    return statsdict


def _makeDFcclags(eventList, row, consisLen=True, 
                  subSamp=False):
    """
    Function to make correlation matrix and lag time matrix
    """
    cols = np.arange(1,len(eventList))
    indicies = np.arange(0,len(eventList)-1)
    DFcc = pd.DataFrame(columns=cols, index=indicies)
    DFlag = pd.DataFrame(columns=cols,index=indicies)
    
    # Loop over indicies and fill in cc and lags
    for b in DFcc.index.values:
        for c in range(b+1,len(DFcc)+1):
            rev = 1 #if order is switched make multiply lags by -1
            mptd1 = row.loc['MPtd'][eventList[b]]
            mptd2 = row.loc['MPtd'][eventList[c]]
            if consisLen: # if all templates same length use faster method
                mpfd1 = row.loc['MPfd'][eventList[b]]
                mpfd2 = row.loc['MPfd'][eventList[c]]
                Nc1 = row.loc['Channels'][eventList[b]]
                Nc2 = row.loc['Channels'][eventList[c]]
                maxcc, sampleLag = _CCX2(mpfd1, mpfd2, mptd1, mptd2, Nc1, Nc2, 
                                         subSamp=subSamp)
            # if all the templates are not the same length use slower method
            else: 
                maxlen = np.max([len(mptd1),len(mptd2)])
                if not len(mptd1) < maxlen: # reverse mptd1 is shorter
                    mptd1, mptd2 =mptd2 , mptd1
                    rev = -1
                lmp = len(mptd1)/2 # padding length
                mptd2 = np.pad(mptd2, (lmp,lmp), 'constant', 
                               constant_values=(0,0))
                cc = fast_normcorr(mptd1, mptd2)
                maxcc = cc.max()
                if subSamp:
                    sampleLag = _subSamp(cc) - len(mptd1)/2
                else:
                    sampleLag = cc.argmax() - len(mptd1)/2
            DFcc.loc[b, c] = maxcc
            DFlag.loc[b, c] = sampleLag
    return DFcc, DFlag * rev
    
def _subSamp(Ceval) :
    """ 
    Method to estimate subsample time delays using cosine-fit interpolation
    Cespedes, I., Huang, Y., Ophir, J. & Spratt, S. 
    Methods for estimation of sub-sample time delays of digitized echo signals. 
    Ultrason. Imaging 17, 142–171 (1995)
    """
    ind = Ceval.argmax()
    # If max occurs at start or end of CC no extrapolation
    if ind == 0 or ind == len(Ceval) - 1: 
        tau=float(ind)
    else:
        alpha = np.arccos((Ceval[ind-1] + Ceval[ind+1])/(2 * Ceval[ind]))
        tau=(-(np.arctan((Ceval[ind-1] - Ceval[ind+1]) / 
            (2 * Ceval[ind] * np.sin(alpha))) / alpha) + ind)
        if abs(tau - ind) > 1:
            msg = ('subsample failing, more than 1 sample shift predicted')
            detex.log(__name__, msg ,level='Warning', pri=True)
            return ind
    return tau
    
def _CCX2(mpfd1, mpfd2, mptd1, mptd2, Nc1, Nc2, subSamp=False):
    """
    Function find max correlation coeficient and corresponding lag time
    between 2 traces. fft should have previously been performed
    """
    if len(Nc1) != len(Nc2): #make sure there are the same number of channels
        msg = 'Number of Channels not equal, cannot perform correlation'
        detex.log(__name__, msg, level='error')
    Nc = len(Nc1) # Number of channels
    if len(mptd1) != len(mptd2) or len(mpfd2) !=len(mpfd1): 
        msg = 'Lengths not equal on multiplexed data, cannot correlate'
        detex.log(__name__, msg, level='error')
    n=len(mptd1)
    mptd2Temp = mptd2.copy()
    mptd2Temp = np.lib.pad(mptd2Temp, (n-1,n-1), 'constant', 
                           constant_values=(0,0))
    a = pd.rolling_mean(mptd2Temp, n)[n-1:]
    b = pd.rolling_std(mptd2Temp, n)[n-1:]
    b *= np.sqrt((n-1.0) / n)
    c = np.real(scipy.fftpack.ifft(np.multiply(np.conj(mpfd1), mpfd2)))    
    c1 = np.concatenate([c[-n+1:],c[:n]])
    result = ((c1 - mptd1.sum() * a) / (n*b*np.std(mptd1)))[Nc-1::Nc]
    try:
        maxcc = np.nanmax(result)
        maxind = np.nanargmax(result)
        if maxcc > 1.1: #if a inf is found in array
            result[result == np.inf] = 0
            maxcc = np.nanmax(result)
            maxind = np.nanargmax(result)
    except:
        return 0.0,0.0
        #deb([mpfd1,mpfd2,mptd1,mptd2,Nc1,Nc2])
    
    #return maxcc,maxind-n +1
    if subSamp:
        maxind = _subSamp(result)
    return maxcc,(maxind+1)*Nc - n
    
def fast_normcorr(t, s): 
    """
    fast normalized cc
    """
    if len(t)>len(s): #switch t and s if t is larger than s
        t, s = s, t
    n = len(t)
    nt = (t-np.mean(t))/(np.std(t)*n)
    sum_nt = nt.sum()
    a = pd.rolling_mean(s, n)[n-1:]
    b = pd.rolling_std(s, n)[n-1:]
    b *= np.sqrt((n-1.0) / n)
    c = np.convolve(nt[::-1], s, mode="valid")
    result = (c - sum_nt * a) / b    
    return result    
    
def _alignTD(delayDF, srow): 
    """
    loop through delay Df and apply offsets to create alligned 
    arrays dictionary
    """
    aligned = {}   
    #find the required length for each aligned stream 
    TDlengths = len(srow.MPtd[delayDF.Events[0]])-max(delayDF.SampleDelays) 
    for ind, row in delayDF.iterrows(): 
        orig = srow.MPtd[row.Events] 
        orig = orig[row.SampleDelays:]
        orig = orig[:TDlengths]
        aligned[row.Events] = orig 
        if len(orig) == 0:
            msg = ('Alignment of multiplexed stream failing, try raising ccreq'
                    'of input cluster object')
            detex.log(__name__, msg, level='error')
        
    return aligned

def _makeSingleEventDict(cl, TRDF, temkey): 
    """
    Make dict of dataframes for singles on each station
    """
    singlesdict={}
    cols = [x for x in TRDF.columns if not x in ['Clust','Link','Lags','CCs']]
    for num, row in TRDF.iterrows():
        singleslist = [0]*len(cl[row.Station].singles) #init list
        DF=pd.DataFrame(index=xrange(len(singleslist)), columns=cols)
        if len(singleslist) < 1: # if no singles on this channel
            break
        DF['Name'] = str
        DF['Offsets'] = list
        for sn, sing in enumerate(singleslist):
            #DF.Events[a[0]]=evlist\
            evelist = [cl[row.Station].singles[sn]]
            temtemkey = temkey.loc[temkey.NAME == evelist[0]].iloc[0]
            DF["Station"][sn] = row.Station
            DF["MPtd"][sn] = _trimDict(row, 'MPtd', evelist)
            DF["MPfd"][sn] = _trimDict(row, 'MPfd', evelist)
            DF["Stats"][sn] = _trimDict(row, 'Stats', evelist)
            DF["Channels"][sn] = _trimDict(row, 'Channels', evelist)
            otime = obspy.UTCDateTime(temtemkey.TIME).timestamp
            stime =  DF.Stats[sn][evelist[0]]['starttime']
            DF.Stats[sn][evelist[0]]['origintime'] = otime
            
            DF.Stats[sn][evelist[0]]['offset'] = stime - otime
            DF.Stats[sn][evelist[0]]['magnitude'] = temtemkey.MAG
            DF.Events[sn] = DF.MPtd[sn].keys()
            DF.Name[sn] = 'SG%d' % sn
        DF['SampleTrims'] = [{} for x in range(len(DF))]
        DF['FAS'] = object
        DF['Threshold'] = object
        singlesdict[row.Station] = DF
    return singlesdict

def _makeSSDF(row, minEvents):
    """
    Function to change form of TRDF for subpace creation
    """
    index = range(len(row.Clust))
    columns = [x for x in row.index if x!='Clust']
    DF = pd.DataFrame(index=index, columns=columns)
    DF['Name'] = ['SS%d' % x for x in range(len(DF))] # name subspaces
    # Initialize columns for future use
    DF['Events'] = object
    DF['AlignedTD'] = object
    DF['SVD'] = object
    DF['UsedSVDKeys'] = object
    DF['FracEnergy'] = object
    DF['SVDdefined'] = False
    DF['SampleTrims'] = [{} for x in range(len(DF))]  
    DF['Threshold'] = np.float
    DF['SigDimRep'] = object
    DF['FAS'] = object
    DF['NumBasis'] = int
    DF['Offsets'] = object
    DF['Stats'] = object
    DF['MPtd'] = object
    DF['MPfd'] = object
    DF['Channels'] = object
    DF['Station'] = row.Station
    DF = DF.astype(object)
    for ind, row2 in DF.iterrows():
        evelist = row.Clust[ind]
        evelist.sort()
        DF['Events'][ind] = evelist
        DF['MPtd'][ind] = _trimDict(row, 'MPtd', evelist)
        DF['MPfd'][ind] = _trimDict(row, 'MPfd', evelist)
        DF['Stats'][ind] = _trimDict(row, 'Stats', evelist)
        DF['Channels'][ind] = _trimDict(row, 'Channels', evelist)
    #only keep subspaces that meet min req, dont renumber
    DF = DF[[len(x) >= minEvents for x in DF.Events]] 
    #DF.reset_index(drop=True, inplace=True)
    return DF

def _trimDict(row, column, evelist): 
    """
    function used to get only desired values form dictionary 
    """
    temdict = {k: row[column].get(k,None) for k in evelist}
    dictout = {k: v for k, v in temdict.items() if not v is None}
    return dictout

############## Shared Subspace and Cluster Functions #####################
        
def _loadEvents(fetcher, filt, trim, stakey, temkey,
                decimate, dtype, enforceOrigin=False):
    """
    Initialize TRDF, a container for a great many things including 
    event templates, multiplexed data, obspy traces etc.   
    """
    columns = ['Events', 'MPtd', 'MPfd', 'Channels', 'Stats', 'Link', 'Clust',
               'Lags', 'CCs', 'numEvents']
    TRDF = pd.DataFrame(columns=columns)
    stanets = stakey.NETWORK + '.' + stakey.STATION
    TRDF['Station'] = stanets
    TRDF['Keep'] = True
    TRDF = TRDF.astype(object) # cast as objects so pandas can be abused ;)
    
    #Make list in data frame that shows which stations are in each event
    # Load streams into dataframe to call later
    for ind, row in TRDF.iterrows():
        TRDF['MPtd'][ind] = {}
        TRDF['MPfd'][ind] = {}  
        sts, eves, chans, stats = _loadStream(fetcher, filt, trim, decimate,
                                              row.Station, dtype, temkey, 
                                              stakey, enforceOrigin)                  
        TRDF['Events'][ind] = eves
        TRDF['Channels'][ind] = chans
        TRDF['Stats'][ind] = stats
        # Make sure some events are good for given station
        if not isinstance(TRDF['Events'][ind], list):
            TRDF.loc[ind, 'Keep'] = False
            continue
        TRDF.loc[ind, 'numEvents'] = len(TRDF.loc[ind, 'Events'])
        
        # get multiplexed time domain and freq. domain arrays
        for key in TRDF.loc[ind, 'Events']: # loop each event     
            Nc = TRDF.loc[ind, 'Stats'][key]['Nc']
            mp = multiplex(sts[key], Nc) # multiplexed time domain
            st = sts[key] # current stream
            TRDF.loc[ind, 'MPtd'][key] = mp
            stu = st[0].stats.starttime.timestamp #updated starttime
            TRDF.loc[ind, 'Stats']['starttime'] = stu
            reqlen = 2 * len(TRDF.loc[ind,'MPtd'][key]) #required length 
            reqlenbits = 2**reqlen.bit_length() # required length fd
            mpfd = scipy.fftpack.fft(mp, n=reqlenbits)
            TRDF.loc[ind, 'MPfd'][key] = mpfd
    TRDF = TRDF[TRDF.Keep]
    TRDF.sort(columns='Station', inplace=True)
    TRDF.reset_index(inplace=True, drop=True) 
    return TRDF
    
def _testStreamLengths(TRDF):
    for ind, row in TRDF.iterrows(): #get lengths
        lens = np.array([len(x) for x in row.MPtd.values()])
    # trim to smallest length if within 90% of median, else kill key
    le = np.min(lens[lens > np.median(lens)*.9])
    for ind, row in TRDF.iterrows():
        keysToKill = [x for x in row.Events if len(row.MPtd[x]) < le]
        # trim events slightly too small if any
        for key in row.Events:
            trimed = row.MPtd[key][:le]
            TRDF.loc[ind, 'MPtd'][key] = trimed
        #rest keys on TRDF
        tmar = np.array(TRDF.Events[ind])
        tk = [not x in keysToKill for x in TRDF.Events[ind]]
        TRDF.Events[ind] = tmar[np.array(tk)]
        for key in keysToKill:
            msg = ('%s on %s is out of length tolerance, removing' % 
                  (key, row.Station))
            detex.log(__name__, msg, level='warn', pri=True)
            TRDF.MPtd[ind].pop(key,None)
            TRDF.MPfd[ind].pop(key,None)
    return TRDF

def _flatNoNan(df):
    """
    Take a dataframe of pure numerics, flatten into 1d array and remove NaNs
    """
    df.fillna(np.NAN, inplace=True)
    ar = df.values.flatten()
    return ar[~np.isnan(ar)]
    
def _getDelays(link, CCtoLag, cx, lags, cxdf):
    N = len(link) # Lumber of events
    #append cluster numbers to link array
    linkup = np.append(link,np.arange(N+1, 2*N+1).reshape(N,1),1) 
    clustDict = _getClustDict(linkup, len(linkup))
    cols = ['i1', 'i2', 'cc', 'num', 'clust']
    dflink = pd.DataFrame(linkup, columns=cols).astype(object)
    if len(dflink) > 0:
        dflink['II'] = list
    dflink['ev1'] = 0
    dflink['ev2'] = 0
    dflink = dflink.astype(object, copy=False)
    for ind, row in dflink.iterrows():
        ii1 = clustDict[int(row.i1)].tolist()
        ii2 = clustDict[int(row.i2)].tolist()
        dflink.loc[ind, 'II'] = ii1 + ii2
        #get a dataframe with only index as ev1 and column as ev2
        tempdf = cxdf[cxdf == row.cc].dropna(how='all').dropna(axis=1) 
        dflink.loc[ind, 'ev1'] = tempdf.index[0]
        dflink.loc[ind, 'ev2'] = tempdf.columns[0]   
    lags = _traceEventDendro(dflink, cx, lags, CCtoLag, clustDict, 
                             clustDict.iloc[-1])
    return lags
    #return clusts,lagByClust   
        
def _traceEventDendro(dflink, x, lags, CCtoLag, clustDict, clus):
    """
    Function to follow ind1 through clustering linkage in linkup 
    and return total offset time (used for alignment)
    """
    if len(dflink)<1: #if event is singleton return zero offset time
        lagSeries=pd.Series([0],index=clus)
    else:       
        allevents = dflink['II'][dflink.index.values.max()] 
        allevents.sort()
        #total lags for each station
        lagSeries = pd.Series([0]*(len(dflink) + 1), index=allevents) 
        for a in dflink.iterrows():
            if a[1].ev1 in clustDict[int(a[1].i1)]:
                cl22 = clustDict[int(a[1].i2)]
            else:
                cl22 = clustDict[int(a[1].i1)]
            currentLag = CCtoLag[a[1].cc]
            for b in cl22: #record and update lags for second cluster                
                lagSeries[b] += currentLag # 
                lags = _updateLags(b, lags, len(dflink), currentLag)
            CCtoLag = _makeLagSeries(x,lags)
    return lagSeries
        
        
def _updateLags(evenum, lags, N, currentLag):
    """
    function to add current lag shifts to effected 
    lag times (see Haris 2006 appendix B)
    """
    dow = _getDow(N, evenum) #get the index to add to lags for columns
    acr = _getAcr(N, evenum)
    for a in acr:
        lags[a] += currentLag
    for a in dow:
        lags[a] -= currentLag
    return lags

def _getDow(N, evenum):
    dow = [0]*evenum
    if len(dow) > 0:
        for a in range(len(dow)):
            dow[a] = _triangular(N-1)-1+evenum-_triangular(N-1-a)
    return dow
    
def _getAcr(N, evenum):
    acr = [0]*(N-evenum)
    if len(acr) > 0:
        acr[0] = _triangular(N)-_triangular(N-(evenum))
        for a in range(1, len(acr)):
            acr[a] = acr[a-1]+1
    return acr

def _triangular(n): 
    """
    calculate sum of triangle with base N, 
    see http://en.wikipedia.org/wiki/Triangular_number
    """

    return sum(range(n+1))

def _getClustDict(linkup,N): 
    """
    get pd series that will define the base events in each cluster 
    (including intermediate clusters)
    """
    inds = np.arange(0,N+1)
    content = [np.array([x]) for x in np.arange(0, N+1)]
    clusdict = pd.Series(content, index=inds) 
    for a in range(len(linkup)):
        ind1 = int(linkup[a,4])
        ind2 = int(linkup[a,0])
        ind3 = int(linkup[a,1])
        clusdict[ind1] = np.append(clusdict[ind2], clusdict[ind3])
    return clusdict       
    
def _ensureUnique(cx, cxdf): 
    """
    Make sure each coeficient is unique so it can be used as a key to 
    reference time lags, if not unique perturb slightly
    """
    se = pd.Series(cx)
    dups = se[se.duplicated()]
    count = 0
    while len(dups)>0:
        msg = ('Duplicates found in correlation coefficients,'
               'perturbing slightly to get unique values')
        detex.log(__name__, msg, level='warning', pri=True)
        for a in dups.iteritems():
            se[a[0]] = a[1] - abs(.00001*np.random.rand())
        count += 1
        dups = se[se.duplicated()]
        if count>10:
            msg = 'cannot make Coeficients unique, killing program'
            detex.log(__name__, msg, level='error')
    if count > 1: # if the cx has been perturbed update cxdf
        for a in range(len(cxdf)):
            sindex = sum(pd.isnull(a[1]))
            tri1 = _triangular(len(cxdf))
            tri2 = _triangular(len(cxdf)-a)
            tri3 = _triangular(len(cxdf)-(a+1))
            cxdf.values[a, sindex:] = cx[tri1 - tri2 , tri1 - tri3]       
    return se.values,cxdf
    
def _makeLagSeries(x, lags):
    LS = pd.Series(lags, index=x)
    return LS        
    
def _loadStream(fetcher, filt, trim, decimate, station, dtype,
                temkey, stakey, enforceOrigin=False): 
    """
    loads all traces into stream object and applies filters and trims
    """
    StreamDict = {} # Initialize dictionary for stream objects
    channelDict = {}
    stats = {}
    STlens = {}
    trLen = [] # trace length
    allzeros = [] # empty list to stuff all zero keys to remove later
    csta = stakey[stakey.STATION == station.split('.')[1]]
    
    # load waveforms
    for st, evename in fetcher.getTemData(temkey, csta, trim[0], trim[1], 
                                          returnName=True):
        if st is None or len(st) < 1:
            continue #skip if stream empty
        st = _applyFilter(st, filt, decimate, dtype)
        tem = temkey[temkey.NAME == evename]
        if len(tem) < 1:
            msg = '%s not in template key'
            detex.log(__name__, msg, pri=True)
            continue
        originTime = obspy.UTCDateTime(tem.iloc[0].TIME)
        Nc = detex.util.get_number_channels(st) #get number of channels
        if Nc != len(st): 
            msg = ('%s on %s is fractured or channels are missing' % 
                  (evename, station))
            detex.log(__name__, msg, pri=True)
            continue
        if enforceOrigin: #if the waveforms should start at the origin
            st.trim(starttime=originTime, pad=True, fill_value=0.0)
        StreamDict[evename] = st
        channelDict[evename] = [x.stats.channel for x in st]
        pros = st[0].stats['processing']
        sr = st[0].stats.sampling_rate
        start = st[0].stats.starttime.timestamp
        statsdi = {'processing':pros, 'sampling_rate':sr, 'starttime':start, 
                   'Nc':Nc}
        stats[evename] = statsdi # put stats in dict
        totalLength = np.sum([len(x) for x in st])
        if any([not np.any(x.data) for x in st]):
            allzeros.append(evename)
        trLen.append(totalLength)
        STlens[evename] = totalLength
    
    mlen = np.median(trLen)
    keysToRemove = [x for x in StreamDict.keys() if STlens[x] < mlen*.2]
    for key in keysToRemove: # Remove fractured or very short waveforms
        msg = '%s is fractured or missing data, removing' % key
        detex.log(__name__, msg, level='warning', pri=True)
        StreamDict.pop(key,None)
        channelDict.pop(key,None)
        stats.pop(key,None)
    for key in set(allzeros): # remove waveforms with channels filled with 0s
        msg = '%s has at least one channel that is all zeros, deleting' %key
        detex.log(__name__, msg, level='warning', pri=True)
        StreamDict.pop(key,None)
        channelDict.pop(key,None)
        stats.pop(key,None)
#    # recalc lengths, trim to smallest to avoid 1 sample different rejects
#    trLen = []
#    for key, st in StreamDict.iteritems():
#        for x in st:
#            trLen.append(len(x.data))
    
    
    if len(StreamDict.keys())<2:
        msg = ('Less than 2 events survived preprocessing for station' 
               '%s Check input parameters, especially trim' % station)
        detex.log(__name__, msg, level='warning', pri=True)        
        return None,None,None,None
    evlist = StreamDict.keys()
    evlist.sort()
    #if 'IMU' in station:
    #    deb(evlist)
    return StreamDict, evlist, channelDict, stats

def multiplex(st, Nc=None, trimTolerance=15, template=False, returnlist=False, 
              retst=False):
    """
    Multiplex an obspy stream object
    Parameters
    ----------
    st : instance of obspy stream
        The stream containing the data to multiplex. 
    Nc : None or int
        if not None the number of channels in stream, else try to determine
    trimTolerance : int
        The number of samples each channel can vary before being rejected
    Template : bool
        If True st is a template waveform, therefore an exception will be 
        raised if trimeTolerance is exceeded
    returnlist : bool
        If true also return np array of un-multiplexed data as a list
    
    Returns
    ------
    list with multiplexed data and other desired waveforms
    """
    if Nc is None:
        Nc = len(set([x.stats.station for x in st]))
    if Nc == 1: # If only one channel do nothing
        C1 = st[0].data
        C = st[0].data
   
    else:
        chans = [x.data for x in st] # Data on each channel
        minlen=np.array([len(x) for x in chans])  
        if max(minlen)-min(minlen) > trimTolerance:
            netsta = st[0].stats.network + '.' + st[0].stats.station
            utc1 = st[0].stats.starttime.formatIRISWebService().split('.')[0]
            utc2 = st[0].stats.endtime.formatIRISWebService().split('.')[0]
            msg = ('Channel lengths are not within %d on %s from %s to %s' %
                  (trimTolerance, netsta, utc1, utc2))
            if template:
                detex.log(__name__, msg, level='error')
            else:
                msg = msg + ' trimming to shortest channel'
                detex.log(__name__, msg, level='warning', pri=True)
                trimDim = min(minlen) #trim to smalles dimension
                chansTrimed = [x[:trimDim] for x in chans] 
        elif max(minlen) - min(minlen)>0 : #if all channels not equal lengths
            trimDim = min(minlen)
            chansTrimed = [x[:trimDim] for x in chans] # trim to shortest
        elif max(minlen) - min(minlen) == 0: #if chan lengths are exactly equal
            chansTrimed = chans
        C = np.vstack((chansTrimed))
        C1 = np.ndarray.flatten(C, order='F')
    out = [C1] # init output list
    if returnlist:
        out.append(C)
    if retst:
        out.append(st)
    if len(out)==1:
        return out[0]
    else:
        return out
        

def _applyFilter(st, filt, decimate=False, dtype='double', fillZeros=False):
    """
    Apply a filter, decimate, and trim to even start/end times 
    """
    st.sort()
    if dtype == 'single': #cast into single
        for num,tr in enumerate(st):
            st[num].data = tr.data.astype(np.float32)
    Nc = list(set([x.stats.channel for x in st]))
    if len(st) > len(Nc): #if data is fragmented only keep largest chunk
        if fillZeros:
            st = _mergeChannelsFill(st)
        else:
            st = _mergeChannels(st)
    if decimate:
        st.decimate(decimate)
        
    startTrim = max([x.stats.starttime for x in st])
    endTrim = min([x.stats.endtime for x in st])
    
    st.trim(starttime=startTrim, endtime=endTrim)
    st = st.split()
    st.detrend('linear')
    if isinstance(filt,list) or isinstance(filt,tuple):
        st.filter('bandpass', freqmin=filt[0] ,freqmax=filt[1], 
                  corners=filt[2], zerophase=filt[3])
    return st
    
def _mergeChannels(st): 
    """
    function to find longest continuous data chucnk and discard the rest
    """
    channels = list(set([x.stats.channel for x in st]))
    temst = st.select(channel=channels[0])
    lengths = np.array([len(x.data) for x in temst])
    lemax = lengths.argmax()
    sttime = st[lemax].stats.starttime
    etime = st[lemax].stats.endtime
    st.trim(starttime=sttime, endtime=etime)
    return st

def _mergeChannelsFill(st):
    st.merge(fill_value=0.0)
    return st




def deb(varlist):
    global de
    de=varlist
    sys.exit(1)   