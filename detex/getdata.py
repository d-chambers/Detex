from obspy.fdsn import Client
import os, sys, glob, obspy, numpy, pandas as pd, detex, joblib, warnings, numpy as np
from multiprocessing import Pool
# Use OBSPY to import data from IRIS

def makeTemplatKey(catalog,filename='TemplateKey.csv',save=True):
    """
    Function to get build the Detex required file TemplateKey.csv from an obspy catalog object,
    or list of obspy catalog obhjects
   
    ::Append: if true a new TemplateKey.csv is not created but the exisiting is appended
    A new one is created if the old one does not exist. 
    
    obspy catalog object docs at:
    http://docs.obspy.org/packages/autogen/obspy.fdsn.client.Client.get_events.html#obspy.fdsn.client.Client.get_events
    """
    if not isinstance(catalog,list or tuple): #make sure input is a list
        catalog=[catalog]
    lats = []
    lons = []
    depths=[]
    mags = []
    names = []
    time = []
    author=[]
    magtypes = []
    for cat in catalog:
        if not isinstance(cat,obspy.core.event.Catalog):
            raise Exception ('None catalog object found, examin imput data')
        for event in cat:
            if not event.origins:
                msg = ("Event '%s' does not have an origin and will not be included." % str(event.resource_id))
                warnings.warn(msg)
                continue
            if not event.magnitudes:
                msg = ("Event '%s' does not have a magnitude but will still be inclused" % str(event.resource_id))
                warnings.warn(msg)
            origin = event.preferred_origin() or event.origins[0]
            lats.append(origin.latitude)
            lons.append(origin.longitude)
            depths.append(origin.depth/1000.0)
            tim=origin.time.formatIRISWebService().replace(':','-')
            time.append(tim)
            names.append(tim.split('.')[0])
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            mags.append(magnitude.mag)
            magtypes.append(magnitude.magnitude_type)
            author.append(origin.creation_info.author)
    columnnames=['NAME','TIME','LAT','LON','DEPTH','MAG','MTYPE','CONTRIBUTOR']
    DF=pd.DataFrame(np.transpose([names,time,lats,lons,depths,mags,magtypes,author]),columns=columnnames)
    DF['STATIONKEY']='StationKey.csv'
    if save:
        DF.to_csv(filename)
    return DF
   

def getAllData(templateKeyPath='TemplateKey.csv',cli='IRIS',formatout='pickle',removeResponse=True,
               prefilt=[.05,.1,15,20],loc='*',templateDir='EventWaveForms',timeBeforeOrigin=3*60,timeAfterOrigin=9*60,
                opType='vel',ConDir='ContinousWaveForms',secBuf=120,multiPro=False,getContinous=1,getTemplates=1,
                gapFile='TimeGaps.pkl', reverse=False):
                    
    """ 

    
    Function designed to fetch data needed for detex waveform or subspace detection from IRIS, not yet tested for other clients.
    Downloads template waveforms and continous waveforms defined by the station and template keys (see example for format)
    that do not currently exits and orginizes ContiousWaveForms and EventWaveForms directories

    Parameters
    ------------    
    
    templateKeyPath : str
        TemplateKeyPath: a path to the TemplateKey csv\n
    cli : str
        String for obspy.fdsn.Client.get_waveforms function, see obspy docs for other options besides IRIS\n
        ..
    formatout : str
        Seismic data file format. If you are using other modules of detex the pickle format is required.
        (pickle serializes an obspy stream object)\n
    removeResponse : boolean (True, False, 1, 0)
        Flags if the instrument response is to be removed for each downloaded trace\n
    prefilt : list containing real numbers (float or int)
        A list containig the pre filter to apply to stabalize the instrument response deconvolution,  
        4 corners must be provided in Hz, response is flat between corners 2 and 3.\n
    loc : str
        the location field required by IRIS for requesting data. '*' is wildcard. Most stations have
        only one location code for a given time so the wildcard should work almost always\n
    tempalateDir : str
        The name of the template directory. Using the default is recommended else the templateDir parameter
        will have to be set in calling most other detex functions\n
    timeBeforeOrigin: real number (int, float, etc.)
        The time in seconds before the origin of each tempalte that is downloaded. Note: The templates
        should be trimmed before correlating or running subspace detection (they are by default very long)\n
    timeAfterOrigin : real number(int,float,etc.)
        The time in seconds to download after the origin time of each template.\n 
    opType : str
        Output type aftering removing instrument response. "DISP" (m), "VEL" (m/s), or "ACC" (m/s**2)\n     
    ConDir : str
        The name of the continous waveform directory. Using the default is recommended\n
    secBuf : real number (int, float, etc.)
        The number of seconds to download after each hour of continous data. This might be non-zero in order to capture
        some detections that would normally be overlooked if data did not overlap. \n
    """
    detex.util.checkExists(templateKeyPath)
    TK=pd.read_csv(templateKeyPath) # read tempalte keys
    if getTemplates:
        print 'Gettting template waveforms'
        _getTemData(TK,templateDir,formatout,removeResponse,prefilt,cli,timeBeforeOrigin,timeAfterOrigin,loc,opType) # get event data
    if getContinous:
       print 'Getting continous data, this may take several hours depending on request size'
       _getConData(TK,loc,ConDir,cli,formatout,removeResponse,prefilt,secBuf,opType,multiPro,reverse)
                    
            
def _getTemData (TK,templateDir,formatout,removeResponse,prefilt,client,timeBeforeOrigin,timeAfterOrigin,loc,opType):
    HD=os.getcwd() # Get current directory 
    client=Client(client)
    for tk in TK.iterrows(): # iterate through each template key row
        eventID=tk[1]['NAME']
        SK=pd.read_csv(tk[1]['STATIONKEY'])
        SK=SK[[not np.isnan(x) for x in SK.LAT]]
        try:
            oTime=obspy.core.UTCDateTime(tk[1]['TIME']) # origin time
        except:
            raise Exception('%s is a bad entry in TIME column for event %s' %(tk[1]['TIME'],tk[1]['NAME']) )
        utcStart=oTime-timeBeforeOrigin # start time is event origin time minus timeBeforeOrigin
        utcEnd=oTime+timeAfterOrigin  # end time is event origin time plus timeafterOrigin
        for sk in SK.iterrows(): # iterate through each row of station keys
            chans=sk[1]['CHANNELS'].replace('-',',')
            net=sk[1]['NETWORK']
            sta=sk[1]['STATION']
            st=True
            UTCstr='%04d-%03dT%02d-%02d-%02d'%(oTime.year,oTime.julday,oTime.hour,oTime.minute,oTime.second)
            sdir=os.path.join(HD,templateDir,eventID).replace(' ','')
            svar=('%s.%s.%s.pkl' %(net,sta,UTCstr)).replace(' ','')
            if os.path.isfile(os.path.join(sdir,svar)): #IF file aready exits skip process
                st=False
            if st != False:
                st=_tryDownloadData(net,sta,chans,loc,utcStart,utcEnd,client)
                if st != False: #_tryDownloadData can return false, second st check is needed
                    if tk[0] % 25 == 0 :
                        print ('%d events downloaded out of %d for Station %s' %(tk[0]+1,len(TK),sk[1].STATION))
                    if removeResponse==True:
                        st=_removeInstrumentResposne(st,prefilt,opType)
                    if st != False:
                        if not os.path.isdir(sdir): # Create Waveform sub directory if it does not exis
                            os.makedirs(sdir)
                        st.write(os.path.join(sdir,svar),formatout)
                    #print ('wrtiing %s in %s failed' %(svar,sdir))
    
def _getConData(TK,loc,ConDir,client,formatout,removeResponse,prefilt,secBuf,opType,multiPro,reverse):
    HD=os.getcwd() # Get current directory   
    client=Client(client)
    stationKeyPaths=list(set(TK.STATIONKEY.values)) # Get unique station key files from all templates
    for kpath in stationKeyPaths: #iterate through all unique station keys
        SK=pd.read_csv(kpath)
        if multiPro==True: #This is still broken
            sklist=[None]*len(SK)
            for sk in SK.iterrows():
                sklist[sk[0]]=sk,HD,ConDir,removeResponse,prefilt,opType,formatout,secBuf,loc,client
            joblib.Parallel(n_jobs=len(sklist))(joblib.delayed(_getConDataStation)(i)for i in sklist)
        else:
            for sk in SK.iterrows(): # iterate through eaceh station in the current key
                _getConDataStation(sk,HD,ConDir,removeResponse,prefilt,opType,formatout,secBuf,loc,client,reverse)
        
def _getConDataStation(sk,HD,ConDir,removeResponse,prefilt,opType,formatout,secBuf,loc,client,reverse):
    chans=sk[1]['CHANNELS'].replace('-',',')
    net=sk[1]['NETWORK']
    sta=sk[1]['STATION']
    utcStart=obspy.core.UTCDateTime(sk[1]['STARTTIME'])
    utcStart=utcStart-utcStart.timestamp%3600 # get time to nearest hour
    utcEnd=obspy.core.UTCDateTime(sk[1]['ENDTIME'])
    utcEnd=utcEnd-utcEnd.timestamp%3600 # get time to nearest hour
    timeArray=[utcStart + x*3600 for x in range(int(utcEnd.timestamp-utcStart.timestamp)/3600+1)] #get array with start stop times by hour
    if reverse:
        timeArray=timeArray[::-1]
    for t in range(len(timeArray)-1):
        if reverse:
            oTime=timeArray[t+1]
        else:
            oTime=timeArray[t]
        st=True
        UTCstr='%04d-%03dT%02d'%(oTime.year,oTime.julday,oTime.hour)
        sdir=os.path.join(HD,ConDir,net+'.'+sta,str(oTime.year),"%03d" % (oTime.julday)) # Save directory for current loop
        svar=('%s.%s.%s.pkl' %(net,sta,UTCstr))
        if os.path.isfile(os.path.join(sdir,svar)): #IF file aready exits skip process
            st=False                
        if st != False:
            if reverse:
                st=_tryDownloadData(net,sta,chans,loc,timeArray[t+1],timeArray[t]+secBuf,client)  
            else:
                st=_tryDownloadData(net,sta,chans,loc,timeArray[t],timeArray[t+1]+secBuf,client)
            if st != False:
                if removeResponse==True:
                    st=_removeInstrumentResposne(st,prefilt,opType)
                if not os.path.isdir(sdir): # Create Waveform sub directory if it does not exist
                    os.makedirs(sdir)
                if st != False:
                    try:
                        st.write(os.path.join(sdir,svar),formatout)
                    except:
                        print ('wrtiing %s in %s failed' %(svar,sdir))
                  
        
def _removeInstrumentResposne(st,prefilt,opType):
    st.detrend('linear')# detrend
    st= _fftprep(st)
    try: 
        st.remove_response(output=opType,pre_filt=prefilt)
    except:
        print ('RemoveResponse Failed for %s,%s, not saving' %(st[0].stats.network,st[0].stats.station))
        st=False
    return st

def _fftprep(st):
    data=st[0].data
    "data is numpy vector, makes sure it is not of odd length or fft drags"
    if len(data) % 2 !=0 and len(data) % 100 > 50:
        data=numpy.insert(data,0,data[0])
        st[0].data=data
        st[0].stats.starttime=st[0].stats.starttime-st[0].stats.delta
    elif len(data) % 2 !=0 and len(data) % 100 < 50:
        data=data[1:]
        st[0].data=data
        st[0].stats.starttime=st[0].stats.starttime+st[0].stats.delta
    return st
          
def _tryDownloadData(net,sta,chan,loc, utcStart,utcEnd,client): # get data, return False if fail
    try:
        st=client.get_waveforms(net,sta,loc,chan,utcStart,utcEnd,attach_response=True)
        return st
    except:
        print ('Download failed for %s.%s %s from %s to %s' % (net,sta,chan,str(utcStart),str(utcEnd)))
        return False      

            

#
#def getevecondata(UTCstart,UTCend,WaveFormDir='EventWaveForms'):
#    """ 
#    function to get the required continous data for each event in EventWaveForms without redundancy
#    """
#    HD=os.getcwd()
#    Events=glob.glob(os.path.join(HD,WaveFormDir,'*'))
#    for eve in Events:
#        instruments=glob.glob(os.path.join(eve,'*'))
#        for inst in instruments:
#            network=os.path.basename(inst).split('.')[0]
#            ins=os.path.basename(inst).split('.')[1]
#            chas=[]
#            Eventdirs=glob.glob(os.path.join(inst,'*'))
#            for event in Eventdirs: #find all chanels recorded in Event files
#                channels=glob.glob(os.path.join(event,'*'))
#                cta=[os.path.basename(x).split('.')[2].split('_')[0] for x in channels]
#                chas.extend(cta)
#            chans=list(set(chas))
#            for chan in chans:
#                _getConData(network,ins,UTCstart,UTCend)
#        #Tstamps=[obspy.core.UTCDateTime(os.path.basename(x)).timestamp for x in Eventdirs]            
            
            
def deb(varlist):
    global de
    de=varlist
    sys.exit(1)                    
            
            
            
            
            
            
            
        