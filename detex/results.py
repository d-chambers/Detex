# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:55:01 2014

@author: Derrick
"""
import os, glob, numpy as np, obspy, matplotlib.pyplot as plt, detex, pandas as pd, sqlite3, sys
import pandas.io.sql as psql, numbers, scipy

def deb(varlist):
    global de
    de=varlist
    sys.exit(1)                   


def detResults(trigCon=0,trigParameter=0,associateReq=0,ss_associateBuffer=1,sg_associateBuffer=2.5,requiredNumStations=4,veriBuffer=1,ssDB='SubSpace.db',
              templatePath='EventWaveForms',condir='ContinuousWaveForms',templateKey='TemplateKey.csv',stationKey='StationKey.csv',
              veriFile=None,includeAllVeriColumns=True,reduceDets=True,Pf=False,Stations=None,
              starttime=None,endtime=None):
    """
    Function to create an instance of the CorResults class. Used to associate detections across multiple stations into 
    coherent events. CorResults class also has some useful methods for creating input files for location
    programs such as hypoDD and hypoInverse
    
    Parameters
    ---------
    trigCon : int 0 or 1
        trigCon is used for filtering the events in the corDB. Currently only options are 0, raw detection statistic value
        or 1, STA/LTA value of DS as reported in the corDB. This parameter might be useful if there are a large number of detections
        in corDB with low STA/LTA or correlation coeficient values. 
    trigParameter : number
        if trigCon==0 trigParameter is the minimum correlation for a detection to be loaded from the CorDB
        if trigCon==1 trigParameter is the minimum STA/LTA value for a detection to be loaded from the CorDB
        Regardless of trigCon if trigParameter == 0 all detections in CorDB will be loaded and processed
    associateReq : int
        The association requirement which is the minimum number of events that must be shared by subspaces in order to permit event association.
        For example, subspace 0 (SS0) on station 1 was created using events A,B,C. subspace 0 (SS0) on station 2 was created using events
        C, D. If both subspace share a detection and associateReq=0 or 1 the detections will be associated into a coherent event. If 
        assiciateReq == 2, however, the detections will not be associated. 
    ss_associateBuffer : real number (int or float)
        The buffertime applied to subspace event assocaition in seconds
    sg_associateBuffer : real number (int or float)
        The buffertime applied to singleton event assocaition in seconds  
    requiredNumStations : int
        The required number of a stations on which a detection must occur in order to be classified as an event        
    veriBuffer : real number (int, float)
        Same as associate buffer but for associating detections with events in the verification file
    ssDB : str
        Path the the database created by detex.xcorr.correlate function
    templatePath : str
        Path to the tempalte directory
    condir : str
        Path to the continous data directory
    templateKey : str
        Path to the template key
    stationKey : str
        Path to template key
    veriFile : None (not used) or str (optional)
        Path to a file in the TemplateKey format for verifying the detection by origin time association. veriFile can either be an 
        sqlite database with table name 'verify', a CSV, or a pickled pandas dataframe. The following fields must be present:
        'TIME','NAME','LAT','LON','MAG'. Any additional columns may be present with any name that will be included in the verification
        dataframe depending on includeAllVeriColumns
    includeAllVeriColumns: boolean
        If true include all columns that are in the veriFile in the verify dataframe that will be part of the SSResults object.
    reduceDets : boolean
        If true loop over each station and delete detections of the same event that don't have the highest detection Stat value.
        It is recommended this be left as True unless there a specific reason lower DS detections are wanted
    Pf : float or False
        The probability of false detection accepted. Uses the pickled subspace object to get teh FAS class and corresponding
        fitted beta in order to only keep detections with thresholds above some Pf, defiend for each subspace station pair
        If used values of 10**-8 and greater generally work well
    Stations : list of str or None
        If not None, a list or tuple of stations to be used in the associations. All others will be discarded
    starttime : None or obspy.UTCDateTime readable object (time stamp or data string)
        If not None, then sets a lower bound on the MSTAMPmin column loaded from the dataframe
    endtime : None or obspy.UTCDateTime readable object (time stamp or data string)
        If not None, then sets a lower bound on the MSTAMPmin column loaded from the dataframe
            
    """
    ### Make sure all inputs exist and are kosher
    _checkExistence([ssDB,templateKey,stationKey]) #make sure all required things exist
    _checkInputs(trigCon,trigParameter,associateReq,ss_associateBuffer,requiredNumStations)
    if associateReq != 0:
        raise Exception('associateReq values other than 0 not yet supported') #TODO: implement this in a way that isnt cripplingly slow
    
    ### Try to read in all input files and dataframes needed in the 
    temkey=pd.read_csv(templateKey) #load template key
    stakey=pd.read_csv(stationKey)#load station key
    
    ss_info,sg_info=_loadInfoDataFrames(ssDB) # load info DataFrames

    ss_hist=detex.util.loadSQLite(ssDB,'ss_hist')  # load histograms
    sg_hist=detex.util.loadSQLite(ssDB,'sg_hist')  

    filt=detex.util.loadSQLite(ssDB,'filt_params') #load filter Parameters
    
    ss_PfKey,sg_PfKey=_makePfKey(ss_info,sg_info,Pf)

    ### Parse each station results and delete detections that occur on multiple subpspace, keeping only the subspace with highest detection stat  
    if reduceDets:
        ssdf=_deleteDetDuplicates(ssDB,trigCon,trigParameter,ss_associateBuffer,starttime,endtime,Stations,'ss_df',PfKey=ss_PfKey)
        sgdf=_deleteDetDuplicates(ssDB,trigCon,trigParameter,sg_associateBuffer,starttime,endtime,Stations,'sg_df',PfKey=sg_PfKey)
    else:
        if Pf:
            raise Exception('When using the Pf parameter reduceDets must be True')
        ssdf=detex.util.loadSQLite(ssDB,'ss_df')
        sgdf=detex.util.loadSQLite(ssDB,'sg_df')
    df=pd.concat([ssdf,sgdf],ignore_index=True)
    df.reset_index(drop=True,inplace=True)
    if isinstance(Stations,(list,tuple)): # reduce stations if requested
        df=df[df.Sta.isin(Stations)]    
    ### Associate detections on different stations together
    Dets,Autos=_associateDetections(df,associateReq,requiredNumStations,ss_associateBuffer,ss_info,temkey)
    
    ### Make a dataframe of verified detections if applicable
    Vers=_verifyEvents(Dets,Autos,veriFile,veriBuffer,includeAllVeriColumns)

    ssres=SSResults(Dets,Autos,Vers,ss_info,filt,temkey,stakey,templateKey,templatePath,condir)
    return ssres


#def _combineSubsAndSings(dfb):
#    """
#    Parse each group (potential events) and for each station delete either the subspace or the single, whichever has lower 
#    detection statistics
#    """
#    sslist=[]
#
#    dfb.sort(columns=['Sta','MSTAMPmin'],inplace=True)
#    dfb['Gnum']=((ssdf.MSTAMPmin-associateBuffer>ssdf.MSTAMPmax.shift())&(ssdf.Sta==ssdf.Sta.shift())).cumsum()
#    ssdf.sort(columns=['Gnum','DS'],inplace=True)
#    ssdf.drop_duplicates(subset='Gnum',take_last=True,inplace=True)
#    ssdf.reset_index(inplace=True,drop=True)
    
def _makePfKey(ss_info,sg_info,Pf):
    """
    Make simple df for defining DS values corresponing to Pf for each subspace station pair
    """
    if not Pf: #if no Pf value passed simply return none
        return None,None
    else:
        ss_df=pd.DataFrame(columns=['Sta','Name','DS','betadist'])
        sg_df=pd.DataFrame(columns=['Sta','Name','DS','betadist'])
        if isinstance(ss_info,pd.DataFrame):
            for num,row in ss_info.iterrows():
                TH=scipy.stats.beta.isf(Pf,row.beta1,row.beta2,0,1)
                if TH>.94: # if isf returns unealistically high pf, initiated forward grid serach
                    TH,Pftemp=_approximateThreshold(row.beta1,row.beta2,Pf,1000,3)
                ss_df.loc[len(ss_df)]=[row.Sta,row.Name,TH,[row.beta1,row.beta2,0,1]]
        if isinstance(sg_info,pd.DataFrame):
            for num,row in sg_info.iterrows():
                TH=scipy.stats.beta.isf(Pf,row.beta1,row.beta2,0,1)
                if TH>.94: # if isf returns unealistically high pf, initiated forward grid serach
                    TH,Pftemp=_approximateThreshold(row.beta1,row.beta2,Pf,1000,3)
                sg_df.loc[len(sg_df)]=[row.Sta,row.Name,TH,[row.beta1,row.beta2,0,1]]            
        sg_df.reset_index(drop=True,inplace=True)
        ss_df.reset_index(drop=True,inplace=True)
        return ss_df,sg_df
        
                
def _approximateThreshold(beta_a,beta_b,target,numintervals,numloops):
    """
    Because scipy.stats.beta.isf can break, if it returns a value near 1 when this is obvious wrong initialize grid search algo to get close
    to desired threshold using forward problem which seems to work where inverse fails
    See this bug report: https://github.com/scipy/scipy/issues/4677
    """
    
    startVal,stopVal=0,1     
    loops=0
    while loops<numloops:
        Xs=np.linspace(startVal,stopVal,numintervals)
        pfs=np.array([scipy.stats.beta.sf(x,beta_a,beta_b) for x in Xs])
        resids=abs(pfs-target)
        minind=resids.argmin()
        bestPf=pfs[minind]
        bestX=Xs[minind]
        startVal,stopVal=Xs[minind-1],Xs[minind+1]
        loops+=1
        if minind==0 or minind==numintervals-1:
            raise Exception ('Grind search failing, set threshold manually')
    return bestX,bestPf 
        
    
    
def _verifyEvents(Dets,Autos,veriFile,veriBuffer,includeAllVeriColumns):
    if not veriFile or not os.path.exists(veriFile): 
        detex.log(__name__, 'No veriFile passed or it does not exist, skipping verification', pri=True)
        return
    else:        
        vertem=_readVeriFile(veriFile)
        vertem['STMP']=[obspy.core.UTCDateTime(x) for x in vertem['TIME']]
        verlist=[]
        additionalColumns=list(set(vertem.columns)-set(['TIME','LAT','LON','MAG','ProEnMag','DEPTH','NAME']))
        vertem
        for vernum,verrow in vertem.iterrows():
            temDets=Dets[(Dets.MSTAMPmin-veriBuffer/2.<verrow.STMP)&(Dets.MSTAMPmax+veriBuffer/2.0>verrow.STMP)&([not x for x in Dets.Verified])]
            if len(temDets)>0: #todo handle this when multiple verifications occur
                trudet=temDets[temDets.DSav==temDets.DSav.max()]
                Dets.loc[trudet.index[0],'Verified']=True

                if includeAllVeriColumns:
                    for col in additionalColumns:
                        if not col in trudet.columns:
                            trudet[col]=verrow[col]
                trudet['VerMag'],trudet['VerLat'],trudet['VerLon'],trudet['VerDepth'],trudet['VerName']=verrow.MAG,verrow.LAT,verrow.LON,verrow.DEPTH,verrow.NAME
                verlist.append(trudet)
            else:
                temAutos=Autos[(Autos.MSTAMPmin-veriBuffer/2.<verrow.STMP)&(Autos.MSTAMPmax+veriBuffer/2.0>verrow.STMP)&([not x for x in Autos.Verified])]
                if len(temAutos)>0: #todo handle this when multiple verifications occur
                    trudet=temAutos[temAutos.DSav==temAutos.DSav.max()]
                    Autos.loc[trudet.index[0],'Verified']=True

                    if includeAllVeriColumns:
                        for col in additionalColumns:
                            if not col in trudet.columns:
                                trudet[col]=verrow[col]
                    trudet['VerMag'],trudet['VerLat'],trudet['VerLon'],trudet['VerDepth'],trudet['VerName']=verrow.MAG,verrow.LAT,verrow.LON,verrow.DEPTH,verrow.NAME
                    verlist.append(trudet)
        if len(verlist)>0:
            verifs=pd.concat(verlist,ignore_index=True)
            verifs.sort(columns=['Event','DSav']) #sort and drop duplicates so each verify event is verified only once
            verifs.drop_duplicates(subset='Event')
            verifs.drop('Verified', axis=1, inplace=True)
        else:
            verifs=pd.DataFrame()
        return verifs
                
        

def _readVeriFile(veriFile):
    try:
        df=pd.read_csv(veriFile)
    except:
        try:
            df=pd.read_pickle(veriFile)
        except:
            try:
                df=detex.util.loadSQLite(veriFile,'verify')
            except:
                raise Exception('%s could not be read, it must either be csv, pickled dataframe or sqlite database with table name "verify"'%veriFile)
    if not set(['TIME','LAT','LON','MAG','DEPTH','NAME']).issubset(df.columns):
        raise Exception('%s does not have the required columns, it needs TIME,LAT,LON,MAG,DEPTH,NAME' % veriFile )
    return df
        

def _buildSQL(PfKey,trigCon,trigParameter,Stations,starttime,endtime,tableName):
    """Function to build a list of SQL commands for loading the database with desired parameters
    """
    SQL=[] #init. blank list
    if not starttime or not endtime:
        starttime=0.0
        endtime=4500*3600*24*365.25
    else:
        starttime=obspy.UTCDateTime(starttime).timestamp
        endtime=obspy.UTCDateTime(endtime).timestamp

    #define stations
    if isinstance(Stations,(list,tuple)):    
        if isinstance(PfKey,pd.DataFrame):
            PfKey=PfKey[PfKey.Sta.isin(Stations)]
    else:
        if isinstance(PfKey,pd.DataFrame):
            Stations=PfKey.Sta.values
        else:
            Stations=['*'] #no stations definition use all
            
    if isinstance(PfKey,pd.DataFrame):
        for num,row in PfKey.iterrows():
            table='sg_df' if 'SG' in row.Name else 'ss_df' #make sure appropraite table is used for either subspace or singleton
            SQL.append('SELECT %s FROM %s WHERE Sta="%s" AND Name="%s" AND  DS>=%f AND MSTAMPmin>%f AND MSTAMPmin<%f'% ('*',table,row.Sta,row.Name,row.DS,starttime,endtime))
    
    else:         
        if trigCon==0:
            cond='DS'
        elif trigCon==1:
            cond='DS_STALTA'
        for sta in Stations:
            if sta=='*':
                SQL.append('SELECT %s FROM %s WHERE %s >= %s AND MSTAMPmin>=%f AND MSTAMPmin<=%f' % ('*', tableName,cond,trigParameter,starttime,endtime))
            else:
                SQL.append('SELECT %s FROM %s WHERE %s="%s" AND %s >= %s AND MSTAMPmin>=%f AND MSTAMPmin<=%f' % ('*', tableName,'Sta',sta,cond,trigParameter,starttime,endtime))
    return SQL
    
def _deleteDetDuplicates(ssDB,trigCon,trigParameter,associateBuffer,starttime,endtime,Stations,tableName,PfKey=None):
    """
    delete dections of same event, keep only detection with highest detection statistic
    """
    sslist=[]
    SQLstr=_buildSQL(PfKey,trigCon,trigParameter,starttime,Stations,endtime,tableName)   
    for sql in SQLstr:
        loadedRes=detex.util.loadSQLite(ssDB,tableName,sql=sql)
        if isinstance(loadedRes,pd.DataFrame):
            sslist.append(loadedRes)
    if len(sslist)<1: #if no events found
        return None
    try:
        ssdf=pd.concat(sslist,ignore_index=True)
    except ValueError:
        raise Exception ('Cannot create detResults instance, no detections meet all requirements')
    ssdf.reset_index(drop=True,inplace=True)

    ssdf.sort(columns=['Sta','MSTAMPmin'],inplace=True)
    ssdf['Gnum']=((ssdf.MSTAMPmin-associateBuffer>ssdf.MSTAMPmax.shift())&(ssdf.Sta==ssdf.Sta.shift())).cumsum()
    ssdf.sort(columns=['Gnum','DS'],inplace=True)
    ssdf.drop_duplicates(subset='Gnum',take_last=True,inplace=True)
    ssdf.reset_index(inplace=True,drop=True)
        
    return ssdf

  
def _associateDetections(ssdf,associateReq,requiredNumStations,associateBuffer,ss_info,temkey): 
    """
    Associate detections together using pandas groupby return dataframe of detections and autocorrelations
    """
    ssdf.sort(columns='MSTAMPmin',inplace=True) #sort bassed on min predicted time
    ssdf.reset_index(drop=True,inplace=True) #reset indecies
    if isinstance(ss_info,pd.DataFrame) and associateReq>0:
        ssdf=pd.merge(ssdf,ss_info,how='inner',on=['Sta','Name'])
    groups=ssdf.groupby((ssdf.MSTAMPmin-associateBuffer>ssdf.MSTAMPmax.shift()).cumsum()) 
    autolist=[pd.DataFrame(columns=['Event','DSav','DSmax','DS_STALTA','MSTAMPmin','MSTAMPmax','Mag','ProEnMag','Verified','Dets'])]
    detlist=[pd.DataFrame(columns=['Event','DSav','DSmax','DS_STALTA','MSTAMPmin','MSTAMPmax','Mag','ProEnMag','Verified','Dets'])]
    temkey['STMP']=np.array([obspy.core.UTCDateTime(x) for x in temkey.TIME])
    temcop=temkey.copy()

    if isinstance(ss_info,pd.DataFrame) and associateReq>0: #if there is an association requirement (IE required number of shaed events)
        for num,g in groups:
            
            g=_checkSharedEvents(g)
            if len(set(g.Sta))>=requiredNumStations: #Make sure detections occur on the required number of stations
                isauto,autoDF=_createAutoTable(g,temcop)
                if isauto:
                    autolist.append(autoDF)
                else:
                    detdf=_createDetTable(g)
                    detlist.append(detdf)        
    else:
        for num,g in groups:
            if len(set(g.Sta))>=requiredNumStations: #Make sure detections occur on the required number of stations
                if len(set(g.Sta))<len(g.Sta): # If there is more than one single or subpspace representing a station on each event only keep the one with highest DS
                    g=g.sort(columns='DS').drop_duplicates(subset='Sta',take_last=True).sort(columns='MSTAMPmin')
                isauto,autoDF=_createAutoTable(g,temcop)
                if isauto:
                    autolist.append(autoDF)
                    #temcop=temcop[temcop.NAME!=autoDF.iloc[0].Event]
                else:
                    detdf=_createDetTable(g)
                    detlist.append(detdf)
    detTable=pd.concat(detlist,ignore_index=True)
    autoTable=pd.concat(autolist,ignore_index=True)
    return [detTable,autoTable]
               

def _checkSharedEvents(g): # Look at the union of the events and delete those that do not meet the requirements
    pass #TODO figure out how to incorporate an association requirement
    
def _createDetTable(g):
    mag,proEnMag=_getMagnitudes(g)
    event=obspy.core.UTCDateTime(np.mean([g.MSTAMPmin.mean(),g.MSTAMPmax.mean()])).formatIRISWebService().replace(':','-').split('.')[0]
    detDF=pd.DataFrame([[event,g.DS.mean(),g.DS.max(),g.DS_STALTA.mean(),g.MSTAMPmin.min(),g.MSTAMPmax.max(),mag,proEnMag,False,g]],columns=['Event','DSav','DSmax','DS_STALTA','MSTAMPmin','MSTAMPmax','Mag','ProEnMag','Verified','Dets'])
    return detDF
         
def _createAutoTable(g,temkey):
    isauto=False
    for num,row in g.iterrows(): #find out if this is an auto detection
        temtemkey=temkey[(temkey.STMP>row.MSTAMPmin)&(temkey.STMP<row.MSTAMPmax)]
        if len(temtemkey)>0:
            isauto=True
            event=temtemkey.iloc[0].NAME
    if isauto:
        mag,proEnMag=_getMagnitudes(g)
        autoDF=pd.DataFrame([[event,g.DS.mean(),g.DS.max(),g.DS_STALTA.mean(),g.MSTAMPmin.min(),g.MSTAMPmax.max(),mag,proEnMag,False,g]],columns=['Event','DSav','DSmax','DS_STALTA','MSTAMPmin','MSTAMPmax','Mag','ProEnMag','Verified','Dets'])
        return isauto,autoDF
    else:
        return isauto,pd.DataFrame()

def _getMagnitudes(g):
    if any ([not np.isnan(x) for x in g.Mag]):
        mag=np.nanmedian(g.Mag)
    else:
        mag=np.NaN
    try:    
        any ([not np.isnan(x) for x in g.ProEnMag])
    except:
        deb(g)
    if any ([not np.isnan(x) for x in g.ProEnMag]):
        PEmag=np.nanmedian(g.ProEnMag)
    else:
        PEmag=np.NaN
    return mag,PEmag
         
def _loadSSdb(ssDB,trigCon,trigParameter,sta=None): #load the subspace database
    if trigCon==0:
        cond='DS'
    elif trigCon==1:
        cond='DS_STALTA'
    if sta:
        sql='SELECT %s FROM %s WHERE %s="%s" AND %s > %s' % ('*', 'ss_df','Sta',sta,cond,trigParameter)
    else:
        sql='SELECT %s FROM %s WHERE %s > %s' % ('*', 'ss_df',cond,trigParameter)
    #sys.exit(1)
    df=detex.util.loadSQLite(ssDB,'ss_df',sql=sql)
    return df

def _checkInputs(trigCon,trigParameter,associateReq,associateBuffer,requiredNumStations):
    if not isinstance(trigCon,int) or not (trigCon==0 or trigCon==1):
        raise Exception ('trigcon must be an int, either 0 or 1')
    if trigCon==0:
        if not isinstance(trigParameter,numbers.Real) or trigParameter>1 or trigParameter<0:
            raise Exception ('When trigCon==0 trigParameter is the required detection statistic and therefore must be between 0 and 1')
    elif trigCon==1:
        if not isinstance(trigParameter,numbers.Real) or (trigParameter<1 and trigParameter!=0): #allow 0 to simply select all
            raise Exception('When trigCon==1 trigParameter is the STA/LTA of the detection statistic vector and therefore must be greater than 1')
    if not isinstance(associateReq,int) or associateReq<0:
        raise Exception ('AssociateReq is the required number of events a subspace must share for detections from different stations to be associated together\
        and therefore must be an integer 0 or greater')
    if not isinstance(associateBuffer,numbers.Real) or associateBuffer<0:
        raise Exception('associateBuffer must be a real number greater than 0' )
    if not isinstance(requiredNumStations,int) or requiredNumStations<1:
        raise Exception('requiredNumStations must be an integer greater than 0')
        
def _checkExistence(existList):
    for fil in existList:
        if not os.path.exists(fil): raise Exception('%s does not exists'%fil)
            
def _loadInfoDataFrames(ssDB):
    ss_info=detex.util.loadSQLite(ssDB,'ss_info') # load subspace info
    if isinstance(ss_info,pd.DataFrame):
        ss_info['NumEvents']=[len(row.Events.split(',')) for num,row in ss_info.iterrows()]
    sg_info=detex.util.loadSQLite(ssDB,'sg_info')
    if isinstance(sg_info,pd.DataFrame):
        sg_info['NumEvents']=1
    return ss_info,sg_info
                
class SSResults(object):
    def __init__(self,Dets,Autos,Vers,ss_info,ss_filt,temkey,stakey,templateKey,templatePath,condir):
        self.Autos=Autos
        self.Dets=Dets
        self.NumVerified=len(Vers) if isinstance(Vers,pd.DataFrame) else 'N/A'
        self.Vers=Vers
        self.info=ss_info
        self.filt=ss_filt
        self.StationKey=stakey
        self.TemplateKey=temkey
        self.TemKeyPath=templateKey
        self.eventDir=templatePath
        self.condir=condir
        
    def writeDetections(self,onlyVerified=False,minDS=False,minMag=False,eventDir=None,updateTemKey=True,temkeyPath=None,
                                timeBeforeOrigin=3*60,timeAfterOrigin=9*60):
        """
        Function to make all of the eligable new detections templates. New event directories will be added to eventDir
        and the template key will be updated with detected events having a lower case "d" before the name
        
        Parameters
        ----------
        onlyVerified : boolean
            If true only use detections that are verified
        minDS : False or float between 0.0 and 1.0
            If float only use detections with average detection statistics above minDSave
        minMag : false or float
            If float only use detections with estimated magnitudes above minMag
        eventDir : None or str
            If None new waveforms of detections are stored in default event directory (usually EventWaveForms). 
            If str then the str must be path to new directory in which detected event waveforms will be stored. 
            If it does not exist it will be created
        updateTemKey : boolean
            If true update the template key with the new detections
        temkeyPath : None or str
            if None use the default path to the template key, else new path to templatekey, if it does not 
            exist it will be created
        timeBeforeOrigin : real number (float or int)
            Seconds before predicted origin to get (default is the same as the getData.getAllData defaults)
        timeAfterOrigin : real number (float or int)
            Seconds after predicted origin to get 
        """
        dets=self.Dets.copy()
        if onlyVerified:
            dets=dets[dets.Verified]
        if minDS:
            dets=dets[dets.minDS]
        if minMag:
            dets=dets[dets.Mag>=minMag]
        if not eventDir:
            eventDir=self.eventDir
        if not temkeyPath:
            temkeyPath=self.TemKeyPath
        temkey=self.TemplateKey
        
        detTem=pd.DataFrame(index=range(len(dets)),columns=temkey.columns)
            
        for num,row in dets.iterrows(): #loop through detections and save each one
            origin=obspy.UTCDateTime(np.mean([row.MSTAMPmax,row.MSTAMPmin]))
            Evename=origin.formatIRISWebService().replace(':','-')
            eveDirName='d'+Evename
            
            if not os.path.exists(os.path.join(eventDir,eveDirName)): #if the directory doesnt exists create it
                os.makedirs(os.path.join(eventDir,eveDirName))
                
            for stanum,sta in self.StationKey.iterrows(): #loop through eac station and load stream then save
                station=sta.NETWORK+'.'+sta.STATION
                try:
                    ST=detex.util.loadContinuousData(origin-timeBeforeOrigin,origin+timeAfterOrigin,station,Condir=self.condir)
                    ST.write(os.path.join(eventDir,eveDirName,station+'.'+Evename+'.'+'pkl'),'pickle')
                except:   
                    detex.log(__name__,'Could not write and save %s for station %s' %(Evename,station),level='warning')
            
            detTem.loc[num,'NAME'],detTem.loc[num,'TIME'],detTem.loc[num,'MAG']=eveDirName,origin.timestamp,row.Mag
        
        temkeyNew=pd.concat([temkey,detTem],ignore_index=True)
        temkeyNew.reset_index(inplace=True,drop=True)
        if updateTemKey:
            temkeyNew.to_csv(temkeyPath)
        
        
    def __repr__(self):
        outstr='SSResults instance with %d autodections and %d new detections, %s are verified'%(len(self.Autos),len(self.Dets),str(self.NumVerified))
        return outstr
        

            
            
            
            
            
            
            
            
            
            
            
            
