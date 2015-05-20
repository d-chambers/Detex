# -*- coding: utf-8 -*-
"""
Created on Tue Nov 04 21:35:44 2014

@author: Derrick

Read directory of unziped ANF files into pandas data frame
"""
import pandas as pd, obspy, numpy as np, glob, os
def readANF(anfdir,lon1=-180,lon2=180,lat1=0,lat2=90,getPhases=False,UTC1='1960-01-01',
            UTC2='3000-01-01',Pcodes=['P','Pg'],Scodes=['S','Sg']):
    """Function to read the ANF directories as downloaded from the ANF Earthscope Website"""
    monthDirs=glob.glob(os.path.join(anfdir,'*'))
    Eve=pd.DataFrame()
    for month in monthDirs:
        utc1=obspy.core.UTCDateTime(UTC1).timestamp
        utc2=obspy.core.UTCDateTime(UTC2).timestamp
        #read files for each month
        
        
        dfOrigin=_readOrigin(glob.glob(os.path.join(month,'*.origin'))[0])
        dfOrigerr=readOrigerr(glob.glob(os.path.join(month,'*.origerr'))[0])
                #merge event files togther
        DF=pd.merge(dfOrigin,dfOrigerr)
            
        #discard all events outside area of interest
        DF=DF[(DF.Lat>lat1)&(DF.Lat<lat2)&(DF.Lon>lon1)&(DF.Lon<lon2)&(DF.time>utc1)&(DF.time<utc2)]
        
        if getPhases:
            dfAssoc=_readAssoc(glob.glob(os.path.join(month,'*.assoc'))[0])
            dfArrival=_readArrival(glob.glob(os.path.join(month,'*.arrival'))[0])
 
            #link associated phases with files
            DF=_linkPhases(DF,dfAssoc,dfArrival,Pcodes,Scodes)
        
        Eve=pd.concat([DF,Eve],ignore_index=True)
        Eve.reset_index(drop=True,inplace=True)
    return Eve

def readOrigerr(origerrFile):
    columnNames=['orid','sobs','smajax','sminax','strike','sdepth','conf']
    columnSpecs=[(0,8),(169,179),(179,188),(189,198),(199,205),(206,215),(225,230)]
    df=pd.read_fwf(origerrFile,colspecs=columnSpecs,header=None,names=columnNames)
    return df

def _readOrigin(originFile):
    columnNames=['Lat','Lon','depth','time','orid','evid',
     'jdate','nass','ndef','ndp','grn','srn','etype','review','depdp','dtype',
     'mb','mbid','ms','msid','ml','mlid','algo','auth','commid','lddate']
    columnSpecs=[(0,9),(10,20),(20,29),(30,47),(48,56),(57,65),(66,74),(75,79),(80,84),
                (85,89),(90,98),(99,107),(108,110),(111,115),(116,125),(126,128),(128,136),
                (136,144),(145,152),(153,161),(162,169),(170,178),(179,194),(195,210),
                (211,219),(220,237)]
    
    df=pd.read_fwf(originFile,colspecs=columnSpecs,header=False,names=columnNames)
    df['DateString']=[obspy.core.UTCDateTime(x).formatIRISWebService() for x in df.time]
    return df

def _readAssoc(assocFile):
    columnNames=['arid','orid','sta','phase','belief','delta']
    columnSpecs=[(0,8),(9,17),(18,24),(25,33),(34,38),(39,47)]
    df=pd.read_fwf(assocFile,colspecs=columnSpecs,header=False,names=columnNames)
    return df

def _readArrival(arrivalFile):
    columnNames=['sta','time','arid','stassid','iphase','amp','per','snr']
    columnSpecs=[(0,6),(7,24),(25,33),(43,51),(70,78),(136,146),(147,154),(168,178)]
    df=pd.read_fwf(arrivalFile,colspecs=columnSpecs,header=False,names=columnNames)
    return df
    
def _linkPhases(DF,dfAssoc,dfArrival,Pcodes,Scodes):
    DF['Picks']=[{} for x in range(len(DF))]
    for a in DF.iterrows():
        dfas=dfAssoc[dfAssoc.orid==a[1].orid] #DF associated with orid, should be one row
        dfas=dfas[dfas.phase.isin(Pcodes+Scodes)]
        dfas['time']=float()
        dfas['snr']=float
        for b in dfas.iterrows(): #get times from df arrival
            dfar=dfArrival[dfArrival.arid==b[1].arid]
            dfas.time[b[0]]=dfar.time.iloc[0]
            dfas.snr[b[0]]=dfar.snr.iloc[0]
        for sta in list(set(dfas.sta.values)):
            dfasSta=dfas[dfas.sta==sta]
            dfasP=dfasSta[dfasSta.phase.isin(Pcodes)]
            dfasS=dfasSta[dfasSta.phase.isin(Scodes)]
            tempdict={sta:[0,0]}
            if len(dfasP)>0:
                tempdict[sta][0]=dfasP.time.iloc[0]
            if len(dfasS)>0:
                tempdict[sta][1]=dfasS.time.iloc[0]
            DF.Picks[a[0]]=dict(DF.Picks[a[0]].items()+tempdict.items())
    return DF
    
def ANFtoTemplateKey(anfDF,temKeyName='TemplateKey_anf.csv',saveTempKey=True):   
    """Convert the dataframe created by the readANF function to a detex templatekey csv"""
    ds=[x.split('.')[0].replace(':','-') for x in anfDF.DateString]
    ts=[x.replace(':','-') for x in anfDF.DateString]
    contrib=['ANF']*len(anfDF)
    mtype=['ML']*len(anfDF)
    stakey=['StationKey.csv']*len(anfDF)
    df=pd.DataFrame()
    df['CONTRIBUTOR'],df['NAME'],df['TIME'],df['LAT'],df['LON'],df['DEPTH'],df['MTYPE'],df['MAG'],df['STATIONKEY']=contrib,ds,ts,anfDF.Lat.tolist(),anfDF.Lon.tolist(),anfDF.depth.tolist(),mtype,anfDF.ml.tolist(),stakey
    if saveTempKey:
        df.to_csv(temKeyName)
    return df

def makePickTimes(ANF,stakey):
    """
    Take the dataframe created by readANF and make a pick file with P and S times for each station also found in station key
    """
    stations=stakey.STATION.values
    DFlist=[]
    for num,row in ANF.iterrows():
        evename=row.DateString.split('.')[0].replace(':','-') 
        stainter=list(set(stations)&set(row.Picks.keys()))
        for sta in stainter:
            Path=os.path.join('EventWaveForms','evename','TA.'+sta+'.'+evename+'.pkl')
            Ptime=row.Picks[sta][0]
            Stime=row.Picks[sta][1]
            starttime=Ptime
            endtime=Stime if Stime>Ptime+30 else Ptime+30
            df=pd.DataFrame([[evename,sta,Ptime,Stime,Path,starttime,endtime]],columns=['Name','Station','P','S','Path','Startttime','Endtime'])
            DFlist.append(df)
    DF=pd.concat(DFlist,ignore_index=True)
    return DF
        
        