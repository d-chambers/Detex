# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:13:28 2014

@author: Derrick
Function to pick a directory full of events and return pick times for each event (saved as an mseed file)
"""

"""
AutoPickFunctions, designed to be a course pick, find out if events occur within station boudnaries using simple
inversion and saves events likely to occur within stationed area
_____________________________________________________________________________________________________________________________
"""


import os, obspy, glob,sys, pandas as pd, numpy as np,shutil, time
from obspy.signal.trigger import arPick, pkBaer, recSTALTA
from sympy import Symbol, sin, cos, asin, sqrt, diff,pi
from obspy.signal import coincidenceTrigger
from obspy.signal.trigger import pkBaer, arPick
"""
Functions for Picker
"""
def AutoPick(pickdir='Trigs',outputFormat='HYP',outdir='PickTimes',picker='pkBaer',goodPick='goodpicking',
             startnum='StartHere.txt',filelist=None):
    StaKey=pd.read_csv('StationKey.csv')
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if not os.path.isdir(goodPick):
        os.makedirs(goodPick)
    latmin,latmax,lonmin,lonmax=_getLimits(StaKey)
    dA,A=_getdA(6371) #calculates symbolically the partial derivatives of the operator for location inversion
    count=0
    if isinstance(filelist,list):
        files=filelist
    else:
        files=glob.glob(os.path.join(pickdir,'*'))
    try: #Look for startnum txt to see where to begin
        start=int(np.loadtxt(startnum).tolist()+int(np.random.rand()*10))
    except:
        print 'No progress file, begining at 0'
    for fil in range(start,len(files)):
        ST=obspy.core.read(files[fil])
        filename=os.path.basename(files[fil]).replace('msd','pha')
        with open(os.path.join(outdir,filename),'wb') as pik:
            Stations=StaKey.STATION
            STR=[0]*len(Stations)
            pks=[1]*len(Stations)
            cou=0
            Key=StaKey.copy()
            Key['Ppick']=0.000000000
            Key['Spick']=0.000000000
            Key['Trace']=object
            for sta in Key.iterrows():
                st=ST.copy()
                st=st.select(station=sta[1].STATION)
                if len(st)==0: #If station not found skip iteration
                    continue
                sr=st[0].stats.sampling_rate
                STR[cou]=st.copy()
                cou+=1
            for tr in range(len(STR)):
                if picker=='arPick':
                    Z,N,E=STR[tr].select(channel='*Z').copy(), STR[tr].select(channel='*N').copy(),STR[tr].select(channel='*E').copy()
                    p_pick, s_pick = arPick(Z[0].data, N[0].data,E[0].data, Z[0].stats.sampling_rate,1.0, 15.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2)
                elif picker=='pkBaer':
                    zst=STR[tr].select(channel='*Z')
                    p_pick, phase_info = pkBaer(zst[0].data, sr,20, 60, 1.2, 10.0, 100, 100)
                    if p_pick==1 or p_pick<0: #if baer/arPicker pick failes resort to basic recursive STA/LTA picks
                        cft=recSTALTA(STR[tr].select(channel='*Z')[0].data, int(1 * sr), int(10 * sr))
                        p_pick=cft.argmax()
                pks[tr]=p_pick
                Key['Ppick'][int(tr)]=STR[tr][0].stats.starttime.timestamp+p_pick/float(sr)
                Key['Trace'][int(tr)]=STR[tr][0]
            #print (p_pick)
            for sta in Key.drop_duplicates(cols='Ppick').iterrows():
                if outputFormat=='NNL':
                    if picker=='arPick':
                        line1,line2=_makeArPickNLLLine(sta[1].Ppick,s_pick,STR[tr])
                        pik.write(line1)
                        pik.write(line2)
                    if picker=='pkBaer':
                        line1=_makePkBaerNLLLine(sta[1].Ppick,sta[1].Trace.stats.sampling_rate,sta[1].Trace)
                        pik.write(line1)
                if outputFormat=='HYP':
                    if picker=='pkBaer':
                        try:
                            line=_makePkBaerHYPLine(sta[1].Ppick,sta[1].Trace.stats.sampling_rate,sta[1].Trace)
                        except:
                            break
                        pik.write(line)   
            m=invertForLocation(Key,dA,A)
            if outputFormat=='NNL':
                pass #write this in later
            if outputFormat=='HYP' and m[0]>-90 and m[0]<90 and m[1]>-180 and m[1]<180:
                endline=writeEndLine(m,Key)
                #print np.min(np.min(Key.Ppick.values))
                pik.write(endline)
                pik.write('\n')
            #Key=_genSynthetics(Key,A)
            

        if m[0] < latmin or m[0] > latmax or m[1] < lonmin or m[1] > lonmax: #If rough location is in station peremeter make picks and copy file
            os.remove(os.path.join(outdir,filename))
        else:
            shutil.copy(files[fil],os.path.join(goodPick,os.path.basename(files[fil])))
            print filename
        count+=1
        if count%10==0:
            _writeProgressFile(count)
            
def hypIntoDF(DF=None,pikle='Tro.pkl',sumfile='sum2000'):
    """Puts in info from the sum file into the dataframe
    """
    if DF==None:
        DF=pd.read_pickle(pikle)
    #Finish this later



"""
Functions for the Inversion
"""
def L2Norm(Mat):
    L2=np.sum(np.sqrt(np.multiply(Mat,Mat)))
    return L2
    
def _genSynthetics(Key,A):
    TV={lat:float(35.625203),lon:float(-100.801808),dep:float(0.0),t:float(9000),v:float(4.5)}
    for a in Key.iterrows():
        stationSub={lati:a[1].LAT,loni:a[1].LON,depi:a[1].ELEVATION/1000.00}
        Key['Ppick'][a[0]]=A.subs(dict(TV.items()+stationSub.items()))
    return Key
    
    
def invertForLocation(Key,dA,A):
    #simple iversion assuming constant velocity medium
    m=_getStartVect(Key) #starting position with some assumptions
    resids=_calcResids(A,m,Key)
    count=0
    while L2Norm([float(x) for x in resids.values()])>.1 and count<10:
        Al=_fillInA(m,Key,dA)
        dm1=_calcdM(Al,Key,resids)
        m=np.add(dm1,m)
        resids=_calcResids(A,m,Key) #residuals 
        count+=1
    #print m,  L2Norm([float(x) for x in resids.values()])
    return m
    
def _calcResids(A,m,Key):
    #sub={lat:float(m[0]),lon:float(m[1]),dep:float(m[2]),t:float(m[3]),v:float(m[4])}
    sub={lat:float(m[0]),lon:float(m[1]),dep:float(0),t:float(m[2]),v:float(4.5)} # fix depth and V
    resids={}
    for a in Key.iterrows():
        stationSub={lati:a[1].LAT,loni:a[1].LON,depi:a[1].ELEVATION/1000.00}
        #print a[1].Ppick,A.subs(dict(sub.items()+stationSub.items()))
        resids[a[1].STATION]=(a[1].Ppick-A.subs(dict(sub.items()+stationSub.items()))).evalf()
    return resids
    
    
def _calcdM(A,Key,resids):
    dD=[0]*len(Key)
    for a in Key.iterrows():
        dD[a[0]]=resids[a[1].STATION]
    m=np.dot(np.linalg.inv(np.dot(A.transpose(),A)),np.dot(A.transpose(),dD))
    return m
    

def _fillInA(sVect,Key,dA):
    #sub={lat:float(sVect[0]),lon:float(sVect[1]),dep:float(sVect[2]),t:float(sVect[3]),v:float(sVect[4])}
    sub={lat:float(sVect[0]),lon:float(sVect[1]),dep:float(0),t:float(sVect[2]),v:float(4.5)}
    Al=np.zeros((len(Key),len(sVect)))
    for a in Key.iterrows():
        stationSub={lati:a[1].LAT,loni:a[1].LON,depi:a[1].ELEVATION/1000.00}
        for b in range(len(sVect)):
            Al[a[0],b]=dA[b].subs(dict(sub.items()+stationSub.items()))
            if np.isnan(Al[a[0],b]):
                raise Exception('Nan in A')
    return Al
    
def _getStartVect(Key): # Start each inversion 1 km below closest station 10s before hit, 4km/s velocity
    sKey=Key[Key.Ppick==min(Key.Ppick)]
    #sVect=[sKey.LAT.values[0]+.0001*sKey.LAT.values[0],sKey.LON.values[0]+.0001*sKey.LON.values[0],sKey.ELEVATION.values[0]/1000.00-1.0,sKey.Ppick.values[0]-10,4.0]
    sVect=[sKey.LAT.values[0]+.0001*sKey.LAT.values[0],sKey.LON.values[0]+.0001*sKey.LON.values[0],sKey.Ppick.values[0]-10]
    
    return(sVect)
    
def _getdA(R):# get A for lat/long model
    #A, not quite right but a devent approximation
    global lat,lati,lon,loni,v,dep,depi,t
    lat,lati=Symbol('lat'),Symbol('lati')
    lon,loni=Symbol('lon'),Symbol('loni')
    v=Symbol('V')
    dep,depi=Symbol('dep'),Symbol('depi')
    t=Symbol('t')
    h=2*R*asin(sqrt((sin(lat*pi/360-lati*pi/360))**2+cos(lati*pi/180)*cos(lat*pi/180)*(sin(lon*pi/360-loni*pi/360))**2))
    A=(1/v)*sqrt((h)**2+(dep-depi)**2)+t
    #A=(1/v)*sqrt((2*R*(asin(sqrt(sin((lat-lati)/2))+cos(lati)*cos(lat)*sin((lon-loni)/2)**2)))**2+(dep-depi)**2)+t
    dlat=diff(A,lat)
    dlon=diff(A,lon)
    ddep=diff(A,dep)
    dotime=diff(A,t)
    dvel=diff(A,v)
    #dA=[dlat,dlon,ddep,dotime,dvel]
    dA=[dlat,dlon,dotime]
    return dA,A

def _getLimits(Key):
    lats=Key.LAT
    lons=Key.LON
    latmin,latmax=min(lats),max(lats)
    lonmin,lonmax=min(lons),max(lons)
    return latmin,latmax,lonmin,lonmax
           
            

"""
functions to write file formats
"""      


def _writeProgressFile(count,name='StartHere.txt'):
    with open(name,'w') as text:
        text.write(str(count))         
 
def _makePkBaerHYPLine(p_pick,sr,tr):
    pt=float(p_pick)/sr
    Ptime=tr.stats.starttime+pt
    sta=tr.stats.station
    net=tr.stats.network
    cha='ZENZ'
    datestring=Ptime.formatIRISWebService().replace('-','').replace('T','').replace(':','').replace('.','')
    YYYYMMDDHHMM=datestring[0:12]
    ssss=datestring[12:16]
    end='01'
    ty='iP 0'
    line="{:<5}{:<3}{:<5}{:<3}{:<13}{:<80}{:<2}\n".format(sta,net,cha,ty,YYYYMMDDHHMM,ssss,end)
    return line
    
def writeEndLine(m,Key):
    mintime=obspy.core.UTCDateTime(np.min(Key.Ppick.values))
    space=' '
    hhmmssss=mintime.formatIRISWebService().replace('-','').replace('T','').replace(':','').replace('.','')[8:16]
    print  hhmmssss
    print obspy.core.UTCDateTime(mintime)
    lat,latminute=str(abs(int(m[0]))),str(abs(60*(m[0]-int(m[0])))).replace('.','')[0:4]
    lon,lonminute=str(abs(int(m[1]))),str(abs(60*(m[1]-int(m[1])))).replace('.','')[0:4]
    trialdepth='  400'
    endline="{:<6}{:<8}{:<3}{:<4}{:<4}{:<4}{:<5}\n".format(space,hhmmssss,lat,latminute,lon,lonminute,trialdepth)
    return endline
 
    
def _makePkBaerNLLLine(p,sr,tr):
    pt=float(p)/sr
    Ptime=tr[0].stats.starttime+pt
    PYMD=Ptime.formatIRISWebService().replace('-','').split('T')[0]
    PHM=Ptime.formatIRISWebService().split('T')[1].replace(':','')[0:4]
    PS=Ptime.second+Ptime.microsecond/1000000.0
    line1='%.6s ?   ?   ?P     ?%6s %4s %7.4f GAU 2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00\n' %(tr[0].stats.station,PYMD,PHM,PS)
    return line1
    
def _makeArPickNLLLine(P,S,tr):
    Ptime=tr[0].stats.starttime+P
    Stime=tr[0].stats.starttime+S
    PYMD=Ptime.formatIRISWebService().replace('-','').split('T')[0]
    SYMD=Stime.formatIRISWebService().replace('-','').split('T')[0]
    PHM=Ptime.formatIRISWebService().split('T')[1].replace(':','')[0:4]
    SHM=Stime.formatIRISWebService().split('T')[1].replace(':','')[0:4]
    PS=Ptime.second+Ptime.microsecond/1000000.0
    SS=Stime.second+Ptime.microsecond/1000000.0
    line1='%.6s ?   ?   ?P     ?%6s %4s %7.4f GAU 2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00\n' %(tr[0].stats.station,PYMD,PHM,PS)
    line2='%.6s ?   ?   ?S     ?%6s %4s %7.4f GAU 2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00\n' %(tr[0].stats.station,SYMD,SHM,SS)
    return line1,line2
    
    
"""
Function for Coincidence Picker
scans contious data and picks out events
"""
def findEvents(condir='ContinousWaveForms',stakey='StationKey.csv',chan='Z',
               trigDir='Trigs',startbuff=25,endbuff=200,trigBuff=2000):
    stations=pd.read_csv(stakey)
    if not os.path.isdir(trigDir):
        os.makedirs(trigDir)
    years,juldays=getConRange(stations,condir)
    for a in range(len(years)):
        for b in juldays[a]:
            STfull=makeStream(stations,years[a],b,condir)
            STfull.sort()
            st=STfull.copy()
            st=st.select(channel='*Z')
            trig = coincidenceTrigger("recstalta", 5, 1.0, st, 7, sta=0.5, lta=15,details=True,trigger_off_extension=20)
            trig=trimTrig(trig,trigBuff)
            tt=[0]*len(trig)
            for c in range(len(tt)):
                if trig[c] != None:
                    try:
                        tt[c]=STfull.slice(starttime=trig[c]['time']-startbuff,endtime=trig[c]['time']+trig[c]['duration']+endbuff)
                    except:
                        global de,t,C
                        de,t,C=trig,tt,c
                        sys.exit(1)
                    saveTrace(tt[c],trigDir,trig[c]['time'])
                
def trimTrig(trig,trigBuff):
    times=[obspy.core.UTCDateTime(x['time']).timestamp for x in trig]
    df=pd.DataFrame()
    df['Time']=times
    df['Dif']=df.Time.diff()
    tokill=df[df.Dif<trigBuff]
    for a in tokill.index.values:
        trig[a]=None
    return trig

def saveTrace(st,trigDir,UTC):
    st.write(os.path.join(trigDir,UTC.formatIRISWebService().replace(':','-')+'.msd'),'PICKLE')
    
def getConRange(stations,condir): # get a list of all years and corresponding jdays in data
    years=[]
    for a in stations.iterrows():
        y=glob.glob(os.path.join(condir,a[1].NETWORK+'.'+a[1].STATION,'*'))
        years=years+[os.path.basename(x) for x in y]
    years=list(set(years)) # Get ride of duplicates
    juldays=[None]*len(years)
    for a in range(len(years)):
        for b in stations.iterrows():
            jd=glob.glob(os.path.join(condir,b[1].NETWORK+'.'+b[1].STATION,years[a],'*'))
            if b[0]==0: #if first station
                juldays[a]=[os.path.basename(x) for x in jd]
            else:
                juldays[a]=juldays[a]+[os.path.basename(x) for x in jd]
        juldays[a]=list(set(juldays[a]))
    return years,juldays

def makeStream(stations,year,jday,condir):
    ST=obspy.core.Stream()
    for a in stations.iterrows():
        chans=glob.glob(os.path.join(condir,a[1].NETWORK+'.'+a[1].STATION,year,jday,'*'))
        for chan in chans:
            try:
                ST += obspy.core.read(os.path.join(chan,'*.sac'))
            except TypeError:
                continue

    ST.detrend('linear')
    ST=ST.merge(fill_value=0)
    ST.filter('bandpass',freqmin=1,freqmax=15)
    return ST
    
    
"""
Functions for tunning automatic pickers
"""

def tunePickers(DF,picker='pkBaer',filt=[1,10,2,True]):
    DF['AutoPick']=dict
    DF['Resids']=float
    minresids=999999999999999999999.0
    for p1 in np.linspace(2,160,num=50):
        for p2 in np.linspace(p1+1,800,num=50):
            for a in DF.iterrows():
                if picker=='pkBaer':
                    TR=obspy.core.read(a[1].FileName)
                    TR.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
                    TR=TR.select(channel="*Z")
                    outdict={}
                    for b in TR:
                        cft=recSTALTA(b[0].data, int(p1), int(p2))
                        p_pick=cft.argmax()
                        Pstamp=p_pick/(float(b.stats.sampling_rate))+b.stats.starttime.timestamp
                        idd=b.stats.station
                        outdict[idd]=Pstamp
                    DF.AutoPick[a[0]]=outdict
                    sq=np.square(pd.Series(outdict)-pd.Series(DF.UTCPtime[a[0]]))
                    DF.Resids[a[0]]=np.sqrt(np.average(sq[~np.isnan(sq)]))
        Resids=np.sum(DF.Resids.values)
        if Resids<minresids:
            bestparams=[p1,p2]
            print Resids
    return bestparams
    """
    DF is data frame from detex.manualPicker
    """


