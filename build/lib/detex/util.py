# -*- coding: utf-8 -*-
"""
Created on Thu May 29 16:41:48 2014

@author: Derrick
"""
import numpy as np, obspy, glob, os, streamPick, sys, sqlite3, detex.pandas_dbms, random
import pandas.io.sql as psql, simplekml, pandas as pd,matplotlib.pyplot as plt

class AllTemplates(object): #class to load all templates
    """Class for visualization purposes"""
    def __init__(self,CorDB='Corrs.db',templatePath='EventWaveForms',condir='ContinousWaveForms',templateKey='TemplateKey.csv',ArcDB='Arc.db'):
        self.__dict__.update(locals())
        self.templateKey=pd.read_csv(templateKey)
        self.temps=[0]*len(templateKey)
        for a in self.templateKey.iterrows():
            staKey=pd.read_csv(a[1]['STATIONKEY'])
            self.temps[a[0]]=[0]*len(staKey)
            for b in staKey.iterrows():
                try:
                    path=glob.glob(os.path.join(templatePath,a[1]['NAME'],b[1]['NETWORK']+'.'+b[1]['STATION'],'*'))[0]
                except:
                    print b[1]['STATION']
                    continue
                TR=obspy.core.read(os.path.join(path,'*.sac'))
                TR.filter('bandpass',freqmin=1,freqmax=10,corners=2,zerophase=True)
                trim=np.load(glob.glob(os.path.join(path,'*.tms.npy'))[0])
                self.temps[a[0]][b[0]]=TR.trim(starttime=obspy.core.UTCDateTime(trim[0]),endtime=obspy.core.UTCDateTime(trim[1]))
    def __getitem__(self,index): # allow indexing
        return self.temps[index]
    def plotAllTemplates(self,index):
        """plot all of template index's templates"""
        for a in range(len(self)):
            plt.subplot(3,3,a)
            
"""
#Functions for writing to output formats such as hypoDD phase formats, hypoInverse phase formats, and kml (google earth)
#__________________________________________
"""
def Get_Ortho(az1,az2,dip1,dip2): #function to return a dip and azimuth orthogonal to two given dips and azimuths in R3
    pass # work on this later 
 
def readSum(sumfile):
   
    """ Read a sum file from hyp2000 and return lat,long,depth,mag, and RMS and TSTMP as pd dataframe"""
    lines=[line.rstrip('\n') for line in open(sumfile)]
    DF=pd.DataFrame(index=range(len(lines)))
    DF['Lat']=float
    DF['Lon']=float
    DF['DateString']=str
    DF['Dep']=float
    DF['RMS']=float
    DF['ELAz']=float #largest error azimuth
    DF['HozError']=float
    DF['VertError']=float
    DF['MaxError']=list
    DF['IntError']=list
    DF['MinError']=list
    for a in range(len(lines)):
        DF['MaxError'][a]=[[0]*3]
        DF['IntError'][a]=[[0]*3]
        DF['MinError'][a]=[[0]*3]
        l=lines[a]
        DF.Lat[a]=float(l[16:18])+(float(l[19:21].replace(' ','0'))+float(l[21:23].replace(' ','0'))/100)/60
        DF.Lon[a]=-float(l[23:26])+-(float(l[27:29].replace(' ','0'))+float(l[29:31].replace(' ','0'))/100)/60 # not yet able to handle entire glob, keep tract of negative signs
        DF.DateString[a]=l[0:4]+'-'+l[4:6]+'-'+l[6:8]+'T'+l[8:10]+'-'+l[10:12]+'-'+l[12:14]+'.'+l[14:16]
        DF.Dep[a]=float(l[31:34].replace(' ','0').replace('-','0'))+float(l[34:36].replace(' ','0'))/100
        DF.RMS[a]=float(l[48:50].replace(' ','0'))+float(l[50:52].replace(' ','0'))/100   
        #DF.ELAz[a]=float(l[52:55].replace('','0'))
        DF.HozError[a]=float(l[85:87].replace(' ','0'))+float(l[87:89].replace(' ','0'))/100.0
        DF.VertError[a]=float(l[89:91].replace(' ','0'))+float(l[91:93].replace(' ','0'))/100.0
        #DF.MaxError[a]=[float(l[52:55]),float(l[55:57]),float(l[57:59])+float(l[59:61])/100]
        #DF.IntError[a]=[float(l[61:64]),float(l[64:66]),float(l[66:68])+float(l[68:70])/100]
        #DF.MinError[a]=[float(l[76:78])+float(l[78:80])/100]
    return DF

def writeKMLFromDF(DF,outname='map.kml'):
    """ Write a KML file from a pandas data frame with the same format as readSum output
    """
    kml = simplekml.Kml(open=1)
    for a in DF.iterrows():             
        pnt=kml.newpoint()
        pnt.name=str(a[1].DateString)
        pnt.coords=[(a[1].Lon,a[1].Lat)]
    kml.save(outname)
           
def writeKMLFromHypInv(hypout='sum2000',outname='hypoInv.kml'):
    """Uses simplekml to create a KML file (used by Google Earth, Google Maps, etc)
    of the results from hypoInverse 2000"""
    C=[]
    with open(hypout,'r') as openfile:
        for line in openfile:
            C=C+[line[0:31]]
    kml = simplekml.Kml(open=1)
    for a in C:
        spl=a.replace(' ','0')
        lat=float(spl[16:18])+(float(spl[19:21])/60+float(spl[21:23])/(100.0*60))
        lon=-float(spl[23:26])+-(float(spl[27:29])/60.0+float(spl[29:31])/(100.0*60)) #assume negative sign needs to be added for west              
        pnt=kml.newpoint()
        pnt.name=str(int(a[0:10]))
        pnt.coords=[(lon,lat)]
    kml.save(outname)

def writeKMLfromHYPInput(hypin='test.pha',outname='hypoInInv.kml'):
    with open(hypin,'rb') as infile:
        kml = simplekml.Kml(open=1)
        cou=1
        for line in infile:
            if line[0:6]!='      ' and len(line)>10:
            #print UTCtstmp
            #print line[17:34]
                pass
            elif line[0:6]=='      ':
            #print 'terminator line'                    
                lat=float(line[14:16])+(float(line[17:19])/60+float(line[19:21])/(100.0*60))
                lon=-float(line[21:24])+-(float(line[25:27])/60.0+float(line[27:29])/(100.0*60))
                pnt=kml.newpoint()
                pnt.name=str(cou)
                pnt.coords=[(lon,lat)]
                cou+=1
        kml.save(outname)
            
def writeKMLFromHypDD(hypreloc='hypoDD.reloc',outname='hypo.kml'):
    """Uses simplekml to create a KML file (used by Google Earth, Google Maps, etc)
    of the results from hypoDD"""
    points=np.array(np.genfromtxt(hypreloc))
    kml = simplekml.Kml(open=1)
    for a in points:
        pnt=kml.newpoint()
        pnt.name=str(int(a[0]))
        pnt.coords=[(a[2],a[1])]
    kml.save(outname)

def writeHypoFromDict(TTdict,phase='P',output='all.phases'):
    """ Function to write a hyp phase input file based on dictionary or list of dictionaries
    where station name are keys and the timestamps are values
    """
    if isinstance(TTdict,dict):
        TTdict=[TTdict] #make iterable
    if isinstance(TTdict,pd.core.series.Series):
        TTdict=TTdict.tolist()
    if not isinstance(TTdict,list):
        raise Exception('TTdict type not understood, must be python dictionary or list of dictionaries')
    with open(output, 'wb') as out:
        out.write('\n') #write intial end character
        for a in TTdict:
            if len(a)>3:
                for key in a.keys():
                    line=_makeSHypStationLine(key,'ZENZ','TA',a[key],'P')
                    out.write(line)
                termline=_makeHypTermLine(a)
                out.write(termline)
                out.write('\n')
            
def _makeHypTermLine(TTdict):
    mintime=obspy.core.UTCDateTime(np.min(np.array(TTdict.values())))
    space=' '
    hhmmssss=mintime.formatIRISWebService().replace('-','').replace('T','').replace(':','').replace('.','')[8:16]
    #lat,latminute=str(abs(int(m[0]))),str(abs(60*(m[0]-int(m[0])))).replace('.','')[0:4]
    #lon,lonminute=str(abs(int(m[1]))),str(abs(60*(m[1]-int(m[1])))).replace('.','')[0:4]
    lat,latminute,lon,lonminute=' ',' ',' ',' '
    trialdepth='  400'
    endline="{:<6}{:<8}{:<3}{:<4}{:<4}{:<4}{:<5}\n".format(space,hhmmssss,lat,latminute,lon,lonminute,trialdepth)
    return endline
    

def _makeSHypStationLine(sta,cha,net,ts,pha):
    Ptime=obspy.core.UTCDateTime(ts)
    datestring=Ptime.formatIRISWebService().replace('-','').replace('T','').replace(':','').replace('.','')
    YYYYMMDDHHMM=datestring[0:12]
    ssss=datestring[12:16]
    end='01'
    ty=' %s 0' % pha
    line="{:<5}{:<3}{:<5}{:<3}{:<13}{:<80}{:<2}\n".format(sta,net,cha,ty,YYYYMMDDHHMM,ssss,end)
    return line
                

def writeKMLFromArcDF(df,outname='Arc.kml'):
    kml = simplekml.Kml(open=1)
    for a in df.iterrows():
        pnt=kml.newpoint()
        pnt.name=str(int(a[0]))
        pnt.coords=[(a[1]['verlon'],a[1]['verlat'])]
    kml.save(outname)
    
def hypoinverseSumtoPhase(hypsum='sum2000'):
    """
    function to convert hypoinvers output into hypoDD input
    """
    
    
    
"""
#misc. functions, used for odds and ends, mostly called by other functions and classes
#__________________________________________________________________________________________
"""
        
        
def loadContinousData(starttime,endtime,station,Condir='ContinousWaveForms'):
    """
    Function to load continous data from the detex directory structure
    
    Parameters
    ------
    
    starttime : number or str
        An obspy.core.UTCDateTime readable objects (see docs for details)
    endttime : number or str
        An obspy.core.UTCDateTime readable objects (see docs for details)
    station : str
        Name of station
    Condir : str
        Path to continuous data directory
    
    Returns
    ---------
    An obspy stream object
    """
    t1=obspy.core.UTCDateTime(starttime)
    t2=obspy.core.UTCDateTime(endtime)
    ST=obspy.core.Stream()
    yearRange=range(t1.year,t2.year+1)
    for year in yearRange:
        if t1.year==year and t2.year == year:
            if t1.julday==t2.julday:
                hourRange=range(t1.hour,t2.hour+2)
                traces=[glob.glob(os.path.join(Condir,station,str(year),'%03d'%t1.julday,'*%02d.pkl'%x)) for x in hourRange]
            else:
                jdayRange=range(t1.julday,t2.julday+1)
                traces=[glob.glob(os.path.join(Condir,station,str(year),'%03d'%x,'*')) for x in jdayRange]
        else:
            raise Exception('starttime and endtime not in same year, multiple years not yet supported')
    traces=[x for y in traces for x in y] #flatten glob list
    for trace in traces:
        ST+=obspy.core.read(trace)
    ST.merge(method=1)
    ST.trim(starttime=t1,endtime=t2)
    return ST
    
def getContinousDataLength(Condir='ContinousWaveForms',numToRead=10):
    """
    Function to randomly read  in several hours from different stations and estimate the duration is seconds
    of each continous data block
    
    numToRead is the number of traces to read for each station in determining continous data length
    """
    stations=glob.glob(os.path.join(Condir,'*'))
    ledict={}
    for sta in stations:
        lenlist=[]
        years=glob.glob(os.path.join(sta,'*'))
        jdays=[glob.glob(os.path.join(x,'*')) for x in years]
        while len(lenlist)<numToRead:
            rannum=int(round(np.random.rand()*len(years)))
            traces=glob.glob(os.path.join(random.choice(jdays[rannum-1]),'*'))
            st=obspy.core.read(random.choice(traces))
            lenlist.append(st[0].stats.endtime-st[0].stats.starttime)
        ledict[sta]=np.median(lenlist)
    lenlist=ledict.values()
    if not all([abs(x-lenlist[0])<1 for x in lenlist]): #if difference in median lengths are greater than 1 second accross all stations abort
        deb(lenlist)
        raise Exception('Not all the channels have the same length for continous data, aborting opperation')
    else:
        return lenlist[0]
        
def getEveDataLength(EveDir='EventWaveForms',numToRead=10):
    """
    Same as getContinousDataLength but with template events
    """
    eves=glob.glob(os.path.join(EveDir,'*'))
    eves=list(set(eves))
    ledict={}
    for eve in eves:
        lenlist=[]
        for traceFile in glob.glob(os.path.join(eve,'*')):
            st=obspy.core.read(traceFile)
            length=st[0].stats.endtime-st[0].stats.starttime
            lenlist.append(length)
        ledict[eve]=np.median(lenlist)
        if len(ledict.keys())>numToRead:
            break
    lenlist=ledict.values()
    if not all([abs(x-lenlist[0])<1 for x in lenlist]): #if difference in median lengths are greater than 1 second accross all stations abort
        deb(lenlist)
        raise Exception('Not all the channels have the same length for continous data, aborting opperation')
    else:
        return lenlist[0]    
    

    
def lookin(direct): # Function to avoid having to write too with glob module
    inside=glob.glob(os.path.join(direct,'*'))
    return inside
    
def saveSQLite(DF,CorDB,Tablename,silent=True): # To save to SQLite database
    """ 
    Basic function to save pandas dataframe to SQL
    
    DF is the data frame
    
    CorDB is a string of the name of the database
    
    Tablename is the name of the table to which DF will be saved
    
    silent will suppress the output of the SQL database
    
    """

    with sqlite3.connect(CorDB, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        if os.path.exists(CorDB):
            detex.pandas_dbms.write_frame(DF, Tablename, con=conn, flavor='sqlite', if_exists='append')
        else:
            detex.pandas_dbms.write_frame(DF, Tablename, con=conn, flavor='sqlite', if_exists='fail')       
def DoldDB(CorDB): # Check if CorDB exists, if so delete
    if os.path.exists(CorDB):
        os.remove(CorDB)
        
        
def loadSQLite(corDB,tableName,sql=None):     
    """
    Load contents of a SQLite database as a dataframe
    
    corDB is a string of the name of the database
    
    tablename is the table which should be loaded
    
    sql allows user to pass sql arguments to filter results
    """              
    if sql==None:
        sql='SELECT %s FROM %s' % ('*', tableName)
    #sys.exit(1)
    with sqlite3.connect(corDB, detect_types=sqlite3.PARSE_DECLTYPES) as con:
        df=psql.read_sql(sql, con)
    return df
        
def parseEvents(EveDir='EventWaveForms'):
    """ Returns a files reference to each sac file in EveDir
    """
    if not os.path.isdir(EveDir):
        raise Exception('target file: '+EveDir+' does not exist')
    init=1
    for idf in lookin(EveDir):
        for sta in lookin(idf):
            for time in lookin(sta):
                for files in lookin(time):
                    if init==1:
                        R=[files]
                        init=0
                    else:
                        R=R+[files]
    return R
    
def checkExists(filename):
    if not type(filename)==list or type(filename)==tuple:
        filename=[filename]
    for a in filename:
        if not os.path.exists(a):
            raise Exception(a+' does not exist')
                
def parseCorrs(Cor='Corrs'):
    """ Returns a files reference to each np file in Cor
    """
    init=1
    for idf in lookin(Cor):
        for sta in lookin(idf):
            for year in lookin(sta):
                for jday in lookin(year):
                    for chan in lookin(jday):
                        for files in glob.glob(os.path.join(chan,'*cc.npy')):
                            if init==1:
                                R=[files]
                                init=0
                            else:
                                R=R+[files]
    return R

def _trimStream(TR,UTC1,UTC2):
    D=TR.slice(starttime=UTC1,endtime=UTC2)
    return D
    

#### ALL function to pick travel times manually from a dataframe formatted by Detex.Results.CorResults
# Consider making this a class and putting it at the end    
#def pickTravelTimes(DF, EveDir='EventWaveForms',templateKey='TemplateKey.csv',stationKey='StationKey.csv',PickChannel='BHZ',
#                    b4time=15,aftime=60,prefilt=[.05,.1,15,20], opType='vel',cli='IRIS',useSavedPicks=True):
#    """ 
#    Function to make travel time picks (p and S) on all events in DF, DF is output of detex.results.CorResults 
#    then make hypoInverse input files
#    
#    """
#    stakey=pd.read_csv(stationKey)
#
#    client=obspy.fdsn.Client(cli)
#    DFPick=pd.DataFrame(index=range(len(DF)),columns=['Picks','Time','Template','Mag','PreOtime'])
#    DFPick.Picks=[{} for x in range(len(DF))]
#    if os.path.exists('TempPick.pkl') and useSavedPicks:
#        DFPick=pd.read_pickle('TempPick.pkl')
#    DF.reset_index(inplace=True,drop=True) #make sure index is ordered sequentially
#    for a in DF.iterrows():
#        print a[1].Template
#        UTCstart=obspy.core.UTCDateTime(a[1].PreOtime-b4time)
#        UTCend=obspy.core.UTCDateTime(a[1].PreOtime+aftime)
#        if len(DFPick.Picks[a[0]].keys())>2:
#            continue
#        DFPick.Picks[a[0]]={}
#        DFPick.Time[a[0]]=a[1].Time
#        DFPick.Mag[a[0]]=a[1].Mag
#        DFPick.Template[a[0]]=a[1].Template
#        DFPick.PreOtime[a[0]]=a[1].PreOtime
#        for sta in stakey.iterrows():
#            picks=[0,0]
#            TR=_tryDownloadData(sta[1].NETWORK,sta[1].STATION,'BH*','*',
#                                UTCstart,UTCend,client)
#            if TR:
#                TR=_removeInstrumentResposne(TR,prefilt,opType) 
#                Pks=detex.streamPick.streamPick(TR)
#                for b in Pks._picks:
#                    if b:
#                        if b.phase_hint=='S':
#                            picks[1]=b.time
#                        if b.phase_hint=='P':
#                            picks[0]=b.time
#                if picks!=[0,0]:
#                    DFPick.Picks[a[0]][sta[1].NETWORK+'.'+sta[1].STATION]=picks
#        _writeTemp(DFPick)
#                
#    return DFPick

def _writeTemp(DFPick):
    DFPick.to_pickle('TempPick.pkl')
    
def writePhaseDD(DFPick,name='DtexDD.pha',hypoInverseSumFile='sum2000',useS=False,useP=True):
    """ Write a phase file used by ph2dt (a program of hypoDD), inputs are the DFPicks file from pickTravelTimes and the
    summary file from hypoinverse
    """
    tb=DFPick
    if not isinstance(tb,pd.DataFrame): # make sure self.table exists and is dataframe
        raise Exception('Table not found, no events were detected and clustered')
    hypSum=detex.util.readSum(hypoInverseSumFile)
    if len(hypSum) != len(DFPick):
        raise Exception('hypo inverse summary file not the same length as DFPick file, make sure they contain exactly the same events')
    DFPick.sort(columns='Time')
    DFPick.reset_index(drop=True,inplace=True)
    hypSum.sort(columns='DateString')
    hypSum.reset_index(drop=True,inplace=True)
    with open(name,'wb') as pha:
        for a in tb.iterrows(): # Loop through CorResults.table
            hyprow=hypSum.iloc[a[0]]
            header=_makeHeader(a,hyprow)
            pha.write(header)
            Stations=a[1].Picks.keys()
            for b in Stations:
                Ptime=obspy.core.UTCDateTime(a[1].Picks[b][0]).timestamp
                Stime=obspy.core.UTCDateTime(a[1].Picks[b][1]).timestamp
                if Ptime-a[1]['PreOtime'] <0 and Ptime!=0 or Stime-a[1]['PreOtime'] <0 and Stime!=0  : #Insure pick is not before origin time reported in templateKey
                    raise Exception('Pick before reported Origin time for ' + a[1]['Template'] + ' ' + b )
                #write S time
                if useS and Stime>0.1:
                    lineData=[b.split('.')[1],Stime-a[1]['PreOtime'],1,'S']
                    line='%s %.4f %02d %s \n' % tuple(lineData)
                    pha.write(line)
                #Write P time
                if useP and Ptime>0.1:
                    print Ptime
                    lineData=[b.split('.')[1],Ptime-a[1]['PreOtime'],1,'P']
                    line='%s %.4f %02d %s \n' % tuple(lineData)
                    pha.write(line)
                    #print (b + ' not found in table')
def _makeHeader(event,hyprow):
    template=event[1].Template
    oTime=obspy.core.UTCDateTime(event[1].PreOtime)
    headDat=[oTime.year,oTime.month,oTime.day,oTime.hour,oTime.minute,oTime.second+oTime.microsecond/1000000.0,hyprow['Lat'],hyprow['Lon'],hyprow['Dep'],event[1]['Mag'],event[0]+1]
    #headDat=[oTime.year,oTime.month,oTime.day,oTime.hour,oTime.minute,oTime.second+oTime.microsecond/1000000.0,38.94748,-107.556892,template['DEPTH'],event[1]['Mag'],event[0]+1]
    header='# %04d %02d %02d %02d %02d %.4f %.5f %.5f %.2f %.2f 0.0 0.0 0.0 %01d \n' % tuple(headDat)
    return header 
                       
    
def writePhaseHyp(DFPick,name='Dtex.pha',fix=0,depth=100,useTempLatLon=False,removeTemp=False,useP=True,useS=True):
    """ Write a y2k complient phase file used by hypoinverse 2000, format defined on 
    page 113 of the manual for version 1.39
    if fix==0 nothing is fixed, fix==1 depths are fixed, fix==2 hypocenters fixed, fix==3 hypocenters and origin time fixed
    depth/100 is the starting depth for hypoinverse in km
    """
    tb=DFPick
    if 'time' in tb.columns:
        tb['PreOtime']=tb['time']
    with open(name,'wb') as pha:
        pha.write('\n')
        for a in tb.iterrows(): # Loop through CorResults.table
            Stations=a[1].Picks.keys()
            for b in Stations:
                Ppick=obspy.core.UTCDateTime(a[1].Picks[b][0]).timestamp
                Spick=obspy.core.UTCDateTime(a[1].Picks[b][1]).timestamp
                if Ppick-a[1]['PreOtime'] < 0 and Ppick !=0.0 or Spick-a[1]['PreOtime'] < 0 and Spick!=0: #Insure pick is not before origin time reported in templateKey
                    
                    deb([a,b,Spick,Ppick])                 
                    raise Exception('Pick before reported Origin time for ' + a[1]['Template'] + ' ' + b )
                #lineData=[b.split('.')[1],a[1][b]-a[1]['PreOtime'],1,'S']
                if len(b.split('.'))>1:
                    net=b.split('.')[0]
                    sta=b.split('.')[1]
                elif len(b.split('.'))==1: # If no network in station name assume TA
                    net='TA'
                    sta=b
                if a[1].Picks[b][0] != 0 and useP:
                    line=_makeSHypStationLine2(sta,'ZENZ',net,a[1].Picks[b][0],'P') #First P waves
                    pha.write(line)
                if a[1].Picks[b][1] != 0 and useS:
                    line=_makeSHypStationLine2(sta,'ZENZ',net,a[1].Picks[b][1],'S') #Then S waves
                    pha.write(line)
            #print (b + ' not found in table')
            el=_makeHypTermLine2(a[1]['PreOtime'],fix,depth,a,useTempLatLon)
            pha.write(el)
            pha.write('\n')
    if removeTemp and os.path.exists('TempPick.csv'):
        os.remove('TempPick.csv')
            
                       
def _makeSHypStationLine2(sta,cha,net,ts,pha):
    Ptime=obspy.core.UTCDateTime(ts)
    datestring=Ptime.formatIRISWebService().replace('-','').replace('T','').replace(':','').replace('.','')
    YYYYMMDDHHMM=datestring[0:12]
    ssss=datestring[12:16]
    end='01'
    ty=' %s 0' % pha
    line="{:<5}{:<3}{:<5}{:<3}{:<13}{:<80}{:<2}\n".format(sta,net,cha,ty,YYYYMMDDHHMM,ssss,end)
    return line
    
def _makeHypTermLine2(Otime,fix,depth,DFrow,useTempLatLon):
    space=' '
    if fix==0:
        fixchar=' '
    elif fix==1:
        fixchar='-'
    elif fix==2:
        fixchar='X'
    elif fix==3:
        fixchar='O'
    UTC=obspy.core.UTCDateTime(Otime)
    hhmmssss=UTC.formatIRISWebService().replace('-','').replace('T','').replace(':','').replace('.','')[8:16]
    #lat,latminute=str(abs(int(m[0]))),str(abs(60*(m[0]-int(m[0])))).replace('.','')[0:4]
    #lon,lonminute=str(abs(int(m[1]))),str(abs(60*(m[1]-int(m[1])))).replace('.','')[0:4]
    if useTempLatLon:
        pass
        print('useTempLatLon not yet programmed')
        #lat,latminute=str(abs(int(DFrow.))),str(abs(60*(m[0]-int(m[0])))).replace('.','')[0:4]
    else:
        lat,latminute,lon,lonminute=' ',' ',' ',' ' #dont give trial lats/lons
    endline="{:<6}{:<8}{:<3}{:<4}{:<4}{:<4}{:<5}{:<1}\n".format(space,hhmmssss,lat,latminute,lon,lonminute,depth,fixchar)
    return endline    
                
 ##### End special pick functions to be made into class
                               
            
def _removeInstrumentResposne(st,prefilt,opType):
    st.detrend('linear')# detrend
    for a in st: #make sure each trace is even as to not slow down fft
        if len(a.data)%2!=0:
            a=a[-1:]
    try: 
        st.remove_response(output=opType,pre_filt=prefilt)
    except:
        print ('RemoveResponse Failed for %s,%s, not saving' %(st[0].stats.network,st[0].stats.station))
        st=False
    return st        
        
def _tryDownloadData(net,sta,chan,loc, utcStart,utcEnd,client): # get data, return False if fail
    try:
        st=client.get_waveforms(net,sta,loc,chan,utcStart,utcEnd,attach_response=True)
        return st
    except:
        print ('Download failed for %s.%s %s from %s to %s' % (net,sta,chan,str(utcStart),str(utcEnd)))
        return False    
            
def trimTemplates(EveDir='EventWaveForms',templatekey='TemplateKey.csv', pickDF='EventPicks.pkl'):
    """
    Uses streamPicks to parse the templates and allow user to manually pick starttimes for events.
    Currently seperate P and S picks are not supported and only the first pick (whichever phase it may be)
    is recorded as the event starttime
    
    Parameters
    -------------
    EveDir : str
        Event Directory with the structure created by detex.getevents.getAllEvents
    templatekey : str
        Path to the template key
    pickDF : str 
        Name for picks to be saved as. If the file already exists it will be read and all picks already
        made will be skipped
    """
 
    temkey=pd.read_csv(templatekey)
    evefiles=glob.glob(os.path.join(EveDir,'*'))
    ef=set(temkey.NAME.values).intersection(set([os.path.basename(x) for x in evefiles])) #get events that both exist and are in template key
    if os.path.exists(pickDF): #if a pickDF already exists then load it
        DF=pd.read_pickle(pickDF)
        if len(DF)<1: #if empty then delete
            os.remove(pickDF)
            DF=pd.DataFrame()
            Eves=[os.path.join(EveDir,x) for x in list(ef)]
        else:
            wfs=ef-set(DF['Name'])
            Eves=[os.path.join(EveDir,x) for x in list(wfs)]
    else:
        DF=pd.DataFrame()
        Eves=[os.path.join(EveDir,x) for x in list(ef)]
    for a in Eves:
        waveforms=glob.glob(os.path.join(a,'*'))
        for wf in waveforms:
            TR=obspy.core.read(wf)
            Pks=detex.streamPick.streamPick(TR)
            d1={} #initialize blank dict
            saveit=0
            for b in Pks._picks:
                if b:
                    d1[b.phase_hint]=b.time.timestamp
                    saveit=1
            if saveit:
                d1['Starttime'],d1['Endtime']=[min(d1.values()),max(d1.values())]
                d1['Station']=TR[0].stats.network+'.'+TR[0].stats.station
                d1['Path']=wf
                d1['Name']=os.path.basename(a)
                
                DF=DF.append(d1,ignore_index=True)
            else:
                print ('no picks passed, cant trim stream object')
            if not Pks.KeepGoing:
                print 'aborting picking, progress saved'
                DF.to_pickle(pickDF)
                return None
    DF.to_pickle(pickDF)
    
def makePKS(inputFile,pickDF='EventPicks.pkl',EveDir='EventWaveForms'):
    """
    Simple function to make the pks dataframe used by detex.xcorr.correlate to define start and stop times of waveform templates
    
    Parameters
    ----------
    inputFile : str
        The path the input file. Input file should be a csv with the same format as the following example:
        
        Station,Event,Phase,UTC
        TA.M17A,2009-04-03T15-39-27,P,1238773167.0
        TA.M17A,2009-04-03T15-39-27,S,2009-04-03T15-39-40
        ......
    
        The Station feild is a string Network.Station
        The Event feild is the name of the event in the TemplateKey.csv file
        The Phase field is the phase which the time pick is for
        The UTC field is any obspy.core.UTCDateTime redable format (timestamp, datestring, etc.)
        
        
    pickDF : str
        The name of the picks df to save
        
    EveDir : str
        Path to the event directory where the template waveforms are stored
        
    """

    df=pd.read_csv(inputFile)
    if set(df.columns) != set(['Station','Event','Phase','UTC']):
        raise Exception ('%s does not have the appropriate headers, make sure the first line of the file is Station,Event,Phase,UTC' % inputFile )
    DFout=pd.DataFrame(columns=['Endtime','Name','Path','Starttime','Station'])
    for event in list(set(df.Event.values)):
        temdf=df[df.Event==event]
        for station in list(set(temdf.Station)):
            temdf2=temdf[temdf.Station==station]
            pathglob=glob.glob(os.path.join(EveDir,event,station+'*'))
            if len(pathglob)<1:
                path=''
            else:
                path=pathglob[0]
            UTCs=[obspy.core.UTCDateTime(x).timestamp for x in temdf2.UTC]
            Endtime=max(UTCs)
            Starttime=min(UTCs)
            picDic={'Path':path,'Name':event,'Station':station,'Endtime':Endtime,'Starttime':Starttime}            
            for num,row in temdf2:
                picDic[row.Phase]=obspy.core.UTCDateTime(row.UTC).timestamp
            DFout.append(picDic,ignore_index=True)
    DFout.to_pickle(pickDF)

    
    
    
def sortFiles(indir='Trigs',outDF='sortedTrigs.pkl',filt=[1,10,2,True]):
    """
    function used to sort through triggered files and return a pandas dataframe that orders the files based on average
    amplitude
    """
    InFiles=glob.glob(os.path.join(indir,'*'))
    DF=pd.DataFrame(index=range(len(InFiles)),columns=['FileName','AveAmplitude','EventHour'])
    DF.FileName=InFiles
    DF.AveAmplitude=0.0
    for a in DF.iterrows():
        TR=obspy.core.read(a[1].FileName)
        DF.AveAmplitude[a[0]]=_getStreamAmplitude(TR,filt)
        DF.EventHour[a[0]]=TR[0].stats.starttime.hour
        
        if a[0]%300==0:
            print a[0]
    DF.sort(columns='AveAmplitude',inplace=True,ascending=False)
    DF.reset_index(drop=True,inplace=True)
    DF.to_pickle(outDF)
    return DF        
def _getStreamAmplitude(TR,filt):
    TR.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
    amps=[0]*len(TR)
    for a in range(len(TR)):
        amps[a]=np.nanmax(np.abs(TR[a].data))
    amps=np.array(amps)
    amps[amps.argmax()]=0.0
    av=np.mean(np.array(amps))
    return av
    
def makeHypStationFile(DF,outname='TAall.sta'):
    with open(outname,'wb') as stafil:
        for a in DF.iterrows():
            linefill=(a[1].STATION,a[1].NETWORK,int(a[1].LAT),(a[1].LAT % int(a[1].LAT))*60,
                      int(abs(a[1].LON)),(abs(a[1].LON) % int(abs(a[1].LON)))*60,a[1].ELEVATION)
            line= '%s  %s  ENZ  %02d %07.4f %03d %07.4f %04d .0  P  0.00  0.00  0.00  0.00 0 0.00 01\n' % linefill
            stafil.write(line)
            
def convertOldDtexDirs(condir='ContinousWaveForms',chans=['BHZ','BHN','BHE']):
    os.rename(condir,condir+'tmp')
    hourange=['%02d'%x for x in range(24)]
    # work on continous waveforms first
    stations=glob.glob(os.path.join(condir+'tmp','*'))
    for sta in stations:
        orderedlist=[]
        years=glob.glob(os.path.join(sta,'*'))
        for year in years:
            jdays=glob.glob(os.path.join(year,'*'))
            for jday in jdays:
                for hour in hourange:
                    TR=obspy.core.Stream()
                    try:
                        for chan in chans:
                            streamstring=os.path.basename(sta)+'.'+chan+'_'+os.path.basename(year)+'-'+os.path.basename(jday)+'T'+hour+'.sac'
                            TR+=obspy.core.read(os.path.join(jday,chan,streamstring))
                    except IOError:
                        pass
                    if len(TR)>1:
                        TR.sort()
                        
                        Savestr=os.path.basename(sta)+'.'+os.path.basename(year)+'-'+'%03d'%int(os.path.basename(jday))+'T'+hour+'.pkl'
                        saveDir=os.path.join(condir,os.path.basename(sta),os.path.basename(year),'%03d'%int(os.path.basename(jday)))
                        if not os.path.exists(saveDir):
                            os.makedirs(saveDir)
                        TR.write(os.path.join(saveDir,Savestr),'pickle')
                        
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      
    Taken from:
    http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])                            
                
def deb(varlist):
    global de
    de=varlist
    sys.exit(1)                       
        
              
#def writeNLLPhases()