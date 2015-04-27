# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:55:01 2014

@author: Derrick
"""
import os, glob, numpy as np, obspy, matplotlib.pyplot as plt, detex, pandas as pd, sqlite3, sys
import pandas.io.sql as psql, numbers, scipy

class allTheFunctions(object):
    """ Nearly all definitions for other classes are contained in allTheFunctions. allTheFunctions is then used as
    the parent class for CorResults
    """    
    def _loadCorDB(self,corDB,template):
        self._checkCoef()
        if self.trigCon==0:
            cond='Coef'
        elif self.trigCon==1:
            cond='STA_LTACoef'       
        sql='SELECT %s FROM %s WHERE %s="%s" AND %s > %s' % ('*', 'cor_df','Template',template,cond,self.trigParameter)
        #sys.exit(1)
        with sqlite3.connect(corDB, detect_types=sqlite3.PARSE_DECLTYPES) as con:
            df=psql.read_sql(sql, con)
        return df
        
    def _checkCoef(self):
        if self.trigCon==0:
            if self.trigParameter>1 or self.trigParameter<0:
                raise Exception('invalid value for trigParameter: 0<trigParameter<1 for TrigCon = 0')
        if self.trigParameter==1:
            if self.trigParameter<=1 and self.trigParameter != 0:
                raise Exception('invalid value for trigParameter: trigParameter>1 for TrigCon = 1 (recomended value of 2 or greater generally)' )
    
    def _checkExists(self,filename):
        if not type(filename)==list or type(filename)==tuple:
            filename=[filename]
        for a in filename:
            if not os.path.exists(a):
                raise Exception(a+' does not exist')
                
    def _cleanStream(self):
        a=0
        while a<len(self.Stream):
            if self.Stream[a]==0:
                del self.Stream[a]
            else:
                a+=1
                
    def _verifyDEvents(self,CORDInsatnce,CorResInstance,eventStremInstance,deventInstance,itdex):
        verTimeTol=CorResInstance.verificationTimeTolerance # amount of seconds to float
        TSTMP=obspy.core.UTCDateTime((deventInstance.table['PreOtime'].values[0])).timestamp #convert time string to time stamp
        if os.path.exists(CorResInstance.ArcDB):
            try:
                df=self._loadArcDB(verTimeTol,TSTMP,CorResInstance.ArcDB)
                if not df.empty:
                    return df
            except:
                print 'something wrong with arc database, skipping verification'
                return None
        else:
            return None

    def _loadArcDB(self,verTimeTol,TSTMP,ArcDB):                   
        sql='SELECT %s FROM %s WHERE %s>=%0.4f AND %s<=%0.4f' % ('*', 'arc','STMP',TSTMP-verTimeTol,'STMP',TSTMP+verTimeTol)
        with sqlite3.connect(ArcDB, detect_types=sqlite3.PARSE_DECLTYPES) as con:
            df=psql.frame_query(sql, con)
        return df

    
    def _testDuplicateStations(self,Devents,iteration,templateName): # Make sure each station is represented no more than 1 time
        Devents.sort(columns=['Coef'], inplace=True)
        Devents.reset_index(drop=True,inplace=True)
        test=Devents.duplicated(subset='Sta')
        g = Devents.groupby((~test).cumsum())
        Devents=Devents.loc[g.Coef.idxmax()]
        if any(test):
            Devents.reset_index(drop=True,inplace=True)
            print (' Event %d of template %s has duplicate station entries' % (iteration, templateName))
            print ('Keeping 1 entry from each station with highest correlation coeficient, !Proceed with Caution!')
        return Devents

    
    def _createStationDict(self,stationKey): # Creae dictionary with channels and spatial coordinates of each staion
        keyDict={}
        for a in stationKey.values:
            keyDict[a[0]+'.'+a[1]]=[a[4],a[5],a[6],a[7].split('-')]
        return keyDict
    
    def _timeDict(self,Devent,allstations): #create dictionary with all stations, overide if data avaliable
        TD={}
        #deb([Devent,allstations])
        for a in range(len(allstations)):
            TD[allstations[a]]=np.nan
            
        for a in range(len(Devent)):
            TD[Devent['Sta'][a].split('.')[1]]=Devent['STMP'][a]
            TD[Devent['Sta'][a]]=Devent['STMP'][a]
        return TD

    def _getTemplateWaveForms(self,sta,chans,eventTemplate,templatePath,filt): # Get each template waveform
        uperDir=glob.glob(os.path.join(templatePath,eventTemplate,sta,'*'))[0]
        waveforms=[0]*len(chans)
        for a in range(len(chans)):
            #print wrdird
            wfdir=glob.glob(os.path.join(uperDir,'*'+chans[a]+'*.sac'))[0]
            trim=glob.glob(os.path.join(uperDir,'*.tms.npy'))
            wf=obspy.core.read(wfdir)
            if isinstance(self.filt,np.ndarray):
                wf.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
            if len(trim)>0:
                trimvals=np.load(trim[0])
                wf=wf.slice(starttime=obspy.core.UTCDateTime(trimvals[0]),endtime=obspy.core.UTCDateTime(trimvals[1]))
            waveforms[a]=wf[0].data
        return waveforms, len(waveforms[0])
        
    def _getContinousWaveForms(self,sta,chans,eventTemplate,templatePath,time,conpath,leTemp,temps,filt): # Load the continous waveform
        waveforms=[0]*len(chans)
        #CCwaveform=[0]*len(chans)
        for a in range(len(chans)):
            wfPath=glob.glob(os.path.join(conpath,sta, str(time.year),str(time.julday),chans[a],'*T'+'%02d' % time.hour+'.sac'))[0]
            WF=obspy.core.read(wfPath)
            if isinstance(self.filt,np.ndarray):
                WF.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
            potentialWF=WF.slice(starttime=time-5*WF[0].stats.delta) # Take 2 extra samples to find best fit
            sr=potentialWF[0].stats.sampling_rate
            starttime=potentialWF[0].stats.starttime.timestamp
            potentialWF=potentialWF[0].data[0:leTemp+11]
            waveforms[a]=self._findTrueFit(potentialWF,temps[a])
            
            
        return waveforms, starttime, sr
        
    def _findTrueFit(self,waveforms,temps): #Finds the best fit bassed on CC of waveforms
        IN=self.fast_normcorr(temps,waveforms)[0].argmax()
        waveout=waveforms[IN:IN+len(temps)]
        return waveout
        
    def fast_normcorr(self,t, s): # Fast normalized Xcor
        if len(t)>len(s): #switch t and s if t is larger than s
            t,s=s,t
        n = len(t)
        nt = (t-np.mean(t))/(np.std(t)*n)
        sum_nt = nt.sum()
        a = pd.rolling_mean(s, n)[n-1:-1]
        b = pd.rolling_std(s, n)[n-1:-1]
        b *= np.sqrt((n-1.0) / n)
        c = np.convolve(nt[::-1], s, mode="valid")[:-1]
        result = (c - sum_nt * a) / b    
        return result, b, np.std(t),n
                
    def _groupEvents(self,RES,maxTimeDiffernce,requiredNumStations,StaList): #Some work to speed this up is needed
        """ Uses a simple coincidence trigger algorythm to group Coefficients together that are likely from the same event
        based on the modified time stamp
        """
        if len(StaList)*requiredNumStations<2:
            print 'Station cluster set to one station, no clustering performed'
            #raise Exception('Station cluster set to one station, raise requiredPrecentHit or acsess each hit via CorRes.RES')
        RES.sort(columns='MSTMP',inplace=True)
        RES.reset_index(inplace=True,drop=True)
        groups = RES.groupby((RES.MSTMP.diff() > maxTimeDiffernce).cumsum())
        #now=datetime.datetime.now()
        EVENTS=[]
        for a in groups:
            if len(a[1])>=requiredNumStations:
                EVENTS=EVENTS+[a[1].reset_index(drop=True)]

        return EVENTS     
        
    def _verifyTemplates(self,corResultsInstance): # verify, by time, that events in TemplateKey happen in ArcDB
        templateKey=corResultsInstance.templateKey
        verTimeTol=corResultsInstance.verificationTimeTolerance
        init=1
        for a in templateKey.iterrows():
            df=None
            TSTMP=obspy.core.UTCDateTime(a[1]['TIME']).timestamp
            df=self._loadArcDB(verTimeTol,TSTMP,'Arc.db')
            if len(df)>1: # if more than one row is returned
                df=df[df['Mag'].idxmax():df['Mag'].idxmax()+1]
            if not df.empty:
                if init==1:
                    tableARC=df
                    tableTEM=pd.DataFrame(a[1],columns=[a[0]]).T
                    init=0
                else:
                    tableARC=tableARC.append(df)
                    tableTEM=tableTEM.append(pd.DataFrame(a[1],columns=[a[0]]).T)
        return tableARC.reset_index(drop=True), tableTEM.reset_index(drop=True)
        
        
    def _removeNearDuplicateEvents(self,DF): # Brute force function to remove events that occur within minEventTimeDifference       
        DF.sort(columns='PreOtime',inplace=True)
        g = DF.groupby((DF.PreOtime.diff() > self.maxTimeDifference).cumsum())
        DF=DF.loc[g.MeanCoef.idxmax()]
        return DF
        
        
    def _getTotalSum(self):
        DF=pd.DataFrame()
        for a in range(len(self.Stream)):
            try: #if no events found in some streams make sure it doesnt break
                DF=DF.append(self.Stream[a].events.table,ignore_index=True)
            except AttributeError:
                pass
        if len(DF)==0:
            return DF
        DF.sort(columns=['Time'],inplace=True)
        DF.reset_index(inplace=True,drop=True)
        DF=self._removeNearDuplicateEvents(DF)
        return DF
              
    def _getTrimTimes(self,template,staList,CorResInstance): # get all pick times
        stations=staList
        offsets=[0]*len(stations)
        for sta in range(len(stations)):
            fill=glob.glob(os.path.join(CorResInstance.templatePath,template['NAME'],stations[sta],'*'))[0]
            ftl=glob.glob(os.path.join(fill,'*tms.npy'))
            if len(ftl)>0:
                offsets[sta]=np.load(ftl[0])
        return np.array(offsets)
                    
    def _getPredictedOriginTime(self,DeventInstance,CORDInstance,CorResInstance):

        template=CORDInstance.templateKey        
        #trimTimes=CorResInstance._getTrimTimes(template,DeventInstance.Devents['Sta'].tolist(),CorResInstance)
        pksDF=CorResInstance.pksDF
        pkTrimed=pksDF[(pksDF.Station.isin(DeventInstance.Devents['Sta'].tolist()))&(pksDF.Name==template['NAME'])]
        mintimeDif=np.min(pkTrimed.Starttime.values)-obspy.core.UTCDateTime(template['TIME']).timestamp #fix to find min of time used, a
        PreTime=np.min([DeventInstance.timeDict[x] for x in list(DeventInstance.Devents['Sta'])])-mintimeDif
        return PreTime
            
    def _makeHeader(self,event):
        template=self.templateKey[self.templateKey.NAME==event[1]['Template']]
        oTime=obspy.core.UTCDateTime(event[1]['PreOtime'])
        headDat=[oTime.year,oTime.month,oTime.day,oTime.hour,oTime.minute,oTime.second+oTime.microsecond/1000000.0,template['LAT'],template['LON'],template['DEPTH'],event[1]['Mag'],event[0]+1]
        #headDat=[oTime.year,oTime.month,oTime.day,oTime.hour,oTime.minute,oTime.second+oTime.microsecond/1000000.0,38.94748,-107.556892,template['DEPTH'],event[1]['Mag'],event[0]+1]
        header='# %04d %02d %02d %02d %02d %.4f %.5f %.5f %.2f %.2f 0.0 0.0 0.0 %01d \n' % tuple(headDat)
        return header

              
    def writePhaseDD(self,name='Dtex.pha',onlyVerified=False,DF=None):
        """ Write a phase file used by ph2dt (a program of hypoDD)
        """
        if onlyVerified==True:
            tb=self.verif
        else:
            tb=self.table
        if DF !=None:
            tb=DF
        if not isinstance(tb,pd.DataFrame): # make sure self.table exists and is dataframe
            raise Exception('Table not found, no events were detected and clustered')
        with open(name,'wb') as pha:
            for a in tb.iterrows(): # Loop through CorResults.table
                header=self._makeHeader(a)
                pha.write(header)
                templateKey=self.templateKey[self.templateKey.NAME==a[1]['Template']]
                stakey=pd.read_csv(templateKey['STATIONKEY'].values[0])
                Stations=stakey.NETWORK+'.'+stakey.STATION
                for b in Stations:
                    if b in a[1].keys().tolist():
                        if not np.isnan(a[1][b]):
                            if a[1][b]-a[1]['PreOtime'] <0: #Insure pick is not before origin time reported in templateKey
                                raise Exception('Pick before reported Origin time for ' + a[1]['Template'] + ' ' + b )
                            lineData=[b.split('.')[1],a[1][b]-a[1]['PreOtime'],1,'S']
                            line='%s %.4f %02d %s \n' % tuple(lineData)
                            pha.write(line)
                        #print (b + ' not found in table')
                        
    def writePhaseHyp(self,name='Dtex.pha',onlyVerified=False,DF=None,verifyHyps=False,fix=0,depth=100,phase='S'):
        """ Write a y2k complient phase file used by hypoinverse 2000, format defined on 
        page 113 of the manual for version 1.39
        If onlyVerified is True than only events with verifications writen to file
        IF verifyhps is True the verified Hypocenters are passed as starting location
        if fix==0 nothing is fixed, fix==1 depths are fixed, fix==2 hypocenters fixed, fix==3 hypocenters and origin time fixed
        depth/100 is the starting depth for hypoinverse in km
        """
        tb=self.table
        if verifyHyps==True:
            onlyVerified=True
        if onlyVerified==True:
            tb=tb[~np.isnan(tb.Verfied)]
        if DF !=None:
            if isinstance(DF,pd.DataFrame):
                tb=DF
            else:
                raise Exception('DF is not a Pandas DataFrame')
        with open(name,'wb') as pha:
            pha.write('\n')
            for a in tb.iterrows(): # Loop through CorResults.table
                templateKey=self.templateKey[self.templateKey.NAME==a[1]['Template']]
                stakey=pd.read_csv(templateKey['STATIONKEY'].values[0])
                Stations=stakey.NETWORK+'.'+stakey.STATION
                for b in Stations:
                    if b in a[1].keys().tolist():
                        if not np.isnan(a[1][b]):
                            if a[1][b]-a[1]['PreOtime'] <0: #Insure pick is not before origin time reported in templateKey
                                raise Exception('Pick before reported Origin time for ' + a[1]['Template'] + ' ' + b )
                            #lineData=[b.split('.')[1],a[1][b]-a[1]['PreOtime'],1,'S']
                            line=self._makeSHypStationLine(b.split('.')[1],'ZENZ',b.split('.')[0],a[1][b],phase) #assume s waves
                            pha.write(line)
                        #print (b + ' not found in table')
                el=self._makeHypTermLine(a[1]['PreOtime'],fix,depth,verifyHyps,a)
                pha.write(el)
                pha.write('\n')
                               
    def _makeSHypStationLine(self,sta,cha,net,ts,pha):
        Ptime=obspy.core.UTCDateTime(ts)
        datestring=Ptime.formatIRISWebService().replace('-','').replace('T','').replace(':','').replace('.','')
        YYYYMMDDHHMM=datestring[0:12]
        ssss=datestring[12:16]
        end='01'
        ty=' %s 0' % pha
        line="{:<5}{:<3}{:<5}{:<3}{:<13}{:<80}{:<2}\n".format(sta,net,cha,ty,YYYYMMDDHHMM,ssss,end)
        return line
        
    def _makeHypTermLine(self,Otime,fix,depth,verifyHyps,DFrow):
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
        if verifyHyps:
            lat,latminute=self._toDegreeMinutes(DFrow[1].verlat)
            lon,lonminute=self._toDegreeMinutes(DFrow[1].verlon)
        else:
            lat,latminute,lon,lonminute=' ',' ',' ',' ' #dont give trial lats/lons
        endline="{:<6}{:<8}{:<3}{:<4}{:<4}{:<4}{:<5}{:<1}\n".format(space,hhmmssss,lat,latminute,lon,lonminute,depth,fixchar)
        return endline
                        
                        
    def _toDegreeMinutes(self,x):
        Deg=int(np.trunc(x))
        Minutes=int(np.trunc((x-np.trunc(x))*60*100)) #multiply by 100 to be in f.2 format
        return Deg, Minutes
        
    def calcDetRatio(self):
        detRat=[0]*len(self)
        for a in range(len(self)):
            try:
                detRat[a]=float(np.sum(self[a].events.sum['NumStations']))/len(self[a].RES)
            except KeyError:
                pass
        return detRat
        
    def analizeRES(self,numStations,BuffTime): #Helps analize detections that do not cluster
        RF=pd.DataFrame()
        rf=[]
        for a in self:
            RF=RF.append(a.RES,ignore_index=True)
        RF.sort(columns='MSTMP',inplace=True)
        RF.reset_index(drop=True,inplace=True)
        RF['Dif']=RF.MSTMP.diff()
        st1='rf=RF[(RF.Dif<%.02f)'%BuffTime
        st2=''
        for a in range(numStations):
            st2=st2+'&(RF.Dif.shift(%d)<%.02f)'%(a,BuffTime)
        st3=']'
        exec(st1+st2+st3)
        rf=rf[rf.Dif>-1]
        return rf
    
    def addReloc(self,relocfile='hypoDD.reloc',moveToTemplate=True):
        """add the hypoDD locations to the results class structure"""
        reloAr=np.genfromtxt(relocfile)
        reloDF=pd.DataFrame(reloAr[:,1:4],index=(reloAr[:,0]),columns=(['Lat','Lon','Dep']))
        self.table=pd.merge(self.table,reloDF,how='outer',right_index=True,left_index=True)
        self.verif=self.table[self.table.Verified!=' ']
        if moveToTemplate:
            self._moveToTemplate()
        distError=[None]*len(self.table)
        for a in range(len(self.table)):
            if not any(np.isnan(self.table[['Lat','Lon','verlat','verlon']].values[a].tolist())):
                try:
                    cur=self.table[['Lat','Lon','verlat','verlon']].values[a].tolist()
                    cur=np.multiply(cur,np.sign(cur)) # sloppy fix for now, make sure all signs of lat/lon are the same
                    distError[a]=obspy.core.util.gps2DistAzimuth(cur[0],cur[1],cur[2],cur[3])[0]/1000
                except:
                    distError[a]=np.nan
            else:
                distError[a]=np.nan
        self.table['EpiError']=distError 
    def _moveToTemplate(self):
        temps=self.table[(self.table.MeanCoef>.99) & (abs(self.table.Lat)>0)]
        if len(temps)==0:
            print ('No template (self detection) found that was also relocated by hypoDD, skipping event shift')
        else:
            latshift=temps['verlat'].tolist()[0]-temps['Lat'].tolist()[0]
            lonshift=temps['verlon'].tolist()[0]-temps['Lon'].tolist()[0]
            self.table['Lat']=self.table['Lat']+latshift
            self.table['Lon']=self.table['Lon']+lonshift            
            
    def _removeTemplatetoTemplateCorrs(self):
        templateTimeStamps=np.array([obspy.core.UTCDateTime(x).timestamp for x in self.templateKey['TIME']])
        DFsum=self.table.copy()
        DFsum['Move']=False #initialize blank column with boolean to move to auto or not
        for a in DFsum.iterrows():
            UTC=obspy.core.UTCDateTime(a[1].Time).timestamp
            DFsum.Move[a[0]]=any(abs(templateTimeStamps-UTC)<self.autoProximity)
        newsum=DFsum[DFsum.Move==False].drop('Move',axis=1)
        auto=DFsum[DFsum.Move==True].drop('Move',axis=1)
        return newsum.reset_index(drop=True),auto.reset_index(drop=True)
            

def corrResults(trigCon=1,trigParameter=0,CorDB='Corrs.db',templatePath='EventWaveForms',condir='ContinousWaveForms',
                 templateKey='TemplateKey.csv',pks='EventPicks.pkl',maxTimeDifference=1,requiredNumStations=4,ArcDB='Arc.db',
                 verificationTimeTolerance=2,minEventTimeDifference=2,autoProximity=5):
    """
    Wrapper function for the CorResults class. Used to associate detections across multiple stations into 
    coherent events. CorResults class also has some useful methods for creating input files for location
    programs such as hypoDD and hypoInverse
    
    Parameters
    ---------
    trigCon : int 0 or 1
        trigCon is used for filtering the events in the corDB. Currently only options are 0, raw coeficient value
        or 1, STA/LTA value as reported in the corDB. This parameter might be useful if there are a large number of detections
        in corDB with low STA/LTA or correlation coeficient values. Currently other statistical methods are not avaliable
    trigParameter : number
        if trigCon==0 trigParameter is the minimum correlation for a detection to be loaded from the CorDB
        if trigCon==1 trigParameter is the minimum STA/LTA value for a detection to be loaded from the CorDB
        Regardless of trigCon if trigParameter == 0 all detections in CorDB will be loaded and processed
    CorDB : str
        Path the the database created by detex.xcorr.correlate function
    templatePath : str
        Path to the tempalte directory
    condir : str
        Path to the continous data directory
    templateKey : str
        Path to the template key
    pks : str
        Path to the event picks file created by detex.util.trimTemplates
    maxTimeDifference : number
        Max time variation in seconds from predicted origin time (of each detection) for detections to be grouped together
        as an event
    requiredNumStations : int
        The required number of a stations for a detection to occur on in order to be classified as an event
    ArcDB : str (optional)
        Path to a SQLit database that contains an event catalog agaisnt which you wish to verify the events found 
        using cross correlation. If the file does not exist this will be skipped
    verificationTimeTolerance : number (optional)
        Number of seconds between an event origin time listed in ArcDB and a predicted orgin calculated using template 
        pick times and origin time in event key for a detected event to be considered verified.
    minEventTimeDifference : number
        pass
    autoProximity : number
        Number of seconds a detection can occur away from a template event and not be classified as the template event
        (or associated side lobs)

    """

    
    corRes=CorResults(trigCon,trigParameter,CorDB,templatePath,condir,templateKey,pks,maxTimeDifference,requiredNumStations,ArcDB,
                 verificationTimeTolerance,minEventTimeDifference,autoProximity)
    return corRes
                              
class CorResults(allTheFunctions): # Object to sperate Events, creates a list-class of the results of each event
    def __init__(self,trigCon,trigParameter,CorDB,templatePath,condir,templateKey,pks,maxTimeDifference,requiredNumStations,ArcDB,
                 verificationTimeTolerance,minEventTimeDifference,autoProximity):
        """Class to read and process Corrs directory.
        
        ceofmin- minimum correlation coeficient to read in
        maxTimeDiffernce- max time variation (in seconds) from adjustd time stamps
        in order for events to be grouped together as the same event
        RequiredNumStations is the number of stations needed to be triggered to count as an event
        trigger con- 0 is abosulte coef 1 is STA/LTA (trigParameter must correspond to TrigerCon type)
        autoProximity is how close a detection can be to a template in time and be counted as the same event
        """
        self.__dict__.update(locals()) # Instantiate all input variables
        self.pksDF=pd.read_pickle(pks)
        self._checkExists([CorDB,templatePath,condir,templateKey])
        Cwd=os.getcwd()
        self.templateKey=pd.read_csv(templateKey,skipinitialspace=True)
        self.conPath=glob.glob(os.path.join(Cwd,condir))[0]
        self.templatePath=os.path.join(Cwd,templatePath)
        self.Stream=len(self.templateKey.NAME)*[0]
        self.itstart=0
        self.suc,self.fail=0,0
        if os.path.isfile(ArcDB):
            self.verifiedTemplates=self._verifyTemplates(self)
            
        try: # Get the filter params used to create Corrs database, if not exist return none
            self.filt=detex.util.loadSQLite(self.CorDB,'filt_params')[['FREQMIN','FREQMAX','CORNERS','ZEROPHASE','DECIMATE']].values.flatten()
        except:
            self.filt=None
        RES=[0]*len(self.templateKey)
        for a in self.templateKey.iterrows(): # Loop through each template
            RES[a[0]]=self._loadCorDB(CorDB,a[1]['NAME']) # Load database that belong to tempate a
            staKey=pd.read_csv(a[1]['STATIONKEY']) # load current station key
            if len(RES[a[0]])>0:
                self.Stream[a[0]]=CORD(a,self,RES[a[0]],staKey) #Initialize CORD instances
            else:
                print a[1]['NAME'] + ' not found in ' +CorDB
        self.Stream=filter(lambda a1:a1 != 0, self.Stream) #remove any 0 values
        self.RES=pd.concat(RES,ignore_index=True)
        
        isempty=[hasattr(x.events,'table') for x in self.Stream] #test if any events were returned
        if any(isempty):
            self._cleanStream()
            self.table=self._getTotalSum()
            self.table,self.autotable=self._removeTemplatetoTemplateCorrs()
            self.table=self.table[self.table.NumStations>=self.requiredNumStations]
            self.sum=self.table[['Mag','MeanCoef','MeanSTA_LTA','NumStations','Template','Time','Verified']]
            self.auto=self.autotable[['Mag','MeanCoef','MeanSTA_LTA','NumStations','Template','Time','Verified']]
            self.verif=self.table[abs(self.table.Verified)>0] # Make table of only verified events
            if len(self.sum>0):
                self.verRatio=float(len(self.verif))/len(self.sum)
            else:
                self.verRatio=np.nan
            self.detRatio=self.calcDetRatio() # get the ratio of detections clustered into events/total detections
        else:
            print ('No events found, try lowering trigParameter or requiredNumStations')
#        except: #Fill in allowed exception types here, dont leave blanket except statement
#            print('No Events found, try lowering trigParameter or requiredNumStations')
         #removes autocorrelations from self.sum and templates correlating with each other and puts them in self.auto
        try:
            print('Complete, %d events found with verification percent = %.02f and detection use percent = %.02f' % (len(self.sum),self.verRatio*100,np.average(self.detRatio)*100))
        except:
            pass
    def __getitem__(self,index): # allow indexing
        return self.Stream[index]
        
    def __iter__(self): # make class iterable
        return iter(self.Stream)
    
    def __len__(self): 
        return len(self.Stream)
    def __repr__(self):
        print(self.sum)
        return ""
        
    
        
               
class CORD:
    """ Class to visaulize and organize Corr
    ceofmin- minimum correlation coeficient to read in
    maxTimeDiffernce- max time variation (in seconds) from adjustd time stamps
    in order for events to be grouped together as the same event
    self.templateKey[a:a+1],self.templatePath,maxTimeDiffernce,requiredNumStations,self.conPath,self.stationKey
    """

    def __init__(self,templateKeyIteration,CorResInstance,RES,staKey):  
        
        self.maxTimeDiffernce=CorResInstance.maxTimeDifference
        self.templateKey=templateKeyIteration[1]
        self.template=templateKeyIteration[1]['NAME']
        self.templatePath=CorResInstance.templatePath
        self.trigParameter=CorResInstance.trigParameter
        self.stationKey=staKey
        self.stations=staKey['NETWORK']+'.'+staKey['STATION']        
        self.RES= RES.drop_duplicates()
        self.events=EventStream(self,CorResInstance)
    
    
class EventStream:
    def __init__(self,CORDInstance,CorResInstance): 
        DEvents=CorResInstance._groupEvents(CORDInstance.RES,CorResInstance.maxTimeDifference,CorResInstance.requiredNumStations,CORDInstance.stationKey)
        #deb(DEvents)        
        self.EvStream=[0]*len(DEvents)
        for a in range(len(self.EvStream)):
            self.EvStream[a]=Devent(DEvents[a],CORDInstance,CorResInstance,self,a)
            if a==0:
                self.table=self.EvStream[a].table
                self.ver=self.EvStream[a].ver
            else:
                self.table=self.table.append(self.EvStream[a].table)
                self.ver=self.ver.append(self.EvStream[a].ver)
        if len(self.EvStream) != 0:
            self.sum=self.table[['Mag','MeanCoef','MeanSTA_LTA','NumStations','Template','Time','Verified']]
        else:
            self.sum=pd.DataFrame()
            print 'no events found for template %s that meet station requirement' % (CORDInstance.template)
        
    def __getitem__(self,index): # allow indexing
        return self.EvStream[index]
        
    def __iter__(self): # make class iterable
        return iter(self.EvStream)
    
    def __len__(self): 
        return len(self.EvStream)            
            
class Devent:         
    def __init__(self,Devents,CORDInstance,CorResInstance,eventStremInstance,itdex):
        #self.summary=self._sumEvent(Devents)
        Devents=CorResInstance._testDuplicateStations(Devents,itdex,CORDInstance.template)
        self.Devents=Devents
        meanCoef=np.mean(Devents['Coef'])
        meanSTA_LTA=np.mean(Devents['STA_LTACoef'])
        self.staDict=CorResInstance._createStationDict(CORDInstance.stationKey)
        alpha=np.mean(Devents['alpha']) # would a weighted average be better?
        mag=CORDInstance.templateKey['MAG']+np.log10(alpha)
        dateStr=obspy.core.UTCDateTime(np.min(Devents['STMP'])).formatIRISWebService()
        self.timeDict=CorResInstance._timeDict(Devents,CORDInstance.stations.values.tolist()) # init a dictionary of lag times with stations as keys
        #Column order should be maintained this way, but if STMPs get mixed consider using
        predictedTime=Devents.MSTMP.mean()
        dateStr=obspy.core.UTCDateTime(predictedTime).formatIRISWebService()
        sumvect=[mag,len(Devents),meanCoef,meanSTA_LTA,dateStr,CORDInstance.template,np.nan,predictedTime,np.nan,np.nan,np.nan]+[self.timeDict[x] for x in list(CORDInstance.stations.values.tolist())] 
        columns=['Mag','NumStations','MeanCoef','MeanSTA_LTA','Time','Template','Verified','PreOtime','verlat','verlon','verdep']+ CORDInstance.stations.values.tolist()
        self.table=pd.DataFrame([sumvect],columns=columns,index=[itdex])
        self.ver=CorResInstance._verifyDEvents(CORDInstance,CorResInstance,eventStremInstance,self,itdex)
        if isinstance(self.ver, pd.DataFrame):
            self.table['Verified'][itdex]=self.ver['Mag'].values[0]
            self.table['verlat'][itdex]=self.ver['Lat'].values[0]
            self.table['verlon'][itdex]=-self.ver['Lon'].values[0] # !!!!!!!!! make negative for NF input. remove when done !!!!!!!!!!!!!!!!!!!!
            self.table['verdep'][itdex]=self.ver['Depth'].values[0]
        elif self.ver == None:
            self.ver=pd.DataFrame(index=[itdex])
        self.sum=self.table[['Mag','MeanCoef','MeanSTA_LTA','NumStations','Template','Time','Verified']]

def deb(varlist):
    global de
    de=varlist
    sys.exit(1)                   


def ssResults(trigCon=1,trigParameter=0,associateReq=0,associateBuffer=1,requiredNumStations=4,veriBuffer=1,ssDB='SubSpace.db',
              templatePath='EventWaveForms',condir='ContinousWaveForms',templateKey='TemplateKey.csv',stationKey='StationKey.csv',
              veriFile=None,includeAllVeriColumns=True,reduceDets=True,sspickle='subspace.pkl',Pf=False,Stations=None,
              starttime=None,endtime=None):
    """
    Wrapper function for the CorResults class. Used to associate detections across multiple stations into 
    coherent events. CorResults class also has some useful methods for creating input files for location
    programs such as hypoDD and hypoInverse
    
    Parameters
    ---------
    trigCon : int 0 or 1
        trigCon is used for filtering the events in the corDB. Currently only options are 0, raw detection statistic value
        or 1, STA/LTA value of SD as reported in the corDB. This parameter might be useful if there are a large number of detections
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
    associateBuffer : real number (int or float)
        The buffertime applied to event assocaition in seconds  
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
    sspickle : str (optional)
        Path to a pickled subspace stream object, if not found Pf parameter cannot be used
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
    _checkExistence([ssDB,templateKey]) #make sure all required things exist
    _checkInputs(trigCon,trigParameter,associateReq,associateBuffer,requiredNumStations)
    if associateReq != 0:
        raise Exception('associateReq values other than 0 not yet supported') #TODO: implement this in a way that isnt cripplingly slow
    
    ### Try to read in all input files and dataframes needed in the 
    temkey=pd.read_csv(templateKey) #load template key
    stakey=pd.read_csv(stationKey)#load station key
    try:
        ss_info=detex.util.loadSQLite(ssDB,'ss_info') # load subspace info
    except:
        print('Loading ss_info failed')
        ss_info=None
    try: 
        ss_filt=detex.util.loadSQLite(ssDB,'filt_params')
    except:
        print ('loading filt_params failed')
        ss_filt=None
        
    ss,PfKey=_makePfKey(sspickle,Pf)

    ### Parse each station results and delete detections that occur on multiple subpspace, keeping only the subspace with highest detection stat  
    if reduceDets:
        ssdf=_deleteDetDuplicates(ssDB,trigCon,trigParameter,associateBuffer,starttime,endtime,Stations,PfKey=PfKey)
    else:
        if Pf:
            raise Exception('When using the Pf parameter reduceDets must be True')
        ssdf=detex.util.loadSQLite(ssDB,'ss_df')
    
    if isinstance(Stations,(list,tuple)): # reduce stations if requested
        ssdf=ssdf[ssdf.Sta.isin(Stations)]    
    ### Associate detections on different stations together
    Dets,Autos=_associateDetections(ssdf,associateReq,requiredNumStations,associateBuffer,ss_info,temkey)
    
    ### Make a dataframe of verified detections if applicable
    Vers=_verifyEvents(Dets,Autos,veriFile,veriBuffer,includeAllVeriColumns)

    ssres=SSResults(Dets,Autos,Vers,ss_info,ss_filt,temkey,stakey)
    return ssres

def _makePfKey(sspickle,Pf):
    """
    Make simple df for defining SD values corresponing to Pf for each subspace station pair
    """
    if not Pf: #if no Pf value passed simply return none
        return None
    else:
        ss=detex.subspace.loadSubSpace(sspickle)
        df=pd.DataFrame(columns=['Sta','Name','SD','betadist'])
        for station in ss.subspaces.keys():
            ssSta=ss.subspaces[station]
            for subspa in ssSta.Name:
                beta=ssSta[ssSta.Name==subspa].iloc[0].FAS['betadist']
                sd=scipy.stats.beta.isf(Pf,beta[0],beta[1])
                if sd>.90: #If isf fails
                    sd=_approximateThreshold(beta[0],beta[1],Pf,1000,2)
                if not isinstance(sd,numbers.Number):
                    sd=sd[0]
                d=pd.DataFrame([[station,subspa,sd,beta]],columns=['Sta','Name','SD','betadist'])
                df=df.append(d,ignore_index=True)
        df.reset_index(drop=True,inplace=True)
        return ss,df
                
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
        print 'No veriFile passed or it does not exist, skipping verification'
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
                trudet=temDets[temDets.SDav==temDets.SDav.max()]
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
                    trudet=temAutos[temAutos.SDav==temAutos.SDav.max()]
                    Autos.loc[trudet.index[0],'Verified']=True

                    if includeAllVeriColumns:
                        for col in additionalColumns:
                            if not col in trudet.columns:
                                trudet[col]=verrow[col]
                    trudet['VerMag'],trudet['VerLat'],trudet['VerLon'],trudet['VerDepth'],trudet['VerName']=verrow.MAG,verrow.LAT,verrow.LON,verrow.DEPTH,verrow.NAME
                    verlist.append(trudet)
        if len(verlist)>0:
            verifs=pd.concat(verlist,ignore_index=True)
            verifs.sort(columns=['Event','SDav']) #sort and drop duplicates so each verify event is verified only once
            verifs.drop_duplicates(subset='Event')
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
        

def _buildSQL(PfKey,trigCon,trigParameter,Stations,starttime,endtime):
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
            SQL.append('SELECT %s FROM %s WHERE Sta="%s" AND Name="%s" AND  SD>%f AND MSTAMPmin>%f AND MSTAMPmin<%f'% ('*','ss_df',row.Sta,row.Name,row.SD,starttime,endtime))
    
    else:         
        if trigCon==0:
            cond='SD'
        elif trigCon==1:
            cond='SD_STALTA'
        for sta in Stations:
            if sta=='*':
                SQL.append('SELECT %s FROM %s WHERE %s > %s AND MSTAMPmin>%f AND MSTAMPmin<%f' % ('*', 'ss_df',cond,trigParameter,starttime,endtime))
            else:
                SQL.append('SELECT %s FROM %s WHERE %s="%s" AND %s > %s AND MSTAMPmin>%f AND MSTAMPmin<%f' % ('*', 'ss_df','Sta',sta,cond,trigParameter,starttime,endtime))
    return SQL
    
def _deleteDetDuplicates(ssDB,trigCon,trigParameter,associateBuffer,starttime,endtime,Stations,PfKey=None):
    """
    delete dections of same event, keep only detection with highest detection statistic
    """
    sslist=[]
    
    SQLstr=_buildSQL(PfKey,trigCon,trigParameter,starttime,Stations,endtime)    
    for sql in SQLstr:
        sslist.append(detex.util.loadSQLite(ssDB,'ss_df',sql=sql))
        
    ssdf=pd.concat(sslist,ignore_index=True)
    ssdf.reset_index(drop=True,inplace=True)

    ssdf.sort(columns=['Sta','MSTAMPmin'],inplace=True)
    ssdf['Gnum']=((ssdf.MSTAMPmin-associateBuffer>ssdf.MSTAMPmax.shift())&(ssdf.Sta==ssdf.Sta.shift())).cumsum()
    ssdf.sort(columns=['Gnum','SD'],inplace=True)
    ssdf.drop_duplicates(subset='Gnum',take_last=True)
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
    autolist=[pd.DataFrame(columns=['Event','SDav','SDmax','SD_STALTA','MSTAMPmin','MSTAMPmax','Mag','ProEnMag','Verified','Dets'])]
    detlist=[pd.DataFrame(columns=['Event','SDav','SDmax','SD_STALTA','MSTAMPmin','MSTAMPmax','Mag','ProEnMag','Verified','Dets'])]
    temkey['STMP']=np.array([obspy.core.UTCDateTime(x) for x in temkey.TIME])
    temcop=temkey.copy()
    if isinstance(ss_info,pd.DataFrame) and associateReq>0:
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
                if len(set(g.Sta))+2<len(g.Sta):
                    g=g.sort(columns='SD').drop_duplicates(subset='Sta',take_last=True).sort(columns='MSTAMPmin')
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
    pass
    
def _createDetTable(g):
    
    event=obspy.core.UTCDateTime(np.mean([g.MSTAMPmin.mean(),g.MSTAMPmax.mean()])).formatIRISWebService().replace(':','-').split('.')[0]
    detDF=pd.DataFrame([[event,g.SD.mean(),g.SD.max(),g.SD_STALTA.mean(),g.MSTAMPmin.min(),g.MSTAMPmax.max(),np.median(g.Mag),np.median(g.ProEnMag),False,g]],columns=['Event','SDav','SDmax','SD_STALTA','MSTAMPmin','MSTAMPmax','Mag','ProEnMag','Verified','Dets'])
    return detDF
         
def _createAutoTable(g,temkey):
    isauto=False
    for num,row in g.iterrows(): #find out if this is an auto detection
        temtemkey=temkey[(temkey.STMP>row.MSTAMPmin)&(temkey.STMP<row.MSTAMPmax)]
        if len(temtemkey)>0:
            isauto=True
            event=temtemkey.iloc[0].NAME
    if isauto:
        autoDF=pd.DataFrame([[event,g.SD.mean(),g.SD.max(),g.SD_STALTA.mean(),g.MSTAMPmin.min(),g.MSTAMPmax.max(),g.Mag.median(),np.median(g.ProEnMag),False,g]],columns=['Event','SDav','SDmax','SD_STALTA','MSTAMPmin','MSTAMPmax','Mag','ProEnMag','Verified','Dets'])
        return isauto,autoDF
    else:
        return isauto,pd.DataFrame()
   
         
def _loadSSDB(ssDB,trigCon,trigParameter,sta=None): #load the subspace database
    if trigCon==0:
        cond='SD'
    elif trigCon==1:
        cond='SD_STALTA'
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
        if not os.path.exists(fil):
            raise Exception('%s does not exists'%fil)
            
class SSResults(object):
    def __init__(self,Dets,Autos,Vers,ss_info,ss_filt,stakey,temkey):
        self.Autos=Autos
        self.Dets=Dets
        self.NumVerified=len(Vers) if isinstance(Vers,pd.DataFrame) else 'N/A'
        self.Vers=Vers
        self.info=ss_info
        self.filt=ss_filt
        self.StationKey=stakey
        self.TemplateKey=temkey
    def __repr__(self):
        outstr='SSResults class with %d autodections and %d new detections, %s are verified'%(len(self.Autos),len(self.Dets),str(self.NumVerified))
        return outstr
                       
            
            
            
            
            
            
            
            
            
            
            
            
