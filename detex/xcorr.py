# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:30:10 2014

@author: Derrick Chambers


"""
import numpy as np, pandas as pd, obspy, glob, os, sys, detex,numbers,scipy,pickle
import multiprocessing, itertools,warnings
#warnings.filterwarnings('error')

def getFAS(numConDat,calcReverse=False,tempKeyPath='TemplateKey.csv',EveDir='EventWaveForms',ConDir='ContinousWaveForms',filt=[1,10,2,True],decimate=False,
           templateDuration=30,saveit=True,filename='FAS.pkl',pks='EventPicks.pkl',secB4FirstArrival=0,LTATime=5,STATime=0.5,staltalimit=6.5):
    """
    getFAS (get false alarm statistic) is a wrapper function to initialize the FAS class which attemptes to empirically calculate the null distribution
    of the template and the continous data. A number of continous data chunks are selected and if a simple STA/LTA power detector doesn't detect any high 
    amplitude signals the chunk of data is used (along with others that meed the requirement) to estimate a pdf. Once a model PDF exists the required correlation 
    coefficient can be set based on some theoritical false alarm probability. 
    
    Parameters
    ------------------------
    numConDat : int
        The number of chunks of continous data (by default each chunk is one hour) to select at random to correlate
        If numConDat> 0.25* number of avalaible data it will be reduced to 1/4 of the avalaible data
    calcReverse : boolean
        If true the time reversed template is also calculated as done in slinkard 2014 to estimate null distribution
    tempKeyPath : str
        The path where the template key is found
    EveDir : str
        The path were the event directory (in the format detex.getdata.getAllData creates) resides
    ConDir : str
        The continous data directory path
    filt : list of length 4 [int,int,int,boolean]
        A string containing the obspy bandpass filter parameters (frequency 1 in hz, frequency 2 in hz, number of corners, and zerophase)
    decimate : None or False
        A deecimation factor to decimate both the templates and the continous data before running correlations. Can greatly speed up the
        correlations but is not extensively tested. False means no decimation is perfomed. See the obspy trace decimation method for details
    templateDuration : int or float
        The duration in seconds from the start of the pick time to trim the event
    saveit : boolean
        If true save FAS for use in waveform correlation. If false FAS will have to be manually passed (IE through the terminal)
    filename : str
        The name to which the fas class will be pickled
    pks : str
        The pickled pick times for the template events
    secB4FirstArrival : int or float
        The number of seconds before the first arrival in the pks file to define as the start of the template
    LTATime : int, float
        The time in seconds to use for the LTA in the STA/LTA filtered applied to continous data to try and avoid using any data with events
    STATime : int, float
        The time in seconds to use for the STA in the STA/LTA filtered applied to continous data to try and avoid using any data with events
    staltalimit : int, float
        The limit at which the data gets descarded and are not used in the null distribtution estimation. The current limit works well
        for the TA data I have used with the default settings
        
        
    Returns
    ---------
    A dataframe with histograms, fitted beta and fitted normal parameters for each template-station pair.
    
    """
    
    
    fas=FAS(numConDat,calcReverse,tempKeyPath,EveDir,ConDir,filt,decimate,templateDuration,saveit,filename,pks,secB4FirstArrival,LTATime,STATime,staltalimit)
    
    return fas.result
    
class FAS(object):
    """ False alarm statistic class, currently only supports normal distribution and trigCon=0 (raw coeficient)
    """
    #TODO remove all methods that are not to be called after calculating the FAS, no reason to pack them all into FAS class
    def __init__(self,numConDat,calcReverse,tempKeyPath,EveDir,ConDir,filt,decimate,templateDuration,saveit,filename,pks,secB4FirstArrival,LTATime,STATime,staltalimit):
        """ Function to randomly scan through continous data and fit statistical distributions to histograms in order to get a suggested value
        for various trigger conditions"""
        self.__dict__.update(locals()) # Instantiate all input variables
        self.tempKey=pd.read_csv(tempKeyPath)
        self.pksDF=pd.read_pickle(pks)
        ContinousDataLength=detex.util.getContinousDataLength(Condir=ConDir)
        #results={}
        histBins=np.linspace(-1,1,num=400)
        result=pd.DataFrame()
        
        for a in self.tempKey.iterrows(): #Loop through each tempalte
            staKey=pd.read_csv(a[1].STATIONKEY)
            df=pd.DataFrame(index=[a[1].NAME],columns=[x.NETWORK+'.'+x.STATION for num,x in staKey.iterrows()])
            for b in staKey.iterrows(): #loop through each station
                chans=b[1].CHANNELS.split('-')
                
                toread=glob.glob(os.path.join(self.EveDir,a[1].NAME,b[1].NETWORK+'.'+b[1].STATION+'*'))
                if len(toread)<1:
                    continue
                template=self._applyFilter(obspy.core.read(toread[0]),filt)
                #template,trimTime=self._loadTemplate(a[1],b[1]) #load template waveform for given station/event
                
                Nc=len(list(set([x.stats.channel for x in template])))
                jdays=self._mapjdays(a[1],b[1])
                if numConDat>len(jdays)*(3600/ContinousDataLength)*4: #make sure there are at least conDatNum samples avaliable
                    print 'Not enough continous data for conDatNum=%d, decreasing to %d' %(numConDat,len(jdays)*(3600/ContinousDataLength)*4)
                    numConDat=int(len(jdays)*(3600/ContinousDataLength)*4)
                CCmat=[0]*numConDat
                MPtem=self._multiplex(template,Nc,Template=True) #multplexed template 
                if calcReverse:
                    MPtem=MPtem[::-1] #Reverse order of MPtem if calculating reverse    
                reqlen=int(len(MPtem)+Nc*ContinousDataLength*template[0].stats.sampling_rate-1)
                MPtemFD,sum_nt=self._FFTemplate(MPtem,reqlen) #multplexed template in frequency domain
                usedJdayHours=[]
                for c in range(numConDat):
                    condat,usedJdayHours=self._loadRandomContinousData(jdays,filt,usedJdayHours,LTATime,STATime,staltalimit)
                    #condat=self._loadContinousData(a[1],b[1],jdays)
                    MPcon=self._multiplex(condat,Nc)
                    CCmat[c]=self._MPXCorr(MPcon,MPtemFD,reqlen,sum_nt,MPtem,Nc)
                CCs=np.fromiter(itertools.chain.from_iterable(CCmat), dtype=np.float64)
                #deb(CCs)
                #deb([CCs,template]) #start here, figure out distribution
                outdict={}
                outdict['bins']=histBins
                outdict['hist']=np.histogram(CCs,bins=histBins)[0]
                #results[a[0]]['normdist']=scipy.stats.norm.fit(CCs)
                betaparams=scipy.stats.beta.fit((CCs+1)/2,floc=0,fscale=1) # transformed beta
                outdict['betadist']=betaparams # enforce hard limits on detection statistic
                normparams=scipy.stats.norm.fit(CCs)
                outdict['normparams']=normparams
                df[b[1].NETWORK+'.'+b[1].STATION][a[1].NAME]=outdict
            result=result.append(df)
        self.result=result
        if saveit:
            result.to_pickle(filename)
            #pickle.dump(self,open(filename,'wb'))

    def _loadRandomContinousData(self,jdays,filt,usedJdayHours,LTATime,STATime,staltalimit): #loads random chunks of data from total availible data
        failcount=0
        while failcount<50:
            rand1=np.round(np.random.rand()*(len(jdays)-1))
            try:
                WFs=glob.glob(os.path.join(jdays[int(rand1)],'*'))
                rand2=np.round(np.random.rand()*len(WFs))
                if [rand1,rand2] in usedJdayHours: # if data has already been used
                    failcount+=1
                    continue
                ST=obspy.core.read(WFs[int(rand2)])
                TR=self._applyFilter(ST,filt)
                TRz=TR.select(component='Z').copy()
                cft = obspy.signal.trigger.classicSTALTA(TRz[0].data, STATime*TRz[0].stats.sampling_rate , LTATime*TRz[0].stats.sampling_rate) #make sure no high amplitude signal is here
                if cft.max()>staltalimit:
                    failcount+=1
                #print 'rejecting continous data'
                else:
                    usedJdayHours.append([rand1,rand2])
                    break
            except: #allow a certain number of failures to account for various possible data problems
                failcount+=1
            if failcount>49: 
                deb([WFs,rand2,ST,cft])
                raise Exception('something is broked')
        return TR,usedJdayHours

    
    def _multiplex(self,TR,Nc,Template=False, trimTolerance=15,returnlist=False):
        startime=[x.stats.starttime for x in TR] # find starttime for each trace
        TR.trim(starttime=max(startime)) # trim to min startime
        if Nc==1 and Template:
            return TR[0].data[::-1]
        elif Nc==1 and not Template:
            return TR[0].data
        if Template: #if it is a tempalte put first column last
            chans=[x.data for x in TR]
            #chans.append(TR[0].data)
            #chans=chans[1:]
        else:
            chans=[x.data for x in TR]     
             #make sure all arrays are same length
        minlen=np.array([len(x) for x in chans])  
        if max(minlen)-min(minlen) > 15:
            if Template:
                raise Exception('timTolerance exceeded, examin Template: \n%s' % TR[0])
            else:
                print('timTolerance exceeded, examin data, Trimming anyway: \n%s' % TR[0])
                trimDim=min(minlen)
                chansTrimed=[x[:trimDim] for x in chans]
                #deb(TR)
                #return False,False
        elif max(minlen)-min(minlen)>0 : #trim a few samples off the end if necesary
            trimDim=min(minlen)
            chansTrimed=[x[:trimDim] for x in chans]
        elif max(minlen)-min(minlen)==0:
            chansTrimed=chans
        C=np.vstack((chansTrimed))
        C1=np.ndarray.flatten(C,order='F')
        if Template: #If it is the template reverse the order so convolution works
            C1=C1[::-1]
        if returnlist:
            return C1,C
        return C1
                
    def _MPXCorr(self,MPcon,MPtemFD,reqlen,sum_nt,MPtem,Nc):
        MPconFD=scipy.fftpack.fft(MPcon,n=2**reqlen.bit_length())
        n = len(MPtem)
        a = pd.rolling_mean(MPcon, n)[n-1:]
        b = pd.rolling_std(MPcon, n)[n-1:]
        b *= np.sqrt((n-1.0) / n)
        c=scipy.real(scipy.fftpack.ifft(np.multiply(MPtemFD,MPconFD))[len(MPtem)-1:len(MPcon)])
        result = (c - sum_nt * a) / b
        return result[::Nc]
        
    def _FFTemplate(self,MPtem,reqlen):
        n = len(MPtem)
        nt = (MPtem-np.mean(MPtem))/(np.std(MPtem)*n)
        sum_nt=nt.sum()
        MPtemFD=scipy.fftpack.fft(nt[::-1],n=2**reqlen.bit_length())
        return MPtemFD,sum_nt
                                     
    def _applyFilter(self,Trace,filt,trimtime=None):# Apply a filter/decimateion to an obspy trace object and trim 
        Trace.detrend('linear')
        startTrim=max([x.stats.starttime for x in Trace])
        endTrim=min([x.stats.endtime for x in Trace])
        Trace=Trace.slice(starttime=startTrim,endtime=endTrim) 
        Trace.sort()

        if self.filt!=None:
            if self.decimate: # Decimate to the point so that sr is about 2.5 time high frequency
                Trace.decimate(self.decimate)
            #Trace[0].data=self._fftprep(Trace[0].data)
            Trace.filter('bandpass',freqmin=self.filt[0],freqmax=self.filt[1],corners=self.filt[2],zerophase=self.filt[3])
        if trimtime!=None:
            deb(trimtime)
            if self.trimSeconds==None:
                Trace=Trace.slice(starttime=obspy.core.UTCDateTime(trimtime[0]),endtime=obspy.core.UTCDateTime(trimtime[1]))
            else:
                Trace=Trace.slice(starttime=obspy.core.UTCDateTime(trimtime[0]),endtime=obspy.core.UTCDateTime(trimtime[0])+self.trimSeconds)
        Trace.sort()
        Trace.detrend()
        return Trace               
                
    def _mapjdays(self,template,station):
        years=glob.glob(os.path.join(self.ConDir,station.NETWORK+'.'+station.STATION,'*'))
        jdays=[0]*len(years)
        for a in range(len(jdays)):
            jdays[a]=glob.glob(os.path.join(years[a],'*'))
        jdays=np.concatenate(np.array(jdays))
        return jdays
        
    def _loadTemplate(self,template,station): #load, trim and filter tempalte
        toread=glob.glob(os.path.join(self.EveDir,template.NAME,station.NETWORK+'.'+station.STATION+'*'))
        TR=obspy.core.read(toread[0])
        TR.sort()
        TR.detrend('linear')
        trimDF=self.pksDF[self.pksDF.Path==toread[0]].iloc[0]
        if trimDF.Starttime == trimDF.Endtime and self.templateDuration == None:
            raise Exception ('starttime is the same as endtime for %s, delete the pks and pick again making sure to make at least two\
            picks defining the starttime and endtime, or udefine the templateDuration variable to use a set duration for all templates' )
        trimtime=[trimDF.Starttime-self.secB4FirstArrival,trimDF.Endtime]
        if self.filt != None:
            TR.filter('bandpass',freqmin=self.filt[0],freqmax=self.filt[1],corners=self.filt[2],zerophase=self.filt[3])
        if self.templateDuration == None:
            TR=TR.slice(starttime=obspy.core.UTCDateTime(trimtime[0]),endtime=obspy.core.UTCDateTime(trimtime[1]))
        else:
            TR=TR.slice(starttime=obspy.core.UTCDateTime(trimtime[0]),endtime=obspy.core.UTCDateTime(trimtime[0])+self.templateDuration)
            TR.detrend('linear')
        return TR,trimtime
        
    def _replaceNanWithMean(self,arg): # Replace where Nans occur with closet non-Nan value
        ind = np.where(~np.isnan(arg))[0]
        first, last = ind[0], ind[-1]
        arg[:first] = arg[first+1]
        arg[last + 1:] = arg[last]
        return arg
        
    def __getitem__(self,index): # allow indexing
        return self.normdists[index]
    def __iter__(self): # make class iterable
        return iter(self.normdists)
    def __len__(self): 
        return len(self.normdists) 
        
        
def correlate(trigCon=1,trigParameter=5.5,tempKeyPath='TemplateKey.csv',ConDir='ContinousWaveForms',EveDir='EventWaveForms',pks='EventPicks.pkl',
                CorDB='Corrs.db',FASfile='FAS.pkl',UTCstart=None,UTCend=None,triggerLTATime=5,triggerSTATime=0,decimate=False,filt=[1,10,2,True],
                delOldCorr=True,extrapolateTimes=True,templateDuration=30,multiprocess=False,secB4FirstArrival=0, calcHist=True,histFile='XcorHist.pkl',
                averageAllTemplates=False):
    """
    Function to run the correlations over all events found in event key
    
    Params
    -------------
    trigCon : int, trigParameter : Value to set based on trigcon
        Used to select the trigger condition for declaring detections and parametes that go with each type of detection.  Options are:
            trigCon==0
                Use a user defined correlation coefficient for all station/template pairs
                trigParameter is the required correlation coeficient (between -1 and 1). A value of 0.5 usually works well depending on 
                the length frequency band product of the template
            trigCon==1
                A STA/LTA type scheme is used on the vector of correlations coeficients using triggerLTATime and triggerSTATime
                trigParameter is the required ratio. Around 5 usuaully works well
            trigCon==2
                Use the beta distribtion false detection statistic stored in FASfile
                trigParameter is a the acceptable likelihood of a false detection. Keep in mind the distribution probably wont fit well
                in th tails so one should be extremely conservative with this parameter (something like 10**-8 or so)
            trigCon==3
                use the normal distribution parameters in the false detection statistic file.
                trigParameter is the number of sigma to the right of the mean. Because the normal distribution does not have hard limits
                as does the correlation coefficient this will only be approximately correct if the distribtution converges quickly
    
    tempKeyPath : str
        Path the to template key
    ConDir : str
        Path to the directory where the continous waveforms are stored
    EveDir : str
        Path the the directory where the event (template) waveforms are stored
    pks : str 
        Path to the the eventpick file created using detex.util.trimTemplates
    CorDB : str
        Name/path of SQL lite database to be created in which the detections will be stored
    FASfile : str
        Path to the False alarm statistic file created using detex.xcorr.getFAS function. Only used if trigCon==3 or trigCon==4
    UTCstart : str, float, int or None 
        A parameter to define the start time of the data processing. If None all of the data in the ConDir will be processed. If not None
        must be an obspy.core.UTCDateTime readable format
    UTCend : str, float, int or None 
        Same as UTCstart but for an end time
    triggerLTATime : int or float
        Number of seconds to use as the LTA in the STA/LTA calculations of the correlation coeficient vector.Value is calculated regardless of 
        trigCon but used as a trigger only if trigCon==1
    triggerSTATime int or float
        Number of seconds to use as the STA time in the STA/LTA calculations. A value of 0 (highly recommended) indicates a single sample is 
        used
    decimate : int or False
        Decimation factor as used by the obspy stream decimation method. If trigCon==3 or trigCon==4 make sure to use the same decimation factor
        that was used in calculating the false alarm statistics. False means no decimation
    filt : list [number,number,number,boolean]
        obspy bandpass filter parameters. [freqmin,freqmax,corners,zerophase]
    delOldCorr : boolean
        If true delete any file named CorDB before starting any correlations
    extrapolateTime: boolean
        If true use cosine extrapolation to try and get subsample precision on lag times
    templateDuration : int, float or None
        The duration in seconds to make each template from defined start time. If None use endtimes in pks (NOT YET SUPPORTED)
        if trigCon==3 or trigCon==4 make sure to use the same templateDuration that was used in creating the FAS file
    multiprocess : boolean
        If True tries to fork various processes in order to improve efficiency using joblib (CURRENTLY DOESNT WORK)
    secB4FirstArrival : int, float
        Number of seconds before the arrivial time in pks to use as starttime of templates
    calcHist : boolean
        If True a continous histogram is kept throughout the process so the distribtion of all the correlation coeficients can be examined
    histFIle : str
        Path of the histogram file to save all the histograms of correlation coeficient to
    averageAllTemplates : boolean
        If true than all the correlograms and thresholds are averaged for every hour of continous data and detections for individual
        templates are not recorded
        
    Returns
    -------------
    Creates corrsDB and 
    """
    DT=Detex(trigCon,trigParameter,tempKeyPath,ConDir,EveDir,pks,CorDB,FASfile,UTCstart,UTCend,triggerLTATime,triggerSTATime,decimate,filt,
                delOldCorr,extrapolateTimes,templateDuration,multiprocess,secB4FirstArrival,calcHist,histFile,averageAllTemplates)
        
class Detex(object):
    """
    Class to perform correlations
    """
        
    def __init__(self,trigCon,trigParameter,tempKeyPath,ConDir,EveDir,pks,CorDB,FASfile,UTCstart,UTCend,triggerLTATime,triggerSTATime,decimate,filt,
                delOldCorr,extrapolateTimes,templateDuration,multiprocess,secB4FirstArrival,calcHist,histFile,averageAllTemplates):

        self.__dict__.update(locals()) # Instantiate all input variables
        self._validateTrigCon()
        self.ContinousDataLength=detex.util.getContinousDataLength(Condir=ConDir)
        if trigCon==2 or trigCon==3:
            self.FAS=pd.read_pickle(FASfile)
        self.pksDF=pd.read_pickle(pks)
        if os.path.exists(histFile): #if histfile already exists remove it
            os.remove(histFile)
        if delOldCorr==True: #delete old results db if told to do so
            detex.util.DoldDB(CorDB)
        self.histBins=np.linspace(-1,1,num=401) #create histogram bins
        self.tempKey=pd.read_csv(tempKeyPath) #read template key
        self.Events=[str(x) for x in self.tempKey['NAME']]
        self._saveFiltParams() #Creates a seperate table in Corrs.db and saves the filter params if used. Necesary for CorResults class later
        eveDF=pd.DataFrame()
        for evenum,eve in self.tempKey.iterrows(): # Iterate through each event
            staKey=pd.read_csv(eve.STATIONKEY)
            Stations=staKey['NETWORK']+'.'+staKey['STATION']
            for sta in Stations: #itterate through each station and create dataframe so no template data has to be accsessed later
                wf=glob.glob(os.path.join(self.EveDir,eve.NAME,sta+'*'))
                if len(wf)==0 or len(wf)>1:
                    print ('Either no file found for Event %s station %s or multiple files found, skipping. Check %s' %(eve.NAME,sta,self.EveDir))
                elif len(wf)==1: #if the file exists
                    try:
                        pk=self.pksDF[(self.pksDF.Name==eve.NAME)&(self.pksDF.Station==sta)].iloc[0]
                    except IndexError:
                        raise Exception('%s on station %s is not in the pks file, run detex.trimTemplates again'%(eve.NAME,sta))
                    offset=obspy.core.UTCDateTime(pk.Starttime)-self.secB4FirstArrival-obspy.core.UTCDateTime(eve.TIME) # first arrivial minus origin time
                    threshold=self._getThreshold(eve.NAME,sta) # calculate the threshold for this event
                    TR,nonImportant=self._loadTemplate(wf[0],pk)
#                    global tr
#                    tr=obspy.read(wf[0])
                    Nc=len(list(set([x.stats.channel for x in TR])))
                    MPtem=self.multiplex(TR,Nc,Template=True)
                    reqlen=int(len(MPtem)+Nc*self.ContinousDataLength*TR[0].stats.sampling_rate-1)
                    MPtemFD,sum_nt=self.FFTemplate(MPtem,reqlen)
                    df=pd.DataFrame([[sta,eve.NAME,wf[0],offset,threshold,TR,Nc,MPtem,reqlen,MPtemFD,sum_nt]],
                                    columns=['Station','Event','Path','offset','threshold','TR','Nc','MPtem','reqlen','MPtemFD','sum_nt'])
                    eveDF=eveDF.append(df,ignore_index=True) #not the best way to do this, try harder next time
        eveDF.reset_index(inplace=True,drop=True)
        self.histDF=self._makeHistDF(eveDF)
        if len(eveDF)<1:
            raise Exception ('event dataframe empty, make sure template key and station key are correct and the waveforms are downloaded')
        jobs = []
        for stanum,starow in staKey.iterrows(): # loop throuogh each station and perform all correlations
            if multiprocess==True:
                p = multiprocessing.Process(target=self._CorStations(eveDF,starow))
                jobs.append(p)
                p.start()
            else:
                self._CorStations(eveDF,starow)
        if self.calcHist:
            self.histDF.to_pickle(self.histFile)
        
    def _CorStations(self,eveDF,starow): #TODO rearrange loops so continous data is only read in once and all templates operate before next read in=huge IO savings           
        sta=starow.NETWORK+'.'+starow.STATION
        DF=pd.DataFrame() # Initialize emptry data frame
        conrangejulday,conrangeyear=self._getContinousRanges(self.ConDir,sta,self.UTCstart,self.UTCend)
        if len(conrangeyear)<1:
            print('No files found for station %s, make sure they exist in the continous data directory'%sta)
        evedf=eveDF[eveDF.Station==sta]
        chans=starow['CHANNELS'].split('-')
        dfCompletes=0 #simple counter to see when DF is saved
        for a in range(len(conrangeyear)): # for each year
            for b in conrangejulday[a]: # for each julian day   
                FilesToCorr=glob.glob(os.path.join(self.ConDir,sta,str(conrangeyear[a]),str(b),'*'))
                for ftc in FilesToCorr:                 
                    CorDF,MPcon=self._getRA(ftc,evedf,sta)
                    if self.averageAllTemplates:
                        CorDF=self._getAverageCorDF(CorDF)
                    if not isinstance(CorDF,pd.DataFrame): #if any aspect of CorDF failed, skip hour
                        continue
                    for coreve,corrow in CorDF.iterrows(): #Loop each shared hour, or row of Tstamp and Results
                        if self._evalTriggerCondition(corrow): # Trigger Condition
                            Sar=self._CreateCoeffDF(corrow,coreve,evedf,MPcon)
                            if len(Sar)>300:
                                print 'over 300 events found in single trace, perphaps min_coef is too low?' 
                            if len(Sar)>0:
                                DF=DF.append(Sar,ignore_index=True)
                            if len(DF)>1000:
                                DF.reset_index(inplace=True,drop=True)
                                DF.sort(inplace=True)
                                detex.util.saveSQLite(DF,self.CorDB,'cor_df')
                                DF=pd.DataFrame()
                                dfCompletes+=1
        if len(DF)>0:
            detex.util.saveSQLite(DF,self.CorDB,'cor_df')  
        print ('Correlations for %s completed, %d potential detection(s) recorded' %(sta,1000*dfCompletes+len(DF)))
    
    def _getAverageCorDF(self,CorDF):
        ind=CorDF.index[0]
        avCors=pd.DataFrame(index=[ind],columns=CorDF.columns)
        for col in CorDF.columns:
            avCors[col][ind]=CorDF[col].mean()
        avCors['MaxCC'][ind]=avCors['Xcor'][ind].max()
        avCors['MaxSTALTA'][ind]=avCors['MaxSTALTA'][ind].max()
        return avCors

    def _validateTrigCon(self): # function to make sure given trigcons and trigParameters will work and have all required parts
        if not os.path.exists(self.pks):
            raise Exception('%s not found, a pks file is required to run correlations. Create it by calling detex.util.trimTemplates' % self.pks)
        if self.trigCon==2 or self.trigCon==3:
            if not os.path.exists(self.FASfile):
                raise Exception('%s not found, a FAS file is required to when trigCon = 2  or 3. Create it by calling detex.xcorr.getFAS' % self.FASfile)
        if self.trigCon==0:
            if not isinstance(self.trigParameter,numbers.Number) or abs(self.trigParameter)>1:
                raise Exception('When trigCon = 0 trigParameter is the required correlation coeficient and therefore must be a number between 0 and 1')
        elif self.trigCon==1:
            if not isinstance(self.trigParameter,numbers.Number) or self.trigParameter<1:
                raise Exception('When trigCon = 1 triparameter is the sta/lta of the correlation coeficients and therefore must be a number greater than 1')

        elif self.trigCon==2:
            if not isinstance(self.trigParameter,numbers.Number) or self.trigParameter < 0 or self.trigParameter >1:
                raise Exception('trigParameter must be a likelihood (between 0 and 1, usually a number very close to 0) when trigCon=2')
        elif self.trigCon==3:
            if not isinstance(self.trigParameter,numbers.Number):
                raise Exception('trigParameter must be the number of sigma left of the mean of the normal distribution to use as an acceptable false detection\
                likelihood')
        else:
            raise Exception('trigCon not supported, see docs for supported values')
            

    def _getThreshold(self,event,station):
        if self.trigCon==0 or self.trigCon==1:
            thresh=self.trigParameter
        elif self.trigCon==3 or self.trigCon==2:
            fas=self.FAS.loc[event,station]
            if self.trigCon==2:
                betaDist=fas['betadist']
                thresh=scipy.stats.beta.isf(self.trigParameter,betaDist[0],betaDist[1])*2 -1 
                thresh=scipy.stats.norm.isf(self.trigParameter,fas['normparams'][0],fas['normparams'][1]) 
            if self.trigCon==3:
                thresh=fas['normparams'][1]*self.trigParameter+fas['normparams'][0]
        return thresh
        

    def _makeHistDF(self,eveDF): #setup a dataframe for storing histograms
        try:
            stations=list(set(eveDF.Station.values))
        except:
            deb(eveDF)
        stations.sort()
        events=list(set(eveDF.Event.values))
        events.sort()
        hDF=pd.DataFrame(index=events,columns=stations)
        for sta in stations:
            for event in events:
                histdict={'bins':self.histBins,'hist':np.zeros(len(self.histBins)-1),
                'histSTALTA':np.zeros(len(self.histBins)-1)}
                hDF[sta][event]=histdict
        return hDF
               
    def _applyFilter(self,Trace,trimtime=None,condat=False):# Apply a filter/decimateion to an obspy trace object and trim 
        
        Trace.detrend('linear')
        startTrim=max([x.stats.starttime for x in Trace])
        endTrim=min([x.stats.endtime for x in Trace])
        Trace=Trace.slice(starttime=startTrim,endtime=endTrim) 
        Trace.sort()

        if self.decimate: # Decimate to the point so that sr is about 2.5 time high frequency
            Trace.decimate(self.decimate)
        
        if self.filt!=None:
            Trace.filter('bandpass',freqmin=self.filt[0],freqmax=self.filt[1],corners=self.filt[2],zerophase=self.filt[3])
        if trimtime!=None:
            if self.trimSeconds==None:
                Trace=Trace.slice(starttime=obspy.core.UTCDateTime(trimtime[0]),endtime=obspy.core.UTCDateTime(trimtime[1]))
            else:
                Trace=Trace.slice(starttime=obspy.core.UTCDateTime(trimtime[0]),endtime=obspy.core.UTCDateTime(trimtime[0])+self.trimSeconds)
        if self.templateDuration and condat:  #trim off the bit of extra cotninous waveform #TODO Make this work with varaible template sizes
            starttime,endtime=min([x.stats.starttime for x in Trace]),max([x.stats.endtime for x in Trace])
            duration=endtime-starttime
            secToTrim=(duration-self.templateDuration)%3600
            if endtime-secToTrim<starttime:
                print('Major data quality issue found with \n%s, skipping' %Trace)
                return None
            Trace=Trace.slice(starttime=starttime,endtime=endtime-secToTrim)
       # Trace.sort()
        #Trace.detrend()
        return Trace
        
    def _loadTemplate(self,path,trimDF): #load, trim and filter tempalte
        TR=self._applyFilter(obspy.core.read(path))
#        TR.sort()
#        TR.detrend('linear')
        trimtime=[trimDF.Starttime-self.secB4FirstArrival,trimDF.Endtime]
#        if self.filt != None:
#            TR.filter('bandpass',freqmin=self.filt[0],freqmax=self.filt[1],corners=self.filt[2],zerophase=self.filt[3])
#        if self.decimate:
#            TR.decimate(self.decimate)
        if self.templateDuration == None:
            pass
            TR.trim(starttime=obspy.core.UTCDateTime(trimtime[0]),endtime=obspy.core.UTCDateTime(trimtime[1]))
        else:
            TR.trim(starttime=obspy.core.UTCDateTime(trimtime[0]),endtime=obspy.core.UTCDateTime(trimtime[0])+self.templateDuration)
            TR.detrend('linear')
        return TR,trimtime
        
    def _saveFiltParams(self):
        if self.filt:
            if len(self.filt)!=4: # Make sure valid parameters are passed
                raise Exception('Bad Filter Parameters, see docs for correct format')
            elif not all([isinstance(x,(numbers.Real)) for x in self.filt[0:3]]) and all([isinstance(x,bool) for x in self.filt[3:4]]):
                raise Exception('Bad Filter Parameters, see docs for correct format')
            if self.decimate and not(isinstance(self.decimate,int)):
                raise Exception('Bad decimation value, must either be None or an integer (generally less than 10)')
            #deb([self.filt,self.decimate])
            df=pd.DataFrame([self.filt+[self.decimate]],columns=['FREQMIN','FREQMAX','CORNERS','ZEROPHASE','DECIMATE'],index=[0])
            detex.util.saveSQLite(df,self.CorDB,'filt_params')
    
    def _calcAlpha(self,trigIndex,MPcon,MPtem,Nc): 
        """
        calculate alpha using iterative method outlined in Gibbons and Ringdal 2007.
        This is a bit rough and dirty at the moment, there are better methods to use
        """
        #xtem=MPcon[Nc*(trigIndex)-4:Nc*(trigIndex)+len(MPtem)+4]
        #xcor=np.correlate(xtem,MPtem,mode='valid')
        #deb([xtem,xcor,Nc,trigIndex])
        
        X,Y=MPtem,MPcon[Nc*(trigIndex):Nc*(trigIndex)+len(MPtem)]
        alpha0=np.dot(X,Y)/np.dot(X,X)
        for a in range(10): 
            if a==0:
                alpha=alpha0
                alphaPrevious=alpha
            res=Y-alpha*X
            Weight=1/(.000000001+np.sqrt(np.abs(res)))
            alpha=np.dot(Y,np.multiply(X,Weight))/np.dot(X,np.multiply(X,Weight))
            if (abs(alpha-alphaPrevious))/(alphaPrevious)<0.01: #If change is less than 1% break, else go 10 iterations
                break
            alphaPrevious=alpha
        #print (X.argmax(),Y.argmax(),alpha)    
        return alpha                
                
    def _CreateCoeffDF(self,corrow,coreve,evdf,MPcon): # creates a datafram with necesariy info about each detection
        # TODO this function is old and sloppy, clean up when possible
        everow=evdf[evdf.Event==coreve].iloc[0]
        dpv=0
        if self.trigCon in [0,2,3]:
            Ceval=corrow.Xcor.copy()
        elif self.trigCon==1:
            Ceval=corrow.STALTA.copy()
        trigIndex=Ceval.argmax()
        alpha=pd.Series(self._calcAlpha(trigIndex,MPcon,everow.MPtem,everow.Nc))
        coef=pd.Series(corrow.Xcor[trigIndex])
        if self.extrapolateTimes:
            times=pd.Series(self._subsampleExtrapolate(Ceval,trigIndex,corrow.SampRate,corrow.TimeStamp))
            times1=[float(trigIndex)/corrow.SampRate+corrow.TimeStamp+self.secB4FirstArrival]
            if abs(times1[0]-(times[0]+self.secB4FirstArrival))>1.0/corrow.SampRate:
                raise Exception('subsample extrapolation shifts time more than one sample')
        else:
            times1=[float(trigIndex)/corrow.SampRate+corrow.TimeStamp]
            times=pd.Series(times1)
        SLValue=pd.Series(corrow.STALTA[trigIndex])
        Ceval=self._downPlayArrayAroundMax(Ceval,corrow.SampRate,dpv) # This doesnt seem to be working, exmain when you can #################################################
        count=0
        while Ceval.max()>=corrow.threshold: 
            trigIndex=Ceval.argmax()
            coef[len(coef)]=corrow.Xcor[trigIndex]
            #times=times+[float(trigIndex)/sr+starttime] corrow
            if self.extrapolateTimes:
                times[len(times)]=self._subsampleExtrapolate(Ceval,trigIndex,corrow.SampRate,corrow.TimeStamp)
                times1=[float(trigIndex)/corrow.SampRate+corrow.TimeStamp+self.secB4FirstArrival]
                if abs(times1[0]-(times[len(times)-1]+self.secB4FirstArrival))>1.0/corrow.SampRate:
                    raise Exception('subsample extrapolation shifts time more than one sample')
            else:
                times1=[float(trigIndex)/corrow.SampRate+corrow.TimeStamp+self.secB4FirstArrival]
                times[len(times)]=times1
            SLValue[len(SLValue)]=corrow.STALTA[trigIndex]
            alpha[len(alpha)]=self._calcAlpha(trigIndex,MPcon,everow.MPtem,everow.Nc)
            Ceval=self._downPlayArrayAroundMax(Ceval,corrow.SampRate,dpv)
            count+=1
            if count>4000:
                raise Exception (' _CreatCoeffArray loop exceeds limit')
        Sar=pd.DataFrame()
        Sar['Coef'],Sar['STA_LTACoef'],Sar['STMP'],Sar['alpha']=coef,SLValue, times,alpha
        Sar['Template']=coreve
        Sar['Sta']=everow.Station
        Sar['MSTMP']=Sar['STMP']-everow.offset
        if any([np.isnan(x) for x in Sar['MSTMP']]):
            deb([Sar,everow,corrow,coreve,evdf,MPcon])
        #Sar=pd.DataFrame([[coef,SLValue, times,alpha]],columns=['Coef','STA_LTACoef','STMP','alpha'])
        return Sar
        
    def _makeDataFrame(self,eve,sta,Sar,offSetDict):
        Sar['Template'],Sar['Sta'],Sar['MSTMP']=eve,sta,[x-offSetDict[sta] for x in Sar['STMP'].tolist()]
        return Sar
        
    def _subsampleExtrapolate(self,Ceval,trigIndex,sr,starttime) :
        """ Method to estimate subsample time delays using cosine-fit interpolation
        Cespedes, I., Huang, Y., Ophir, J. & Spratt, S. 
        Methods for estimation of sub-sample time delays of digitized echo signals. 
        Ultrason. Imaging 17, 142–171 (1995)"""
        ind=Ceval.argmax()
        if trigIndex != ind:
            raise Exception('something is messed up, trigIndex and CC.argmax no equal')
        if ind==0 or ind==len(Ceval)-1: # If max occurs at beg or end of CC set as beg or end, no extrapolation
            tau=float(ind)/sr + starttime
        else:
            alpha=np.arccos((Ceval[ind-1]+Ceval[ind+1])/(2*Ceval[ind]))
            tau=-(np.arctan((Ceval[ind-1]-Ceval[ind+1])/(2*Ceval[ind]*np.sin(alpha)))/alpha)*1.0/sr+ind*1.0/sr+starttime
            if -np.arctan((Ceval[ind-1]-Ceval[ind+1])/(2*Ceval[ind]*np.sin(alpha)))/alpha >1:
                raise Exception('Something wrong with extrapolation, more than 1 sample shift predicted ')
        return tau
        
    def _replaceNanWithMean(self,arg): # Replace where Nans occur with closet non-Nan value
        ind = np.where(~np.isnan(arg))[0]
        first, last = ind[0], ind[-1]
        arg[:first] = arg[first+1]
        arg[last + 1:] = arg[last]
        return arg
    
    def _getStaLtaArray(self,C,LTA,STA): # Get STA/LTA 
        if STA==0:
            STA=1
            STArray=np.abs(C)
        else:
            STArray=pd.rolling_mean(np.abs(C),STA,center=True)
            STArray=self._replaceNanWithMean(STArray)
        LTArray=pd.rolling_mean(STArray,LTA)
        try:
            LTArray=self._replaceNanWithMean(LTArray)
        except:
            deb([C,LTA,STA])
        
        out=np.divide(STArray,LTArray)
        out[np.where(np.isinf(out))]=0.0 #make sure no infs
        return out
        
    def _evalTriggerCondition(self,corrow,returnValue=False): 
        """ Evaluate if Trigger condition is met and return true or false or correlation value if returnValue=True
        """
        Out=False
        if self.trigCon in [0,2,3]:
            trig=corrow.MaxCC
            if trig>corrow.threshold:
                Out=True
        elif self.trigCon==1:
            trig=corrow.MaxSTALTA
            if trig>corrow.threshold:
                Out=True
            #trig=Cors[maxIn]
        if returnValue==True:
            return trig
        if returnValue==False:
            return Out
                
                
    def RMS(self,data): # calculate RMS of vector
        rms=np.sqrt(np.mean(np.square(data)))
        return rms
    
    def fast_normcorr(self,t, s): # Fast normalized Xcor
        if len(t)>len(s): #switch t and s if t is larger than s
            t,s=s,t
        n = len(t)
        nt = (t-np.mean(t))/(np.std(t)*n)
        sum_nt = nt.sum()
        a = pd.rolling_mean(s, n)[n-1:]
        b = pd.rolling_std(s, n)[n-1:]
        b *= np.sqrt((n-1.0) / n)
        c = np.convolve(nt[::-1], s, mode="valid")
        result = (c - sum_nt * a) / b    
        return result, b, np.std(t),n
        
    def _downPlayArrayAroundMax(self,C,sr,dpv,buff=20): #function to zero out correlation arrays around where max occurs
        index=C.argmax()
        if index<buff*sr+1:
            C[0:int(index+buff*sr)]=dpv
        elif index>len(C)-buff*sr:
            C[int(index-sr*buff):]=dpv
        else:
            C[int(index-sr*buff):int(sr*buff+index)]=dpv
        return C
    
    def UTCfitin(self,yearrange,jdayrange,UTCstart,UTCend):
        """Function called by DeTex to make sure years and times fit into times and dates"""
        if not UTCstart:
            UTCstart=-np.inf
        if not UTCend:
            UTCend=np.inf
        jdrangeOut=[[]]*len(yearrange)
        for a in range(len(yearrange)):
            for b in range(len(jdayrange[a])):
                UTCrep=obspy.core.UTCDateTime(year=int(yearrange[a]),julday=int(jdayrange[a][b]))
                if UTCrep.timestamp <= UTCend.timestamp and UTCrep.timestamp >= UTCstart:
                    jdrangeOut[a].append(jdayrange[a][b])
        return jdrangeOut
    
            
    def _standardizeCorrelation(self,R): #Function to take CC coefficients and rolling std matricies and calculate combined coeficients
        X=np.zeros(len(R[0][0]))
        sigt=0
        sigs=np.zeros(len(R[0][1]))
        for a in range(len(R)):
            X=X+np.multiply(R[a][0],R[a][1])*R[a][2]*R[a][3]
            sigt=sigt+np.square(R[a][2]) #Equiv template sigma
            sigs=sigs+np.square(R[a][1]) #Equiv Search space sigma
        sigs=np.sqrt(sigs/len(R))
        sigt=np.sqrt(sigt/len(R))
        CC=np.divide(X,sigs*sigt*len(R)*R[0][3])
        
        return CC,X
        
    def _getTemplate(self,EveDir,eve,sta): #Function to load templates, also returns channels found in template
        eventSubDir=glob.glob(os.path.join(EveDir,eve,sta,'*'))[0] # Length will only be 1
        chans=[os.path.basename(x).split('.')[2].split('_')[0] for x in glob.glob(os.path.join(eventSubDir,'*'))]
        tem=[0]*len(chans)
        for a in range(len(chans)):
            tem[a]=obspy.core.read(os.path.join(eventSubDir,'*'+chans[a]+'*.sac'))
        global debug
        return chans, tem
    
    def _getContinousRanges(self,ConDir,sta,UTCstart,UTCend):
        # get lists of all hours and days for which continous data is avaliable
        conrangeyear=[os.path.basename(x) for x in glob.glob(os.path.join(ConDir,sta,'*'))]
        conrangejulday=[0]*len(conrangeyear)
        for a in range(len(conrangeyear)):
            conrangejulday[a]=[os.path.basename(x) for x in glob.glob(os.path.join(ConDir,sta,conrangeyear[a],'*'))]
        if UTCstart is not None and UTCend is not None: #Trim conrangeyear and conrangejulday to fit in UTCstart and UTCend if defined
            conrangejulday=self.UTCfitin(conrangeyear,conrangejulday,UTCstart,UTCend)
        return conrangejulday,conrangeyear
    
    def _checkTraceStartTimes(self,TR): #Make sure all channels of each station have equal starttimes
        T0=abs(TR[0].stats.starttime.timestamp)
        sr=TR[0].stats.sampling_rate
        for a in range(len(TR)):
            Tn=abs(TR[a].stats.starttime.timestamp)
            if Tn>2*sr+T0 or Tn<-sr*2+T0:
                raise Exception('Time stamps not equal for all channels of ' +TR[0].stats.station)
                        
    def _getFilesToCorr(self,sta,jday,year,chans,ConDir):
                                #Get unique hours and corresponding files for all components
        FilesToCorr=[] 
        for a2 in range(len(chans)):
            FilesToCorr=FilesToCorr+glob.glob(os.path.join(ConDir,sta,str(year),str(jday),chans[a2],'*'))
        sharedHours=list(set([os.path.basename(x).split('-')[1].split('.')[0].split('T')[1] for x in FilesToCorr]))
        FilesToCorr=[0]*len(chans)
        for a2 in range(len(chans)):
            FilesToCorr[a2]=[0]*len(sharedHours)
            for a3 in range(len(sharedHours)):
                #FilesToCorr is now a len(chans) by len(sharedHours) matrix populated with paths to continous waveforms
                FilesToCorr[a2][a3]=glob.glob(os.path.join(ConDir,sta,year,jday,chans[a2],'*'+'T'+sharedHours[a3]+'.sac'))
        return FilesToCorr, sharedHours
    
    def _getRA(self,ftc,evedf,sta):
        CorDF=pd.DataFrame(index=evedf.Event.values,columns=['Xcor','STALTA','TimeStamp','SampRate','MaxCC','MaxSTALTA','threshold','Nc'])
        #CorDF['FilesToCorr']=FilesToCorr
        
        conStream=self._applyFilter(obspy.core.read(ftc),condat=True)
        if not isinstance(conStream,obspy.core.stream.Stream):
            return None,None
        CorDF['Nc']=len(list(set([x.stats.channel for x in conStream])))
        CorDF['SampRate']=conStream[0].stats.sampling_rate
        MPcon,TR=self.multiplex(conStream,evedf.Nc.median(),retTR=True)
        CorDF['TimeStamp']=min([x.stats.starttime.timestamp for x in TR])
        #get continous data parameters for Xcor
        MPconFD=scipy.fftpack.fft(MPcon,n=2**int(evedf.reqlen.median()).bit_length())
        n = int(np.median([len(x) for x in evedf.MPtem])) ##TODO This assumes all templates are of equal length, if not will break
        a = pd.rolling_mean(MPcon, n)[n-1:]
        b = pd.rolling_std(MPcon, n)[n-1:]
        b *= np.sqrt((n-1.0) / n)

            
        for corevent,corrow in CorDF.iterrows():
            evrow=evedf[evedf.Event==corevent].iloc[0]
            CorDF.threshold[corevent]=evrow.threshold
            if len(MPcon)<=(len(evrow.MPtem))+self.triggerLTATime*corrow.SampRate: # make sure the template is shorter than continous data else skip
                return None,None           
            if isinstance(MPcon,np.ndarray): #If channels not equal in length and multiplexing fails delete hour, else continue with CCs
                try:
                    CorDF.Xcor[corevent]=self._MPXCorr(MPcon,MPconFD,evrow.MPtemFD,evrow.reqlen,evrow.sum_nt,evrow.MPtem,evrow.Nc,n,a,b)
                except:
                    deb([corevent,evrow,n,a,b,CorDF,MPcon,MPconFD])
                CorDF.MaxCC[corevent]=CorDF.Xcor[corevent].max()
                CorDF.STALTA[corevent]=self._getStaLtaArray(CorDF.Xcor[corevent],self.triggerLTATime*corrow.SampRate,self.triggerSTATime*corrow.SampRate)
                if max(CorDF.Xcor[corevent])==np.inf or max(CorDF.STALTA[corevent])==np.inf:
                    deb([sta,jday,year,chans,ConDir])
                CorDF.MaxSTALTA[corevent]=CorDF.STALTA[corevent].max()
            else:
                return None,None
            if self.calcHist: #calculate histograms
                #deb([self.histBins,CorDF.Xcor[corevent],self.histDF,corevent,sta])
                self.histDF.loc[corevent,sta]['hist']=self.histDF.loc[corevent,sta]['hist']+np.histogram(CorDF.Xcor[corevent],bins=self.histBins)[0]
                self.histDF.loc[corevent,sta]['histSTALTA']=self.histDF.loc[corevent,sta]['histSTALTA']+np.histogram(CorDF.STALTA[corevent],bins=self.histBins)[0]
        return CorDF,MPcon


     
    def FFTemplate(self,MPtem,reqlen):# apply the fft to the template, pad appropriately
        n = len(MPtem)
        nt = (MPtem-np.mean(MPtem))/(np.std(MPtem)*n)
        sum_nt=nt.sum()
        MPtemFD=scipy.fftpack.fft(nt[::-1],n=2**reqlen.bit_length())
        return MPtemFD,sum_nt   


            
    def _MPXCorr(self,MPcon,MPconFD,MPtemFD,reqlen,sum_nt,MPtem,Nc,n,a,b): #multiplex normalized cross correlation
        # note: MPtemFD was previously normalized and time reversed so MPtemFD does not need to be normalized or conjugated
        #MPconFD=scipy.fftpack.fft(MPcon,n=2**int(reqlen).bit_length())
#        n = len(MPtem)
#        a = pd.rolling_mean(MPcon, n)[n-1:]
#        b = pd.rolling_std(MPcon, n)[n-1:]
#        b *= np.sqrt((n-1.0) / n)
        #deb([MPcon,MPtemFD,reqlen,sum_nt,MPtem,Nc])        
        c=scipy.fftpack.ifft(np.multiply(MPtemFD,MPconFD))[len(MPtem)-1:len(MPcon)]
        result = (c - sum_nt * a) / b
        #result[np.where(np.isinf(result))]=0.0 # when filling missing values with 0 infs can occur, change to 0 because they are mwaningless anyway
        return np.real(result[::Nc])     

    def multiplex(self,TR,Nc,trimTolerance=15,Template=False,retTR=False):
        if Nc==1:
            return TR[0].data
        if len(TR)>Nc:
            print('data fragmented for \n%s, keeping largest chunk of continous data'% (TR))
            TR=self._mergeChannels(TR)
        startime=[x.stats.starttime for x in TR] # find starttime for each trace
        TR.trim(starttime=max(startime)) # trim to min startime
        chans=[x.data for x in TR]
        minlen=np.array([len(x) for x in chans])  
        if max(minlen)-min(minlen) > 15:
            if Template:
                raise Exception('timTolerance exceeded, examin Template: \n%s' % TR[0])
            else:
                print('timTolerance exceeded, examin data, Trimming anyway: \n%s' % TR[0])
                trimDim=min(minlen)
                chansTrimed=[x[:trimDim] for x in chans]
        elif max(minlen)-min(minlen)>0 : #trim a few samples off the end if necesary
            trimDim=min(minlen)
            chansTrimed=[x[:trimDim] for x in chans]
        elif max(minlen)-min(minlen)==0:
            chansTrimed=chans
        C=np.vstack((chansTrimed))
        C1=np.ndarray.flatten(C,order='F')
        if retTR:
            return C1,TR
        else:
            return C1

    def _mergeChannels(self,TR): #function to find longest continous data chucnk and discard the rest
        channels=list(set([x.stats.channel for x in TR]))
        temTR=TR.select(channel=channels[0])
        lengths=np.array([len(x.data) for x in temTR])
        lemax=lengths.argmax()
        TR.trim(starttime=TR[lemax].stats.starttime,endtime=TR[lemax].stats.endtime)
        return TR

def deb(varlist):
    global de
    de=varlist
    sys.exit(1)  