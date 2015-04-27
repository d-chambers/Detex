# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:00:46 2014

@author: Derrick
"""
import obspy, detex, pandas as pd, numpy as np
import operator

def manualPickerTrona(filelist='Tro.pkl',out='dataframe',chan='Z',filt=[1,10,2,True],staToView=['M18A','L18A','M17A']):
    DF=pd.read_pickle(filelist)
    if not 'Picked' in DF.columns.tolist():
        DF['Picked']=0
        DF['UTCPtime']=dict #intialize blank dict for station and times
    DFtoPick=DF[DF.Picked==0]
    for a in DFtoPick.iterrows():
        TR=obspy.core.read(a[1].FileName)
        Tell=obspy.core.Stream()
        for b in range(len(staToView)):
            T1=obspy.core.Trace(data=TR.select(station=staToView[b],channel='*z')[0].data)
            T1.stats.channel=staToView[b]
            T1.stats.station='SME'
            T1.stats.sampling_rate=40
            Tell+=T1
        Tell.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
        BPks=detex.streamPick.streamPick(Tell)
        global de
        de=BPks
        if len(BPks._picks)>0:    
            TR=TR.select(channel='*'+chan)
            TR.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
            Pks=detex.streamPick.streamPick(TR)
            outdict={}
            for b in Pks._picks:
                if b:
                    idd=b.waveform_id['station_code']
                    if b.phase_hint=='P':
                        outdict[idd]=b.time.timestamp
            DF.UTCPtime[a[0]]=outdict
            DF.Picked[a[0]]=1
            DF.to_pickle(filelist)



def manualPicker(numtoPick,filelist='Tro.pkl',out='dataframe',chan='Z',filt=[1,10,2,True]):
    DF=pd.read_pickle(filelist)
    if not 'Picked' in DF.columns.tolist():
        DF['Picked']=0
        DF['UTCPtime']=dict #intialize blank dict for station and times
    DFtoPick=DF[DF.Picked==0]
    DFtoPick=DFtoPick[0:numtoPick]
    for a in DFtoPick.iterrows():
        TR=obspy.core.read(a[1].FileName)
        TR=TR.select(channel='*'+chan)
        TR.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
        Pks=detex.streamPick.streamPick(TR)
        outdict={}
        for b in Pks._picks:
            if b:
                idd=b.waveform_id['station_code']
                if b.phase_hint=='P':
                    outdict[idd]=b.time.timestamp
        DF.UTCPtime[a[0]]=outdict
        DF.Picked[a[0]]=1
    DF.to_pickle(filelist)
    
def _writeMax(filelist='sortedTrigs.pkl'):
    DF=pd.read_pickle(filelist)
    DF['MaxSta']=str
    DF['StaAmps']=dict
    for a in DF.iterrows():
        TR=obspy.core.read(a[1].FileName)
        tramps={}
        for b in range(len(TR)):
            tramps[TR[b].stats.station]=np.max(np.abs(TR[b].data))
        maxsta=max(tramps.iteritems(), key=operator.itemgetter(1))[0]
        DF.MaxSta[a[0]]=maxsta
        DF.StaAmps[a[0]]=tramps
    DF.to_pickle(filelist)

        
        
            
        