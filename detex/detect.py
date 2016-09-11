# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:21:46 2015

@author: derrick
"""
# python 2 and 3 compatibility imports
from __future__ import print_function, absolute_import, unicode_literals
from __future__ import with_statement, nested_scopes, generators, division

import collections

import numpy as np
import obspy
import pandas as pd
import scipy
import pdb

import detex
from detex.construct import fast_normcorr, multiplex, _applyFilter


class _SSDetex(object):
    """
    Private class to run subspace detections or event classifications
    """

    def __init__(self, TRDF, utcStart, utcEnd, cfetcher, clusters, subspaceDB,
                 trigCon, triggerLTATime, triggerSTATime, multiprocess,
                 calcHist, dtype, estimateMags, classifyEvents, eventCorFile,
                 utcSaves, fillZeros, issubspace=True, pick_times=None):

        # Instantiate input varaibles that are needed by many functions
        self.utcStart = utcStart
        self.utcEnd = utcEnd
        self.filt = clusters.filt
        self.decimate = clusters.decimate
        self.triggerLTATime = triggerLTATime
        self.triggerSTATime = triggerSTATime
        self.calcHist = calcHist
        self.dtype = dtype
        self.estimateMags = estimateMags
        self.eventCorFile = eventCorFile
        self.utcSaves = utcSaves
        self.fillZeros = fillZeros
        self.issubspace = issubspace
        self.stakey = clusters.stakey
        self.classifyEvents = classifyEvents
        self.trigCon = trigCon
        self.subspaceDB = subspaceDB
        self.pick_times = pick_times
        self.offsets = self._get_offset_df(TRDF)


        # set DataFetcher and read classifyEvents key, get data length
        if classifyEvents is not None:
            self.eveKey = detex.util.readKey(classifyEvents)
            fetcher = clusters.fetcher
            dur = fetcher.timeBeforeOrigin + fetcher.timeAfterOrigin
        else:
            fetcher = cfetcher
            dur = fetcher.conDatDuration + fetcher.conBuff
        self.fetcher = fetcher
        self.dataLength = dur

        # if using utcSavs init list and make sure all inputs are UTCs
        if utcSaves is not None:
            if isinstance(utcSaves, collections.Iterable):
                self.UTCSaveList = []
                try:
                    ts = [obspy.UTCDateTime(x).timestamp for x in utcSaves]
                except ValueError:
                    msg = ('Not all elements in utcSaves are readable by obspy'
                           ' UTCDateTime class')
                    detex.log(__name__, msg, level='error')
                self.utcSaves = np.array(ts)
            else:
                msg = 'UTCSaves must be a list or tupple'
                detex.log(__name__, msg, level='error')

        # init histogram stuff if used
        if calcHist:
            self.hist = {}
            self.hist['Bins'] = np.linspace(0, 1, num=401)

        for sta in TRDF.keys():  # loop through each station 
            DFsta = TRDF[sta]  # current row (all ss or singletons on this sta)
            DFsta.reset_index(inplace=True, drop=True)
            if len(DFsta) > 0:
                self.hist[sta] = self._corStations(DFsta, sta)

                # if classifyEvents was used try to write results to DataFrame
            if classifyEvents is not None:
                try:
                    DFeve = pd.concat(self.eventCorList, ignore_index=True)
                    DFeve.to_pickle(self.eventCorFile + '_%s,pkl' % sta)
                except ValueError:
                    msg = 'classify events failed for %s, skipping' % sta
                    detex.log(__name__, msg, level='warn', pri=True)

        # If utcSaves was used write results to DataFrame
        if isinstance(utcSaves, collections.Iterable):
            try:
                DFutc = pd.concat(self.UTCSaveList, ignore_index=True)
                try:  # try and read, pass
                    DFutc_current = pd.read_pickle('UTCsaves.pkl')
                    DFutc = DFutc.append(DFutc_current, ignore_index=True)
                except Exception:
                    pass
                DFutc.to_pickle('UTCsaves.pkl')
            except ValueError:
                msg = 'Failed to save data in utcSaves'
                detex.log(__name__, msg, level='warning', pri=True)

    def _get_offset_df(self, TRDF):
        """
        make a dictonary of dataframes for getting offset times for
        each event/station on each subsapce
        """
        cols = ['event', 'offset']
        out = {}
        for item, val in TRDF.items():
            df = pd.DataFrame(columns=cols)
            for ind, row in val.iterrows():
                offsets = row.SampleTrims['Starttime']
                for event in row.Events:
                    ot = row.Stats[event]['origintime']
                    starttime = row.Stats[event]['starttime']
                    nc = row.Stats[event]['Nc']
                    sr = row.Stats[event]['sampling_rate']
                    chop_time = starttime + (offsets) / (sr * nc)
                    offset = chop_time - ot
                    df.loc[len(df), cols] = event, offset
            out[item] = df
        return out

    def _corStations(self, DFsta, sta):
        """
        Function to perform subspace detection on a specific station
        """
        # get station key for current station
        skey = self.stakey
        stakey = skey[skey.STATION == sta.split('.')[1]]

        # get chans, sampling rates, and trims
        channels = _getChannels(DFsta)
        samplingRates = _getSampleRates(DFsta)
        threshold = {x.Name: x.Threshold for num, x in DFsta.iterrows()}
        names = DFsta.Name.values
        names.sort()

        # make sure samp rate and chans is kosher, get trims
        if channels is None or samplingRates is None:
            return None
        samplingRate = samplingRates[0]
        contrim = self._getConTrims(DFsta, channels, samplingRate)

        # Proceed to subspace operations
        histdict = self._corDat(threshold, sta, channels, contrim, names,
                                DFsta, samplingRate, stakey)
        return histdict

    def _corDat(self, threshold, sta, channels, contrim, names,
                DFsta, samplingRate, stakey):
        """
        Function to perform subspace detection (sub function of _corStations)
        """
        # init various parameters
        numdets = 0  # counter for number of detections
        tableName = 'ss_df' if self.issubspace else 'sg_df'
        DF = pd.DataFrame()  # DF for results, dumped to SQL database
        histdic = {na: [0.0] * (len(self.hist['Bins']) - 1) for na in names}
        nc = len(channels)

        lso = self._loadMPSubSpace(DFsta, sta, channels, samplingRate, True)
        ssTD, ssFD, reqlen, offsets, mags, ewf, events, WFU, UtU = lso
        if self.classifyEvents is not None:
            datGen = self.fetcher.getTemData(self.evekey, stakey)
        else:
            datGen = self.fetcher.getConData(stakey, utcstart=self.utcStart,
                                             utcend=self.utcEnd,
                                             returnTimes=True)
        for st, utc1, utc2 in datGen:  # loop each data chunk
            msg = 'starting on sta %s from %s to %s' % (sta, utc1, utc2)
            detex.log(__name__, msg, level='info')
            if st is None or len(st) < 1:
                msg = 'could not get data on %s from %s to %s' % (
                    stakey.STATION.iloc[0], utc1, utc2)
                detex.log(__name__, msg, level='warning', pri=True)
                continue

            # make dataframe with info for each hour (including det. stats.)
            CorDF, MPcon, ConDat = self._getRA(ssTD, ssFD, st, nc, reqlen,
                                               contrim, names, sta)
            # if something is broken skip hours
            if CorDF is None or MPcon is None:
                msg = (('failing to run detector on %s from %s to %s ') %
                       (sta, utc1, utc2))
                detex.log(__name__, msg, level='warning', pri=True)
                continue

            # iterate through each subspace/single
            for name, row in CorDF.iterrows():
                if self.calcHist and len(CorDF) > 0:
                    try:
                        hg = np.histogram(row.SSdetect, bins=self.hist['Bins'])
                        histdic[name] = histdic[name] + hg[0]
                    except Exception:
                        msg = (('binning failed on %s for %s from %s to %s') %
                               (sta, name, utc1, utc2))
                        detex.log(__name__, msg, level='warning')
                if isinstance(self.utcSaves, collections.Iterable):
                    self._makeUTCSaveDF(row, name, threshold, sta, offsets,
                                        mags, ewf, MPcon, events, ssTD)
                if self._evalTrigCon(row, name, threshold):
                    Sar = self._CreateCoeffArray(row, name, threshold, sta,
                                                 offsets, mags, ewf, MPcon,
                                                 events, ssTD, WFU, UtU)
                    # if lots of detections are being made raise warning
                    if len(Sar) > 300:
                        msg = (('over 300 events found in singledata block, on'
                                ' %s form %s to %s perphaps minCoef is too '
                                'low?') % (sta, utc1, utc2))
                        detex.log(__name__, msg, level='warning', pri=True)
                    if any(Sar.DS > 1.05):
                        msg = (('DS values above 1 found in sar, at %s on %s '
                                'this can happen when fillZeros==True, removing'
                                ' values above 1') % (utc1, st[0].stats.station))
                        detex.log(__name__, msg, level='warn', pri=True)
                        Sar = Sar[Sar.DS <= 1.05]
                    if len(Sar) > 0:
                        DF = DF.append(Sar, ignore_index=True)
                    if len(DF) > 500:
                        detex.util.saveSQLite(DF, self.subspaceDB, tableName)
                        DF = pd.DataFrame()
                        numdets += 500
        if len(DF) > 0:
            detex.util.saveSQLite(DF, self.subspaceDB, tableName)
        detType = 'Subspaces' if self.issubspace else 'Singletons'
        msg = (('%s on %s completed, %d potential detection(s) recorded') %
               (detType, sta, len(DF) + numdets))
        detex.log(__name__, msg, pri=1)
        if self.calcHist:
            return histdic

    def _getRA(self, ssTD, ssFD, st, Nc, reqlen, contrim, names, sta):
        """
        Function to make DataFrame of this datachunk with all subspaces and 
        singles that act on it
        """
        cols = ['SSdetect', 'STALTA', 'TimeStamp', 'SampRate', 'MaxDS',
                'MaxSTALTA', 'Nc']
        CorDF = pd.DataFrame(index=names, columns=cols)
        utc1 = st[0].stats.starttime
        utc2 = st[0].stats.endtime
        try:
            conSt = _applyFilter(st, self.filt, self.decimate, self.dtype,
                                 fillZeros=self.fillZeros)
        except Exception:
            msg = 'failed to filter %s, skipping' % st
            detex.log(__name__, msg, level='warning', pri=True)
            return None, None, None
        if len(conSt) < 1:
            return None, None, None
        sr = conSt[0].stats.sampling_rate
        CorDF.SampRate = sr
        MPcon, ConDat, TR = multiplex(conSt, Nc, returnlist=True, retst=True)
        CorDF.TimeStamp = TR[0].stats.starttime.timestamp
        if isinstance(contrim, dict):
            ctrim = np.median(contrim.values())
        else:
            ctrim = contrim

        # Trim continuous data to avoid overlap
        if ctrim < 0:
            MPconcur = MPcon[:len(MPcon) - int(ctrim * sr * Nc)]
        else:
            MPconcur = MPcon

        # get freq. domain rep of data
        rele = 2 ** np.max(reqlen.values()).bit_length()
        MPconFD = scipy.fftpack.fft(MPcon, n=rele)

        # loop through each subpsace/single and calc sd
        for ind, row in CorDF.iterrows():
            # make sure the template is shorter than continuous data else skip

            if len(MPcon) <= np.max(np.shape(ssTD[ind])):
                msg = ('current data block on %s ranging from %s to %s is '
                       'shorter than %s, skipping') % (sta, utc1, utc2, ind)
                detex.log(__name__, msg, level='warning')
                return None, None, None
            ssd = self._MPXDS(MPconcur, reqlen[ind], ssTD[ind],
                              ssFD[ind], Nc, MPconFD)
            CorDF.SSdetect[ind] = ssd  # set detection statistic
            if len(ssd) < 10:
                msg = ('current data block on %s ranging from %s to %s is too '
                       'short, skipping') % (sta, utc1, utc2, ind)
                detex.log(__name__, msg, level='warning')
                return None, None, None
            CorDF.MaxDS[ind] = ssd.max()
            CorDF.Nc[ind] = Nc
            # If an infinity value occurs, zero it.
            if CorDF.MaxDS[ind] > 1.1:
                ssd[np.isinf(ssd)] = 0
                CorDF.SSdetect[ind] = ssd
                CorDF.MaxDS[ind] = ssd.max()
            if not self.fillZeros:  # dont calculate sta/lta if zerofill used
                try:
                    CorDF.STALTA[ind] = self._getStaLtaArray(
                        CorDF.SSdetect[ind],
                        self.triggerLTATime * CorDF.SampRate[0],
                        self.triggerSTATime * CorDF.SampRate[0])
                    CorDF.MaxSTALTA[ind] = CorDF.STALTA[ind].max()

                except Exception:
                    msg = ('failing to calculate sta/lta of det. statistic'
                           ' on %s for %s start at %s') % (sta, ind, utc1)
                    detex.log(__name__, msg, level='warn')
                    # else:
                    # return None, None, None
        return CorDF, MPcon, ConDat

    def _makeUTCSaveDF(self, row, name, threshold, sta, offsets, mags, ewf,
                       MPcon, events, ssTD):
        """
        Function to make utc saves dataframe, which allows times of interest
        to be saved and examined. Results are appended to UTCSaveList
        """
        TS1 = row.TimeStamp
        TS2 = row.TimeStamp + len(MPcon) / (row.SampRate * float(row.Nc))
        inUTCs = (self.utcSaves > TS1) & (self.utcSaves < TS2)
        if any(inUTCs):
            Th = threshold[name]
            of = offsets[name]
            dat = [sta, name, Th, of, TS1, TS2, self.utcSaves[inUTCs], MPcon]
            inds = ['Station', 'Name', 'Threshold', 'offset', 'TS1', 'TS2',
                    'utcSaves', 'MPcon']
            ser = pd.Series(dat, index=inds)
            df = pd.DataFrame(pd.concat([ser, row])).T
            self.UTCSaveList.append(df)
        return

    # function to load subspace representations
    def _loadMPSubSpace(self, DFsta, sta, channels, samplingRate,
                        returnFull=False):
        """
        Function to parse out important information from main DataFrame
        for performing subspace operations. Also recalcs the freq. domain
        rep of each event to the correct length to multiply by the feq domain
        rep of the continuous data
        """
        # init dicts that can be returned (Keys are subspace/single name)
        ssTD = {}
        ssFD = {}
        rele = {}
        offsets = {}
        mags = {}
        ewf = {}
        eves = {}
        WFU = {}
        UtUdict = {}

        # variables needed for analysis
        Nc = len(channels)  # num of channels
        dataLength = self.dataLength

        # get values and preform calcs
        for ind, row in DFsta.iterrows():
            events = row.Events
            if self.issubspace:
                U = np.array([row.SVD[x] for x in row.UsedSVDKeys])
                dlen = np.shape(U)[1]
                if 'Starttime' in row.SampleTrims.keys():
                    start = row.SampleTrims['Starttime']
                    end = row.SampleTrims['Endtime']
                    WFl = [row.AlignedTD[x][start:end] for x in events]
                    WFs = np.array(WFl)
                else:
                    WFl = [row.AlignedTD[x] for x in events]
                    WFs = np.array(WFl)
            else:  # if single trim and normalize (already done for subspaces)
                mptd = row.MPtd.values()[0]
                if row.SampleTrims:  # if this is a non empty dict
                    start = row.SampleTrims['Starttime']
                    end = row.SampleTrims['Endtime']
                    upr = mptd[start:end]
                else:
                    upr = mptd
                U = np.array([x / np.linalg.norm(x) for x in [upr]])
                dlen = len(upr)
                WFs = [upr]
            UtU = np.dot(np.transpose(U), U)
            r2d2 = dataLength * samplingRate * Nc  # beep beep
            reqlen = int(r2d2 + dlen)
            rbi = 2 ** reqlen.bit_length()
            mpfd = np.array([scipy.fftpack.fft(x[::-1], n=rbi) for x in U])
            mag = np.array([row.Stats[x]['magnitude'] for x in events])

            # Populate dicts
            ssFD[row.Name] = mpfd  # freq domain of required length
            ssTD[row.Name] = U  # basis vects
            mags[row.Name] = mag  # mag of events
            eves[row.Name] = events  # event names
            ewf[row.Name] = WFs  # event waveforms
            offsets[row.Name] = row.Offsets  # offsets (from eve origin)
            WFU[row.Name] = np.dot(WFs, UtU)  # events projected into subspace
            UtUdict[row.Name] = UtU  # UtU
            rele[row.Name] = reqlen  # required lengths

        if returnFull:
            return ssTD, ssFD, rele, offsets, mags, ewf, eves, WFU, UtUdict
        else:
            return ssTD, ssFD, rele

    def _CreateCoeffArray(self, corSeries, name, threshold, sta, offsets, mags,
                          ewf, MPcon, events, ssTD, WFU, UtU):
        """
        function to create an array of results for each detection, including
        time of detection, estimated magnitude, etc. 
        """
        dpv = 0
        cols = ['DS', 'DS_STALTA', 'STMP', 'Name', 'Sta', 'MSTAMPmin',
                'MSTAMPmax', 'Mag', 'SNR', 'ProEnMag', 'EstOrigin',
                'BestEvent']
        sr = corSeries.SampRate  # sample rate
        start = corSeries.TimeStamp  # start time of data block

        # set array to evaluate for successful triggers
        if self.trigCon == 0:
            Ceval = corSeries.SSdetect.copy()
        elif self.trigCon == 1:
            Ceval = corSeries.STALTA.copy()
        Sar = pd.DataFrame(columns=cols)
        count = 0
        # while there are any values in the det stat. vect that exceed thresh. 
        while Ceval.max() >= threshold[name]:
            trigIndex = Ceval.argmax()
            coef = corSeries.SSdetect[trigIndex]
            times = float(trigIndex) / sr + start
            if self.fillZeros:  # if zeros are being filled dont try STA/LTA
                SLValue = 0.0
            else:
                try:
                    SLValue = corSeries.STALTA[trigIndex]
                except TypeError:
                    SLValue = 0.0
            Ceval = self._downPlayArrayAroundMax(Ceval, sr, dpv)
            # estimate mags else return NaNs as mag estimates
            if self.estimateMags:  # estimate magnitudes
                M1, M2, ori, SNR, be = self._estMag(trigIndex, corSeries, MPcon,
                                           mags[name], events[name], WFU[name],
                                           UtU[name], ewf[name], coef, times,
                                           name, sta)
                peMag, stMag = M1, M2
            else:
                peMag, stMag, ori, SNR, be = np.NaN, np.NaN, np.NaN, np.NaN, ''

            # kill switch to prevent infinite loop (just in case)
            if count > 4000:
                msg = (('over 4000 events found in single data block on %s for'
                        '%s around %s') % (sta, name, times))
                detex.log(__name__, msg, level='error')

            # get predicted origin time ranges
            minof = np.min(offsets[name])
            maxof = np.max(offsets[name])
            MSTAMPmax, MSTAMPmin = times - minof, times - maxof
            Sar.loc[count] = [coef, SLValue, times, name, sta, MSTAMPmin,
                              MSTAMPmax, stMag, SNR, peMag, ori, be]
            count += 1
        return Sar

    def _estMag(self, trigIndex, corSeries, MPcon, mags, events,
                WFU, UtU, ewf, coef, times, name, sta):
        """
        Estimate magnitudes by applying projected subspace mag estimates 
        and standard deviation mag estimates as outlined in Chambers et al. 
        2015.
        """
        WFlen = np.shape(WFU)[1]  # event waveform length
        nc = corSeries.Nc  # number of chans
        # continuous data chunk that triggered  subspace
        ConDat = MPcon[trigIndex * nc:trigIndex * nc + WFlen]
        if self.issubspace:
            # continuous data chunk projected into subspace
            ssCon = np.dot(UtU, ConDat)
            # projected energy
            proEn = np.var(ssCon) / np.var(WFU, axis=1)

        # Try and estimate pre-event noise level (for estimating SNR)
        if trigIndex * nc > 5 * WFlen:  # take 5x waveform length before event
            pe = MPcon[trigIndex * nc - 5 * WFlen: trigIndex * nc]
            rollingstd = pd.rolling_std(pe, WFlen)[WFlen - 1:]
        else:  # if not enough data take 6 times after event
            pe = MPcon[trigIndex * nc: trigIndex * nc + WFlen + 6 * WFlen]
            rollingstd = pd.rolling_std(pe, WFlen)[WFlen - 1:]
        baseNoise = np.median(rollingstd)  # take median of std for noise level
        SNR = np.std(ConDat) / baseNoise  # estimate SNR
        # ensure mags are greater than -15, else assume no mag value for event
        pts = self.offsets[sta]
        pt = pts[(pts.event.isin(events))]
        touse = mags > -15
        if self.issubspace:  # if subspace
            if not any(touse):  # if no defined magnitudes available
                msg = (('No magnitudes above -15 usable for detection at %s on'
                        ' station %s and %s') % (times, sta, name))
                detex.log(__name__, msg, level='warn')
                return np.NaN, np.Nan, np.Nan, SNR, ''
            else:

                # correlation coefs between each event and data block
                ecor = [fast_normcorr(x, ConDat)[0] for x in ewf]
                eventCors = np.array(ecor)
                projectedEnergyMags = _estPEMag(mags, proEn, eventCors, touse)
                stdMags = _estSTDMag(mags, ConDat, ewf, eventCors, touse)
                df_t = pd.DataFrame([eventCors, mags, events],
                                    index=['cc', 'mag', 'event']).T
                ddf = pt.merge(df_t, on='event')
                ori = times - ddf[ddf.cc == ddf.cc.max()].offset.iloc[0]
                be = ddf[ddf.cc == ddf.cc.max()].iloc[0].event
        else:  # if singleton
            assert len(mags) == 1
            if np.isnan(mags[0]) or mags[0] < -15:
                projectedEnergyMags = np.NaN
                stdMags = np.NaN
            else:
                assert len(pt) == 1
                # use simple waveform scaling if single
                d1 = np.dot(ConDat, WFU[0])
                d2 = np.dot(WFU[0], WFU[0])
                projectedEnergyMags = mags[0] + d1 / d2
                stdMags = mags[0] + np.log10(np.std(ConDat) / np.std(WFU[0]))
                ori = times - pt.offset.iloc[0]
                be = events[0]
        return projectedEnergyMags, stdMags, ori, SNR, be

    def _estimate_origin(self, df_t):
        """ estimate origin time based on weighted average of CC """


    def _getStaLtaArray(self, C, LTA, ori, STA):
        """
        Function to calculate the sta/lta of the detection statistic 
        """
        if STA == 0:
            STA = 1
            STArray = np.abs(C)
        else:
            STArray = pd.rolling_mean(np.abs(C), STA, center=True)
            STArray = self._replaceNanWithMean(STArray)
        LTArray = pd.rolling_mean(np.abs(C), LTA, center=True)
        LTArray = self._replaceNanWithMean(LTArray)
        out = np.divide(STArray, LTArray)
        return out

    def _replaceNanWithMean(self, arg):
        """
        Function to replace any NaN values in sta/lta array with mean
        """
        ind = np.where(~np.isnan(arg))[0]
        first, last = ind[0], ind[-1]
        arg[:first] = arg[first + 1]
        arg[last + 1:] = arg[last]
        return arg

    def _evalTrigCon(self, Corrow, name, threshold, returnValue=False):
        """ 
        Evaluate if Trigger condition is met and return True or False.
        Also return detection statistic value if returnValue==True
        """
        Out = False
        if self.trigCon == 0:
            trig = Corrow.MaxDS
            if trig > threshold[name]:
                Out = True
        elif self.trigCon == 1:
            trig = Corrow.maxSTALTA
            if trig > threshold[name]:
                Out = True
        if returnValue:
            return trig
        if not returnValue:
            return Out

    def _downPlayArrayAroundMax(self, C, sr, dpv, buff=20):
        """
        function to zero out det. stat. array around where max occurs, 
        important to avoid counting side lobs as detections. 
        """
        index = C.argmax()
        if index < buff * sr + 1:
            C[0:int(index + buff * sr)] = dpv
        elif index > len(C) - buff * sr:
            C[int(index - sr * buff):] = dpv
        else:
            C[int(index - sr * buff):int(sr * buff + index)] = dpv
        return C

    def _MPXDS(self, MPcon, reqlen, ssTD, ssFD, Nc, MPconFD):
        """
        Function to preform subspace detection on multiplexed data
        MPcon is time domain rep of data block, MPconFD is freq. domain,
        ssTD is time domain rep of subspace, ssFD id freq domain rep,
        Nc is the number of channels in the multiplexed stream
        """
        n = np.int32(np.shape(ssTD)[1])  # length of each basis vector
        a = pd.rolling_mean(MPcon, n)[n - 1:]  # rolling mean of data block
        b = pd.rolling_var(MPcon, n)[n - 1:]  # rolling var of data block
        b *= n  # rolling power in vector
        sum_ss = np.sum(ssTD, axis=1)  # the sum of all the subspace basis vects
        ares = a.reshape(1, len(a))  # reshaped a
        sumres = sum_ss.reshape(len(sum_ss), 1)  # reshaped sum
        av_norm = np.multiply(ares, sumres)  # to account for non 0 mean vects
        m1 = np.multiply(ssFD, MPconFD)  # fd correlation with each basis vect
        # preform inverse fft
        if1 = scipy.real(scipy.fftpack.ifft(m1))[:, n - 1:len(MPcon)] - av_norm
        result = np.sum(np.square(if1), axis=0) / b  # get detection statistcs
        return result[::Nc]  # account for multiplexing

    def _getConTrims(self, df, chans, sr):
        """
        Get trim values, in samples, for each subspace/single. This is 
        only used on continuous data to prevent overlap so that detections 
        are not counted twice (although the results module can 
        take care of this)
        """
        outdi = {}
        nc = len(chans)
        for ind, row in df.iterrows():
            if self.classifyEvents is None:
                outdi[row.Name] = 0  # no trim if event classification
            else:
                start = row.SampleTrims['Starttime']
                end = row.SampleTrims['Endtime']
                contrim = self.fetcher.conBuff - (end - start) / (sr * nc)
                outdi[row.Name] = contrim
        return outdi


def _getChannels(df):
    """
    Function to get the channels on the main detex DataFrame
    """
    if isinstance(df, pd.DataFrame):
        row = df.iloc[0]
    else:
        row = df
    chansAr = np.array(row.Channels.values())
    chans = set([x for x in chansAr.flat])
    # make sure all channels are the same for each event
    if not all([chans == set(x) for x in row.Channels.values()]):
        msg = ('Not all channels are the same for all event on %s, skipping '
               'subspace or singles %s') % (row.Station, df.Names.values)
        detex.log(__name__, msg, level='warning', pri=True)
        return None
    return list(chans)


def _getSampleRates(df):
    """
    Function to get the sample rates on the main detex DataFrame
    """
    if isinstance(df, pd.DataFrame):
        row = df.iloc[0]
    else:
        row = df
    srs = set([row.Stats[x]['sampling_rate'] for x in row.Events])
    # make sure all sampling rates are the same else error out
    if len(srs) > 1:
        msg = ('Not all samp rates equal for all events on %s, skipping '
               'subspace or singles %s') % (row.Station, df.Names.values)
        detex.log(__name__, msg, level='warn', pri=True)
        return None
    return list(srs)


def _estPEMag(mags, proEn, eventCors, touse):
    """
    Function to estimate projected energy magnitude for subspaces. Squared 
    weighting on the correlation coef is applied. 
    """
    ma = 0.0
    weDenom = np.sum(np.square(eventCors[touse]))
    for x in range(len(proEn)):
        if mags[x] > -15:
            we = np.square(eventCors[x])
            lr = np.log10(np.sqrt(proEn[x]))
            ma += (mags[x] + lr) * we
    return ma / weDenom


def _estSTDMag(mags, ConDat, ewf, eventCors, touse):
    """
    Function to estimate standard deviation magnitude for subspaces. squared
    weighting on the correlation coef is applied.
    """
    ma = 0.0
    weDenom = np.sum(np.square(eventCors[touse]))
    for x in range(len(ewf)):
        if mags[x] > -15:
            we = np.square(eventCors[x])
            lr = np.log10(np.std(ConDat) / np.std(ewf[x]))
            ma += (mags[x] + lr) * we
    return ma / weDenom
