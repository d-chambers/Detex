# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:26:48 2015

@author: derrick
fas = false alarm stats
"""

from itertools import chain  # from_iterable

import numpy as np
import obspy
import pandas as pd
import scipy
import pdb
from obspy.signal.trigger import classic_sta_lta
from scipy.fftpack import fft


import detex


############## Subspace Detex and FAS #######################

def _initFAS(TRDF, conDatNum, cluster, fetcher, LTATime=5,
             STATime=0.5, numBins=401, dtype='double', staltalimit=7.5,
             issubspace=True, utcstart=None, utcend=None):
    """ Function to randomly scan through continuous data and fit statistical 
    distributions in order to get a DS threshold for each subspace/station 
    pair"""

    results = [{}] * len(TRDF)
    histBins = np.linspace(-.01, 1, num=numBins)  # create bins for histograms
    conLen = fetcher.conDatDuration + fetcher.conBuff  # con. data length (secs)
    TRDF.reset_index(drop=True, inplace=True)
    # Loop through each station on the subspace or singles data frame
    for ind, row in TRDF.iterrows():
        results[ind] = {'bins': histBins}
        # Load subspace (used left singular vectors or singles)
        if issubspace:
            ssArrayTD, ssArrayFD, reqlen, Nc = _loadMPSubSpace(row, conLen)
        else:
            ssArrayTD, ssArrayFD, reqlen, Nc = _loadMPSingles(row, conLen)
        sta = row.Station.split('.')[1]
        stakey = cluster.stakey[cluster.stakey.STATION == sta]

        if utcstart is None:
            utc1 = obspy.UTCDateTime(stakey.iloc[0].STARTTIME)
        else:
            utc1 = obspy.UTCDateTime(utcstart)
        if utcend is None:
            utc2 = obspy.UTCDateTime(stakey.iloc[0].ENDTIME)
        else:
            utc2 = obspy.UTCDateTime(utcend)
        # get processing params from cluster instance
        filt = cluster.filt
        deci = cluster.decimate
        dsvec, count, scount = _getDSVect(fetcher, stakey, utc1, utc2,
                                          filt, deci, dtype, conDatNum, Nc,
                                          reqlen, STATime, LTATime,
                                          ssArrayTD, ssArrayFD, staltalimit)
        if count != conDatNum:
            msg = '%d samps not avaliable, using all avaliable' % (conDatNum)
            detex.log(__name__, msg, level='warn')
        sratio = float(scount) / count  # success ratio
        if sratio <= .25:  # if failing sta/lta req
            msg = ('sta lta req of %d failing on station %s, dropping sta/lta'
                   ' requirement') % (staltalimit, stakey.STATION.iloc[0])
            detex.log(__name__, msg, level='warn', pri=True)
            dsvec, count, scount = _getDSVect(fetcher, stakey, utcstart,
                                              utcend, filt, deci, dtype,
                                              conDatNum, Nc, reqlen, STATime,
                                              LTATime, ssArrayTD, ssArrayFD)
        # flatten DS Vector
        if dtype == 'double':
            dss = np.fromiter(chain.from_iterable(dsvec), dtype=np.float64)
        elif dtype == 'single':
            dss = np.fromiter(chain.from_iterable(dsvec), dtype=np.float32)
            # Bin up and clean up
        results[ind]['bins'] = histBins
        results[ind]['hist'] = np.histogram(dss, bins=histBins)[0]
        # results[a[0]]['normdist']=scipy.stats.norm.fit(CCs)
        betaparams = scipy.stats.beta.fit(dss, floc=0, fscale=1)
        results[ind]['betadist'] = betaparams
        # calculate negative log likelihood for a "goodness of fit" measure
        results[ind]['nnlf'] = scipy.stats.beta.nnlf(betaparams, dss)
        # results[a[0]]['histRev']=histRev+np.histogram(CCsrev,bins=histBins)[0]
    return results


def _getDSVect(fetcher, stakey, utc1, utc2, filt, deci, dtype,
               conDatNum, Nc, reqlen, sta, lta, ssArrayTD, ssArrayFD,
               limit=None):
    # get a generator to return streams, ask for 4x more for rejects
    stgen = fetcher.getConData(stakey, utcstart=utc1, utcend=utc2,
                               randSamps=conDatNum * 4)
    count = 0  # normal count
    scount = 0  # success count
    DSmat = []
    for st in stgen:  # loop over random samps of continuous data
        if st is None or len(st) < 1:
            continue  # no need to log, fetcher will do it
        count += 1
        st = detex.construct._applyFilter(st, filt, deci, dtype)
        if st is None or len(st) < 1:
            continue  # no need to log, fetcher will do it
        passSTALTA = _checkSTALTA(st, filt, sta, lta, limit)
        if not passSTALTA:
            continue
        if scount >= conDatNum:  # if we have all we need
            break
        mpCon = detex.construct.multiplex(st, Nc)
        dsVect = _MPXSSCorr(mpCon, reqlen, ssArrayTD, ssArrayFD, Nc)
        DSmat.append(dsVect)
        scount += 1
    if count == 0:
        msg = 'Could not get any data for %s' % (stakey.STATION.iloc[0])
        detex.log(__name__, msg, level='error')
    return DSmat, count, scount


def _MPXSSCorr(MPcon, reqlen, ssArrayTD, ssArrayFD, Nc):
    """
    multiplex subspace detection statistic function
    """
    MPconFD = fft(MPcon, n=2 ** reqlen.bit_length())
    n = np.int32(np.shape(ssArrayTD)[1])  # length of each basis vector
    a = pd.rolling_mean(MPcon, n)[n - 1:]  # rolling mean of continuous data
    b = pd.rolling_var(MPcon, n)[n - 1:]  # rolling var of continuous data
    b *= n  # rolling power in vector
    sum_ss = np.sum(ssArrayTD, axis=1)  # the sume of all the subspace basis vects
    av_norm = np.multiply(a.reshape(1, len(a)), sum_ss.reshape(len(sum_ss), 1))
    m1 = np.multiply(ssArrayFD, MPconFD)
    if1 = scipy.real(scipy.fftpack.ifft(m1))[:, n - 1:len(MPcon)] - av_norm
    result1 = np.sum(np.square(if1), axis=0) / b
    return result1[::Nc]


def _loadMPSingles(row, conLen):
    """
    function to load trimed waveforms of singles
    """
    Nc = list(row.Stats.values())[0]['Nc']  # num of channels
    sts = row.SampleTrims['Starttime']
    ste = row.SampleTrims['Endtime']
    ssArrayTDp = np.array([row.MPtd[x][sts:ste] for x in row.MPtd.keys()])
    ssArrayTD = np.array([x / np.linalg.norm(x) for x in ssArrayTDp])  # normalize
    sr = conLen * list(row.Stats.values())[0]['sampling_rate']  # samp rate
    rele = int(sr * Nc + np.max(np.shape(ssArrayTD)))
    releb = 2 ** rele.bit_length()
    ssArrayFD = np.array([fft(x[::-1], n=releb) for x in ssArrayTD])
    return ssArrayTD, ssArrayFD, rele, Nc


def _loadMPSubSpace(row, conLen):
    """
    function to load subspace representations
    """
    if 'UsedSVDKeys' in row.index:  # test if input TRDF row is subspace
        if not isinstance(row.UsedSVDKeys, list):
            msg = ('SVD not defined, run SVD on subspace stream class before '
                   'calling false alarm statistic class')
            detex.log(__name__, msg, level='error')
        # check that all stations report the same channels
        if not all(x == list(row.Channels.values())[0]
                   for x in row.Channels.values()):
            msg = 'all stations in subspace do not have the same channels'
            detex.log(__name__, msg, level='error')
        Nc = len(list(row.Channels.values())[0])  # num of channels
        ssArrayTD = np.array([row.SVD[x] for x in row.UsedSVDKeys])
        sr = list(row.Stats.values())[0]['sampling_rate']  # samp rate
        rele = int(conLen * sr * Nc + np.max(np.shape(ssArrayTD)))
        releb = 2 ** rele.bit_length()
        # Get freq domain rep. with required length
        ssArrayFD = np.array([fft(x[::-1], n=releb) for x in ssArrayTD])
    return ssArrayTD, ssArrayFD, rele, Nc


def _checkSTALTA(st, filt, STATime, LTATime, limit):
    """
    Take a stream and make sure it's vert. component (or first comp 
    if no vert) does not exceed limit given STATime and LTATime
    Return True if passes, false if fails
    """
    if limit is None:
        return True
    if len(st) < 1:
        return None
    try:
        stz = st.select(component='Z')[0]
    except IndexError:  # if no Z found on trace
        return None
    if len(stz) < 1:
        stz = st[0]
    sz = stz.copy()
    sr = sz.stats.sampling_rate
    ltaSamps = LTATime * sr
    staSamps = STATime * sr
    try:
        cft = classic_sta_lta(sz.data, staSamps, ltaSamps)
    except:
        return False
        detex.deb([sz, staSamps, ltaSamps])
    if np.max(cft) <= limit:
        return True
    else:
        sta = sz.stats.station
        t1 = sz.stats.starttime
        t2 = sz.stats.endtime
        msg = ('%s fails sta/lta req of %d between %s and %s' % (sta, limit,
                                                                 t1, t2))
        detex.log(__name__, msg, level='warn')
        return False


def _replaceNanWithMean(self, arg):  # Replace where Nans occur with closet non-Nan value
    ind = np.where(~np.isnan(arg))[0]
    first, last = ind[0], ind[-1]
    arg[:first] = arg[first + 1]
    arg[last + 1:] = arg[last]
    return arg
