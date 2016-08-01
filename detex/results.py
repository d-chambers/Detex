# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:55:01 2014

@author: Derrick
"""
from __future__ import (print_function, absolute_import, unicode_literals,
                        division)
from six import string_types

import numbers
import os
import numpy as np
import obspy
import pandas as pd
import scipy
import detex
import PyQt4
import sys
import pdb

def detResults(trigCon=0, trigParameter=0, associateReq=0,
               ss_associateBuffer=1, sg_associateBuffer=2.5,
               requiredNumStations=4, veriBuffer=1, ssDB='SubSpace.db',
               templateKey='TemplateKey.csv', stationKey='StationKey.csv',
               veriFile=None, includeAllVeriColumns=True, reduceDets=True,
               Pf=False, stations=None, starttime=None, endtime=None,
               fetch='ContinuousWaveForms', exceptionalThreshold=None):
    """
    Function to create an instance of the CorResults class. Used to associate 
    detections across multiple stations into coherent events. CorResults class 
    also has some useful methods for creating input files for location
    programs such as hypoDD and hypoInverse

    Parameters
    ---------
    trigCon : int 0 or 1
        trigCon is used for filtering the events in the corDB. Currently 
        only options are 0, raw detection statistic value or 1, STA/LTA 
        value of DS as reported in the corDB. This parameter might be 
        useful if there are a large number of detections in corDB with low 
        STA/LTA or correlation coeficient values.
    trigParameter : number
        if trigCon==0 trigParameter is the minimum correlation for a 
        detection to be loaded from the CorDB if trigCon==1 trigParameter is 
        the minimum STA/LTA value for a detection to be loaded from the CorDB. 
        Regardless of trigCon if trigParameter == 0 all detections in CorDB 
        will be loaded and processed
    associateReq : int
        The association requirement which is the minimum number of events 
        that must be shared by subspaces in order to permit event association.
        For example, subspace 0 (SS0) on station 1 was created using events 
        A,B,C. subspace 0 (SS0) on station 2 was created using events C, D. 
        If both subspace share a detection and associateReq=0 or 1 the 
        detections will be associated into a coherent event. If assiciateReq 
        == 2, however, the detections will not be associated.
    ss_associateBuffer : real number (int or float)
        The buffertime applied to subspace event assocaition in seconds
    sg_associateBuffer : real number (int or float)
        The buffertime applied to singleton event assocaition in seconds
    requiredNumStations : int
        The required number of a stations on which a detection must occur 
        in order to be classified as an event
    veriBuffer : real number (int, float)
        Same as associate buffer but for associating detections with events 
        in the verification file
    ssDB : str
        Path the the database created by detex.xcorr.correlate function
    templateKey : str
        Path to the template key
    stationKey : str
        Path to template key
    veriFile : None (not used) or str (optional)
        Path to a file in the TemplateKey format for verifying the detection 
        by origin time association. veriFile can either be an sqlite database
        with table name 'verify', a CSV, or a pickled pandas dataframe. The 
        following fields must be present: 'TIME','NAME','LAT','LON','MAG'. 
        Any additional columns may be present with any name that will be 
        included in the verification dataframe depending on 
        includeAllVeriColumns
    includeAllVeriColumns: bool
        If true include all columns that are in the veriFile in the verify 
        dataframe that will be part of the SSResults object.
    reduceDets : bool
        If true loop over each station and delete detections of the same 
        event that don't have the highest detection Stat value.
        It is recommended this be left as True unless there a specific 
        reason lower DS detections are wanted
    Pf : float or False
        The probability of false detection accepted. Uses the pickled 
        subspace object to get teh FAS class and corresponding fitted beta 
        in order to only keep detections with thresholds above some Pf, 
        defiend for each subspace station pair. If used values of 10**-8 
        and greater generally work well
    stations : list of str or None
        If not None, a list or tuple of stations to be used in the 
        associations. All others will be discarded
    starttime : None or obspy.UTCDateTime readable object
        If not None, then sets a lower bound on the MSTAMPmin column loaded
        from the dataframe
    endtime : None or obspy.UTCDateTime readable object
        If not None, then sets a lower bound on the MSTAMPmin column loaded
        from the dataframe
    fetch : str or instance of detex.getdata.DataFetcher
        Used to determine where to get waveforms, fed into 
        detex.getdata.quickfetch
    exceptionalThreshold : None, float, or dict
        If float the threshold required for a detection to be considered legit
        regardless of the number of stations. Can also be a dict where the 
        keys are the stations (net.station) and the values are the thresholds
        to consider exceptional for that station. 
    """
    _checkExistence([ssDB, templateKey, stationKey])
    _checkInputs(trigCon, trigParameter, associateReq,
                 ss_associateBuffer, requiredNumStations)
    if associateReq != 0:
        msg = 'associateReq values other than 0 not yet supported'
        raise detex.log(__name__, msg, level='error')

    # Try to read in all input files and dataframes needed
    temkey = detex.util.readKey(templateKey, 'template')  # load template key
    stakey = detex.util.readKey(stationKey, 'station')  # load station key

    ss_info, sg_info = _loadInfoDataFrames(ssDB)  # load info DataFrames
    fetcher = detex.getdata.quickFetch(fetch)

    # load histograms #TODO: Create visualization methods for hists
    # ss_hist = detex.util.loadSQLite(ssDB, 'ss_hist')
    # sg_hist = detex.util.loadSQLite(ssDB, 'sg_hist')

    filt = detex.util.loadSQLite(ssDB, 'filt_params')  # load filter Parameters

    ss_PfKey, sg_PfKey = _makePfKey(ss_info, sg_info, Pf)

    # Parse each station results and delete detections that occur on multiple
    # subpspace, keeping only the subspace with highest detection stat
    if reduceDets:
        ssdf = _deleteDetDups(ssDB, trigCon, trigParameter, ss_associateBuffer,
                              starttime, endtime, stations, 'ss_df',
                              PfKey=ss_PfKey)
        sgdf = _deleteDetDups(ssDB, trigCon, trigParameter, sg_associateBuffer,
                              starttime, endtime, stations, 'sg_df',
                              PfKey=sg_PfKey)
    else:
        if Pf:
            msg = 'When using the Pf parameter reduceDets must be True'
            detex.log(__name__, msg, level='error')
        ssdf = detex.util.loadSQLite(ssDB, 'ss_df')
        sgdf = detex.util.loadSQLite(ssDB, 'sg_df')
    if ssdf is None and sgdf is None:
        msg = 'No detections found that meet given criteria'
        detex.log(__name__, msg, level='error')
    df = pd.concat([ssdf, sgdf], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    if isinstance(stations, (list, tuple)):  # filter stations
        df = df[df.Sta.isin(stations)]

    # Associate detections on different stations together
    Dets, Autos = _associateDetections(df, associateReq, requiredNumStations,
                                       ss_associateBuffer, ss_info, temkey,
                                       exceptionalThreshold)

    # Make a dataframe of verified detections if applicable
    Vers = _verifyEvents(Dets, Autos, veriFile, veriBuffer,
                         includeAllVeriColumns)

    ssres = SSResults(Dets, Autos, Vers, ss_info, filt, temkey,
                      stakey, templateKey, fetcher)
    return ssres


def _makePfKey(ss_info, sg_info, Pf):
    """
    Make simple df for defining DS values corresponing to Pf for each 
    subspace station pair
    """
    if not Pf:  # if no Pf value passed simply return none
        return None, None

    ss_df = pd.DataFrame(columns=['Sta', 'Name', 'DS', 'betadist'])
    sg_df = pd.DataFrame(columns=['Sta', 'Name', 'DS', 'betadist'])
    if isinstance(ss_info, pd.DataFrame):
        for num, row in ss_info.iterrows():
            TH = scipy.stats.beta.isf(Pf, row.beta1, row.beta2, 0, 1)
            # if isf gives unrealistic pf, initiated forward grid serach
            if TH > .94:
                TH, Pftemp = _approximateThreshold(
                    row.beta1, row.beta2, Pf, 1000, 3)
            ss_df.loc[len(ss_df)] = [row.Sta, row.Name,
                                     TH, [row.beta1, row.beta2, 0, 1]]
        ss_df.reset_index(drop=True, inplace=True)
    else:
        ss_df = None
    if isinstance(sg_info, pd.DataFrame):
        for num, row in sg_info.iterrows():
            TH = scipy.stats.beta.isf(Pf, row.beta1, row.beta2, 0, 1)
            if TH > .94:
                TH, Pftemp = _approximateThreshold(
                    row.beta1, row.beta2, Pf, 1000, 3)
            sg_df.loc[len(sg_df)] = [row.Sta, row.Name,
                                     TH, [row.beta1, row.beta2, 0, 1]]
        sg_df.reset_index(drop=True, inplace=True)
    else:
        sg_df = None
    return ss_df, sg_df


def _approximateThreshold(beta_a, beta_b, target, numintervals, numloops):
    """
    Because scipy.stats.beta.isf can break, if it returns a value near 1 when 
    this is obvious wrong initialize grid search algo to get close to desired
    threshold using forward problem which seems to work where inverse fails
    See this bug report: https://github.com/scipy/scipy/issues/4677
    """

    startVal, stopVal = 0, 1
    loops = 0
    while loops < numloops:
        Xs = np.linspace(startVal, stopVal, numintervals)
        pfs = np.array([scipy.stats.beta.sf(x, beta_a, beta_b) for x in Xs])
        resids = abs(pfs - target)
        minind = resids.argmin()
        bestPf = pfs[minind]
        bestX = Xs[minind]
        startVal, stopVal = Xs[minind - 1], Xs[minind + 1]
        loops += 1
        if minind == 0 or minind == numintervals - 1:
            raise ValueError('Grid search failing, set threshold manually')
    return bestX, bestPf


def _verifyEvents(Dets, Autos, veriFile, veriBuffer, includeAllVeriColumns):
    if veriFile is None:
        return
    if isinstance(veriFile, string_types):
        if not veriFile or not os.path.exists(veriFile):
            msg = 'No veriFile passed or it does not exist, skipping verification'
            detex.log(__name__, msg, pri=True, level='warn')
            return
    elif not isinstance(veriFile, pd.DataFrame):
        msg = 'verifile type not supported, must be string or df'
        detex.log(__name__, msg, level='warn', pri=True)

    vertem = _readVeriFile(veriFile)
    tstmp = [obspy.UTCDateTime(x).timestamp for x in vertem['TIME']]
    vertem['STMP'] = tstmp
    verlist = []
    cols = ['TIME', 'LAT', 'LON', 'MAG', 'ProEnMag', 'DEPTH', 'NAME']
    additionalColumns = list(set(vertem.columns) - set(cols))

    for vernum, verrow in vertem.iterrows():
        con1 = Dets.MSTAMPmin - veriBuffer / 2.0 < verrow.STMP
        con2 = Dets.MSTAMPmax + veriBuffer / 2.0 > verrow.STMP
        con3 = [not x for x in Dets.Verified]
        temDets = Dets[(con1) & (con2) & (con3)]
        if len(temDets) > 0:  # TODO handle multiple verification situations
            trudet = temDets[temDets.DSav == temDets.DSav.max()]
            Dets.loc[trudet.index[0], 'Verified'] = True
            if includeAllVeriColumns:
                for col in additionalColumns:
                    if not col in trudet.columns:
                        trudet[col] = verrow[col]
            trudet['VerMag'] = verrow.MAG
            trudet['VerLat'] = verrow.LAT
            trudet['VerLon'] = verrow.LON
            trudet['VerDepth'] = verrow.DEPTH
            trudet['VerName'] = verrow.NAME
            verlist.append(trudet)
        else:
            con1 = Autos.MSTAMPmin - veriBuffer / 2.0 < verrow.STMP
            con2 = Autos.MSTAMPmax + veriBuffer / 2.0 > verrow.STMP
            con3 = [not x for x in Autos.Verified]
            temAutos = Autos[(con1) & (con2) & (con3)]
            if len(temAutos) > 0:  # TODO same as above
                trudet = temAutos[temAutos.DSav == temAutos.DSav.max()]
                Autos.loc[trudet.index[0], 'Verified'] = True
                if includeAllVeriColumns:
                    for col in additionalColumns:
                        if not col in trudet.columns:
                            trudet[col] = verrow[col]
                trudet['VerMag'] = verrow.MAG
                trudet['VerLat'] = verrow.LAT
                trudet['VerLon'] = verrow.LON
                trudet['VerDepth'] = verrow.DEPTH
                trudet['VerName'] = verrow.NAME
                verlist.append(trudet)
    if len(verlist) > 0:
        verifs = pd.concat(verlist, ignore_index=True)
        # sort and drop duplicates so each verify event is verified only
        # once
        verifs.sort_values(by=['Event', 'DSav'])
        verifs.drop_duplicates(subset='Event')
        verifs.drop('Verified', axis=1, inplace=True)
    else:
        verifs = pd.DataFrame()
    return verifs


def _readVeriFile(veriFile):
    try:
        df = pd.read_csv(veriFile)
    except Exception:
        try:
            df = pd.read_pickle(veriFile)
        except Exception:
            try:
                df = detex.util.loadSQLite(veriFile, 'verify')
            except Exception:
                msg = ('%s could not be read, it must either be csv, pickled'
                       'dataframe or sqlite database') % veriFile
                detex.log(__name__, msg, level='error')
    reqcols = ['TIME', 'LAT', 'LON', 'MAG', 'DEPTH', 'NAME']  # required cols
    if not set(reqcols).issubset(df.columns):
        msg = ('%s does not have the required columns, it needs '
               'TIME,LAT,LON,MAG,DEPTH,NAME') % veriFile
        detex.log(__name__, msg, level='error')
    return df


def _buildSQL(PfKey, trigCon, trigParameter, stations,
              starttime, endtime, tableName):
    """Function to build a list of SQL commands for loading the database 
    with desired parameters
    """
    SQL = []  # init. blank list
    if not starttime or not endtime:
        starttime = 0.0
        endtime = 4500 * 3600 * 24 * 365.25
    else:
        starttime = obspy.UTCDateTime(starttime).timestamp
        endtime = obspy.UTCDateTime(endtime).timestamp

    # define stations
    if isinstance(stations, (list, tuple)):
        if isinstance(PfKey, pd.DataFrame):
            PfKey = PfKey[PfKey.Sta.isin(stations)]
    else:
        if isinstance(PfKey, pd.DataFrame):
            stations = PfKey.Sta.values
        else:
            stations = ['*']  # no stations definition use all

    if isinstance(PfKey, pd.DataFrame):
        for num, row in PfKey.iterrows():
            # make sure appropraite table is used for either subspace or
            # singleton
            table = 'sg_df' if 'SG' in row.Name else 'ss_df'
            sqstr = ('SELECT %s FROM %s WHERE Sta="%s" AND Name="%s" AND  '
                     'DS>=%f AND MSTAMPmin>%f AND MSTAMPmin<%f') % ('*', table,
                                                                    row.Sta, row.Name, row.DS, starttime, endtime)
            SQL.append(sqstr)
    else:
        if trigCon == 0:
            cond = 'DS'
        elif trigCon == 1:
            cond = 'DS_STALTA'
        for sta in stations:
            if sta == '*':
                sqstr = ('SELECT %s FROM %s WHERE %s >= %s AND MSTAMPmin>=%f '
                         'AND MSTAMPmin<=%f') % ('*', tableName, cond,
                                                 trigParameter, starttime, endtime)
                SQL.append(sqstr)
            else:
                sqstr = ('SELECT %s FROM %s WHERE %s="%s" AND %s >= %s AND '
                         'MSTAMPmin>=%f AND MSTAMPmin<=%f') % ('*', tableName,
                                                               'Sta', sta, cond, trigParameter, starttime, endtime)
                SQL.append(sqstr)
    return SQL


def _deleteDetDups(ssDB, trigCon, trigParameter, associateBuffer, starttime,
                   endtime, stations, tableName, PfKey=None):
    """
    delete dections of same event, keep only detection with highest 
    detection statistic
    """
    sslist = []
    SQLstr = _buildSQL(PfKey, trigCon, trigParameter,
                       starttime, stations, endtime, tableName)
    for sql in SQLstr:
        loadedRes = detex.util.loadSQLite(ssDB, tableName, sql=sql)
        if isinstance(loadedRes, pd.DataFrame):
            sslist.append(loadedRes)
    if len(sslist) < 1:  # if no events found
        return None
    try:
        ssdf = pd.concat(sslist, ignore_index=True)
    except ValueError:
        msg = 'Cant create detResults instance, no detections meet all reqs'
        detex.log(__name__, msg, level='error')
    ssdf.reset_index(drop=True, inplace=True)
    ssdf.sort_values(by=['Sta', 'MSTAMPmin'], inplace=True)
    con1 = ((ssdf.MSTAMPmin - associateBuffer) > ssdf.MSTAMPmax.shift())
    con2 = ssdf.Sta != ssdf.Sta.shift()
    ssdf['Gnum'] = (con1 | con2).cumsum()
    ssdf.sort_values(by=['Gnum', 'DS'], inplace=True)
    ssdf.drop_duplicates(subset='Gnum', keep='last', inplace=True)
    ssdf.reset_index(inplace=True, drop=True)

    return ssdf


def _associateDetections(ssdf, associateReq, requiredNumStations,
                         associateBuffer, ss_info, temkey, exceptionalThreshold):
    """
    Associate detections together using pandas groupby return dataframe of 
    detections and autocorrelations
    """
    ssdf.sort_values(by='MSTAMPmin', inplace=True)
    ssdf.reset_index(drop=True, inplace=True)
    cols = ['Event', 'DSav', 'DSmax', 'NumStations', 'DS_STALTA', 'MSTAMPmin',
            'MSTAMPmax', 'Mag', 'ProEnMag', 'Verified', 'Dets']
    if isinstance(ss_info, pd.DataFrame) and associateReq > 0:
        ssdf = pd.merge(ssdf, ss_info, how='inner', on=['Sta', 'Name'])
    gs = (ssdf.MSTAMPmin - associateBuffer > ssdf.MSTAMPmax.shift()).cumsum()
    groups = ssdf.groupby(gs)
    autolist = [pd.DataFrame(columns=cols)]
    detlist = [pd.DataFrame(columns=cols)]
    temkey['STMP'] = np.array([obspy.core.UTCDateTime(x).timestamp
                               for x in temkey.TIME])
    temcop = temkey.copy()

    # if there is a required number of shared events
    if isinstance(ss_info, pd.DataFrame) and associateReq > 0:
        for num, g in groups:
            g = _checkSharedEvents(g)
            # Make sure detections occur on the required number of stations
            if len(set(g.Sta)) >= requiredNumStations:
                isauto, autoDF = _createAutoTable(g, temcop, cols)
                if isauto:
                    autolist.append(autoDF)
                else:
                    detdf = _createDetTable(g, cols)
                    detlist.append(detdf)
    else:
        for num, g in groups:
            # Make sure detections occur on the required number of stations
            con1 = len(set(g.Sta)) >= requiredNumStations
            if not con1 and isinstance(exceptionalThreshold, float):
                con2 = g.DS.max() >= exceptionalThreshold
                con1 = con1 or con2
            elif not con1 and isinstance(exceptionalThreshold, dict):
                con2 = _check_if_exceptional(g, exceptionalThreshold)
                con1 = con1 or con2
            if con1:
                # If there is more than one single or subpspace representing a
                # station on each event only keep the one with highest DS
                if len(set(g.Sta)) < len(g.Sta):
                    g = g.sort_values(by='DS').drop_duplicates(
                        subset='Sta', keep='last').sort_values('MSTAMPmin')
                isauto, autoDF = _createAutoTable(g, temcop, cols, associateBuffer)
                if isauto:
                    autolist.append(autoDF)
                    # temcop=temcop[temcop.NAME!=autoDF.iloc[0].Event]
                else:
                    detdf = _createDetTable(g, cols)
                    detlist.append(detdf)
    detTable = pd.concat(detlist, ignore_index=True)

    autoTable = pd.concat(autolist, ignore_index=True)
    return [detTable, autoTable]


def _check_if_exceptional(g, exth):
    gg = g.copy()
    gg['exceptional'] = [exth.get(x.Sta, 100) for _, x in gg.iterrows()]
    #    if any((gg['DS'] >= gg['exceptional']) & (gg['DS'] <= 1.01)) :
    return any((gg['DS'] >= gg['exceptional']) & (gg['DS'] <= 1.01))


# Look at the union of the events and delete those that do not meet the
# requirements
def _checkSharedEvents(g):
    pass  # TODO figure out how to incorporate an association requirement


def _createDetTable(g, cols):
    mag, proEnMag = _getMagnitudes(g)
    utc = obspy.UTCDateTime(np.mean([g.MSTAMPmin.mean(), g.MSTAMPmax.mean()]))
    event = str(utc).replace(':', '-').split('.')[0]
    data = [event, g.DS.mean(), g.DS.max(), len(g), g.DS_STALTA.mean(),
            g.MSTAMPmin.min(), g.MSTAMPmax.max(), mag, proEnMag, False, g]
    detDF = pd.DataFrame([data], columns=cols)
    return detDF


def _createAutoTable(g, temkey, cols, associateBuffer):
    isauto = False
    for num, row in g.iterrows():  # find out if this is an auto detection
        con1 = temkey.STMP + associateBuffer > row.MSTAMPmin
        con2 = temkey.STMP - associateBuffer < row.MSTAMPmax
        temtemkey = temkey[con1 & con2]
        if len(temtemkey) > 0:
            isauto = True
            event = temtemkey.iloc[0].NAME
    if isauto:
        mag, proEnMag = _getMagnitudes(g)
        data = [event, g.DS.mean(), g.DS.max(), len(g), g.DS_STALTA.mean(),
                g.MSTAMPmin.min(), g.MSTAMPmax.max(), mag, proEnMag, False, g]
        autoDF = pd.DataFrame([data], columns=cols)
        return isauto, autoDF
    else:
        return isauto, pd.DataFrame()


def _getMagnitudes(g):
    if any([not np.isnan(x) for x in g.Mag]):
        mag = np.nanmedian(g.Mag)
    else:
        mag = np.NaN
    if any([not np.isnan(x) for x in g.ProEnMag]):
        PEmag = np.nanmedian(g.ProEnMag)
    else:
        PEmag = np.NaN
    return mag, PEmag


def _loadSSdb(ssDB, trigCon, trigParameter, sta=None):
    """
    Load a subspace database
    """
    if trigCon == 0:
        cond = 'DS'
    elif trigCon == 1:
        cond = 'DS_STALTA'
    if sta:
        sql = 'SELECT %s FROM %s WHERE %s="%s" AND %s > %s' % (
            '*', 'ss_df', 'Sta', sta, cond, trigParameter)
    else:
        sql = 'SELECT %s FROM %s WHERE %s > %s' % (
            '*', 'ss_df', cond, trigParameter)
    # sys.exit(1)
    df = detex.util.loadSQLite(ssDB, 'ss_df', sql=sql)
    return df


def _checkInputs(trigCon, trigParameter, associateReq,
                 associateBuffer, requiredNumStations):
    if not isinstance(trigCon, int) or not (trigCon == 0 or trigCon == 1):
        msg = 'trigcon must be an int, either 0 or 1'
        detex.log(__name__, msg, level='error')
    if trigCon == 0:
        con1 = isinstance(trigParameter, numbers.Real)
        con2 = trigParameter > 1
        con3 = trigParameter < 0
        if not con1 or con2 or con3:
            msg = ('When trigCon==0 trigParameter is the required detection '
                   'statistic and therefore must be between 0 and 1')
            detex.log(__name__, msg, level='error')
    elif trigCon == 1:
        # allow 0 to simply select all
        con1 = isinstance(trigParameter, numbers.Real)
        con2 = trigParameter < 1
        con3 = trigParameter != 0
        if not con1 or (con2 and con3):
            msg = ('When trigCon==1 trigParameter is the STA/LTA of the '
                   'detection statistic vector and therefore must be greater than 1')
            detex.log(__name__, msg, level='error')
    if not isinstance(associateReq, int) or associateReq < 0:
        msg = ('AssociateReq is the required number of events a subspace must '
               'share for detections from different stations to be associated '
               'together and therefore must be an integer 0 or greater')
        detex.log(__name__, msg, level='error')
    if not isinstance(associateBuffer, numbers.Real) or associateBuffer < 0:
        msg = 'associateBuffer must be a real number greater than 0'
        detex.log(__name__, msg, level='error')
    if not isinstance(requiredNumStations, int) or requiredNumStations < 1:
        msg = 'requiredNumStations must be an integer greater than 0'
        detex.log(__name__, msg, level='error')


def _checkExistence(existList):
    for fil in existList:
        if not os.path.exists(fil):
            raise IOError('%s does not exists' % fil)


def _loadInfoDataFrames(ssDB):
    ss_info = detex.util.loadSQLite(ssDB, 'ss_info')  # load subspace info
    if isinstance(ss_info, pd.DataFrame):
        ss_info['NumEvents'] = [len(row.Events.split(','))
                                for num, row in ss_info.iterrows()]
    sg_info = detex.util.loadSQLite(ssDB, 'sg_info')
    if isinstance(sg_info, pd.DataFrame):
        sg_info['NumEvents'] = 1
    return ss_info, sg_info





class SSResults(object):
    def __init__(self, Dets, Autos, Vers, ss_info, ss_filt,
                 temkey, stakey, templateKey, fetcher):
        DF = pd.DataFrame
        self.Autos = Autos
        self.Dets = Dets
        self.NumVerified = len(Vers) if isinstance(Vers, DF) else 'N/A'
        self.Vers = Vers
        self.info = ss_info
        self.filt = ss_filt
        self.StationKey = stakey
        self.TemplateKey = temkey
        self.TemKeyPath = templateKey
        self.fetcher = fetcher

    def writeDetections(self, onlyVerified=None, minDS=None, minMag=None,
                        eventDir='EventWaveForms', updateTemKey=True,
                        temkeyPath=None, timeBeforeOrigin=1 * 60,
                        timeAfterOrigin=4 * 60, waveFormat="mseed",
                        pick_times='PickTimes.csv',**kwargs):
        """
        Function to make all of the eligable new detections templates. New 
        event directories will be added to eventDir and the template key 
        will be updated with detected events having a lower case "d" 
        before the name

        Parameters
        ----------
        onlyVerified : boolean
            If true only use detections that are verified
        minDS : None or float between 0.0 and 1.0
            If float only use detections with average detection statistics 
            above minDSave
        minMag : false or float
            If float only use detections with estimated magnitudes above minMag
        eventDir : None or str
            If None new waveforms of detections are stored in default 
            event directory (usually EventWaveForms).
            If str then the str must be path to new directory in which detected
            event waveforms will be stored. If it does not exist it will be 
            created
        updateTemKey : boolean
            If true update the template key with the new detections
        temkeyPath : None or str
            if None use the default path to the template key, else new path 
            to templatekey, if it does not exist it will be created
        timeBeforeOrigin : real number (float or int)
            Seconds before predicted origin to get (default is the same as 
            the getData.getAllData defaults)
        timeAfterOrigin : real number (float or int)
            Seconds after predicted origin to get
        waveFormat = str
            Format to save files, "mseed", "pickle", "sac", and "Q" supported
        pick_times : str or df
            Path to the pick times file, used for calculating predicted origin
            times, can also pass loaded picktimes dataframe
        """
        dets = self.Dets.copy()
        if onlyVerified:
            dets = dets[dets.Verified]
        if minDS:
            dets = dets[dets.minDS >= minDS]
        if minMag:
            dets = dets[dets.Mag >= minMag]
        if eventDir is None:
            eventDir = self.eventDir
        if temkeyPath is None:
            temkeyPath = self.TemKeyPath
        if isinstance(pick_times, string_types):
            assert os.path.exists(pick_times)
            pick_times = pd.read_csv(pick_times)

        temkey = self.TemplateKey.copy()
        detTem = pd.DataFrame(index=range(len(dets)), columns=temkey.columns)

        for num, row in dets.iterrows():  # loop through detections and save
            origin = self._predict_origin(row, pick_times)
            origin = obspy.UTCDateTime(np.mean([row.MSTAMPmax, row.MSTAMPmin]))
            Evename = row.Event
            eveDirName = 'd' + Evename

            # if the directory doesnt exists create it
            if not os.path.exists(os.path.join(eventDir, eveDirName)):
                os.makedirs(os.path.join(eventDir, eveDirName))
            else:  # else delete its index so it will be reindexed
                index_path = os.path.join(eventDir, 'index.db')
                if os.path.exists(index_path):
                    os.remove(index_path)

            # loop through each station and load stream, then save
            for stanum, starow in self.StationKey.iterrows():
                net, sta = starow.NETWORK, starow.STATION
                start = origin - timeBeforeOrigin
                stop = origin + timeAfterOrigin
                ext = detex.getdata.formatKey[waveFormat]
                fname = '.'.join([net, sta, Evename, ext])
                path = os.path.join(eventDir, eveDirName, fname)
                try:
                    st = self.fetcher.getStream(start, stop, net, sta)
                    st.write(path, waveFormat)
                except Exception:
                    msg = ('Could not write and save %s for station %s' % (
                        Evename, sta))
                    detex.log(__name__, msg, level='warning', pri=True)

            detTem.loc[num, 'NAME'] = eveDirName
            time = str(obspy.UTCDateTime(origin.timestamp))
            detTem.loc[num, 'TIME'] = time.replace(':', '-').replace('Z', '')
            detTem.loc[num, 'MAG'] = row.Mag

        temkeyNew = pd.concat([temkey, detTem], ignore_index=True)
        temkeyNew.reset_index(inplace=True, drop=True)
        temkeyNew.to_csv(temkeyPath, index=False)

    def visualize(self, fetcher_arg='ContinuousWaveForms',
                  detype='dets', time_before=10, time_after=60):
        """
        Call the stream picks gui to examin detections one at a time. Make
        A p pick anywhere to mark the detection as accepted, and make an S
        pick anywhere to mark the detection as rejected

        Parameters
        ----------
        fetcher_arg : str, DataFetcher, or client
            The argument used to init the datafetcher
        detype: str (Det, Auto, Vers)
            The type of detection (Det = new detection, Auto = autodetection,
            vers = verified detections)
        time_before : float or int
            The time before the (predicted) origin time to fetch
        time_after : float or int
            The time after the (predicted) origin time to fetch
        Returns
        -------
        None, GUI called in place
        """
        # init qt app
        qApp = PyQt4.QtGui.QApplication(sys.argv)
        # dict to map detection_type to correct dfs, or use df if passed
        df = self._get_detype(detype)
        if not 'Review' in df.columns:
            df['Review'] = 0
        # init fetcher
        fetcher = detex.getdata.quickFetch(fetcher_arg)
        # iter df and plot row one at a time
        for st, ind in _get_streams(fetcher, df, time_before, time_after):
            pks = detex.streamPick.streamPick(st, ap=qApp)
            # mark rejected or accepted if pick found
            picks = pks._picks
            if picks and 'S' in picks[0].phase_hint:
                df.loc[ind, 'Review'] = -1
            elif picks and 'P' in picks[0].phase_hint:
                df.loc[ind, 'Review'] = 1
            if not pks.KeepGoing:
                break

    def yield_streams(self, fetcher_arg='ContinuousWaveForms',
                      detype='dets', time_before=10, time_after=60,
                      yield_ind=True):
        """
        Return an iterator of streams containing each detection

        Parameters
        ----------
        fetcher_arg : str, DataFetcher, or client
            The argument used to init the datafetcher
        detype : str or df
            The type of detections to use, options are: 'autos', 'vers',
            'dets'. Can also pass a custom df
        time_before : int or float
            The time before the predicted origin time to pull
        time_after : int or float
            The time after the predicted origin time to pull
        yield_ind : bool
            If true yield the index of the dataframe with the stream
        Returns
        -------
        An iterator of obspy streams for each detection
        """
        df = self._get_detype(detype)
        fetcher = detex.getdata.quickFetch(fetcher_arg)
        for st, ind in _get_streams(fetcher, df, time_before, time_after):
            if yield_ind:
                yield  st, ind
            else:
                yield st

    def _get_detype(self, detype):
        """ get the correct dataframe based on detype argument """
        if isinstance(detype, pd.DataFrame):
            df = detype
        else:
            det_dic = dict(dets=self.Dets, autos=self.Autos, vers=self.Vers)
            df = det_dic[detype.lower()]
        return df

    def __repr__(self):
        lens = (len(self.Autos), len(self.Dets), str(self.NumVerified))
        outstr = ('SSResults instance with %d autodections and %d new '
                  'detections, %s are verified') % lens
        return outstr

    def _predict_origin(self, row, pick_times):
        """
        Predict the pick times based on the best correlated event
        on each station
        """
        pdb.set_trace()
        pass


def _get_streams(fetcher, df, time_before, time_after):
    """ function to get streams from a results df, yield for each row """
    # start, end, net, sta,
    stmp_start = df.MSTAMPmin - time_before
    stmp_end = df.MSTAMPmin + time_after
    indicies = df.index
    # iter each detection and yield
    for ind, start, end in zip(indicies, stmp_start, stmp_end):
        if df.loc[ind].Review != 0:
            continue
        st = obspy.Stream()
        dets = df.loc[ind].Dets
        for net, sta in list(dets.Sta.str.split('.')):
            st1 = fetcher.getStream(start, end, net, sta)
            if st1 is not None and len(st1):
                st += st1
        yield st, ind
