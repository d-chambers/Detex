# -*- coding: utf-8 -*-
"""
Created on Thu May 29 16:41:48 2014

@author: Derrick
"""
import numpy as np
import obspy
import os
import detex.pandas_dbms
import pandas.io.sql as psql
import simplekml
import pandas as pd
import detex

from sqlite3 import PARSE_DECLTYPES, connect


######################### KML functions ###############################

def writeKMLFromDF(DF, outname='map.kml'):
    """
    Write a KML file from a pandas data frame with the same format as
    readSum output
    """
    kml = simplekml.Kml(open=1)
    for a in DF.iterrows():
        pnt = kml.newpoint()
        pnt.name = str(a[1].DateString)
        pnt.coords = [(a[1].Lon, a[1].Lat)]
    kml.save(outname)


def writeKMLFromTemplateKey(df='TemplateKey.csv', outname='templates.kml'):
    """
    Write a KML file from a templateKey

    Parameters
    -------------
    DF : str or pandas Dataframe
        If str then the path to the template key csv. If dataframe then 
        template key loaded with read_key function with key_type='template'
    outname : str
        path of the kml file
    """
    if isinstance(df, str):
        df = pd.read_csv(df)
    elif not isinstance(df, pd.DataFrame):
        msg = ('Input type not understood, must be path to template key or '
               'dataframe of templatekey')
        detex.log(__name__, msg, level='error')

    kml = simplekml.Kml(open=1)
    for a in df.iterrows():
        pnt = kml.newpoint()
        pnt.name = str(a[1].NAME)
        pnt.coords = [(a[1].LON, a[1].LAT)]
    kml.save(outname)


def writeKMLFromStationKey(df='StationKey.csv', outname='stations.kml'):
    """
    Write a KML file from a tempalteKey

    Parameters
    -------------
    DF : str or pandas Dataframe
        If str then the path to the station key. If dataframe then
        station key loaded witj read_key function with key_type='template'
    outname : str
        name of the kml file
    """
    if isinstance(df, str):
        df = pd.read_csv(df)
    elif not isinstance(df, pd.DataFrame):
        msg = ('Input type not understood, must be path to station key or '
               'dataframe of station key')
        detex.log(__name__, msg, level='error')
    kml = simplekml.Kml(open=1)
    for a in df.iterrows():
        pnt = kml.newpoint()
        pnt.name = str(a[1].STATION)
        pnt.coords = [(a[1].LON, a[1].LAT)]
        # print(a[1].STATION,a[1].LON,a[1].LAT)
    kml.save(outname)


def writeKMLFromHypInv(hypout='sum2000', outname='hypoInv.kml'):
    """
    Uses simplekml to create a KML file (used by Google Earth, Google Maps,
    etc) of the results from hypoInverse 2000
    """
    C = []
    with open(hypout, 'r') as openfile:
        for line in openfile:
            C = C + [line[0:31]]
    kml = simplekml.Kml(open=1)
    for a in C:
        spl = a.replace(' ', '0')
        lat = float(spl[16:18]) + (float(spl[19:21]) /
                                   60 + float(spl[21:23]) / (100.0 * 60))
        # assume negative sign needs to be added for west
        lon = -float(spl[23:26]) + -(float(spl[27:29]) /
                                     60.0 + float(spl[29:31]) / (100.0 * 60))
        pnt = kml.newpoint()
        pnt.name = str(int(a[0:10]))
        pnt.coords = [(lon, lat)]
    kml.save(outname)


def writeKMLFromArcDF(df, outname='Arc.kml'):
    kml = simplekml.Kml(open=1)
    for a in df.iterrows():
        pnt = kml.newpoint()
        pnt.name = str(int(a[0]))
        pnt.coords = [(a[1]['verlon'], a[1]['verlat'])]
    kml.save(outname)

def writeKMLfromHYPInput(hypin='test.pha', outname='hypoInInv.kml'):
    with open(hypin, 'rb') as infile:
        kml = simplekml.Kml(open=1)
        cou = 1
        for line in infile:
            if line[0:6] != '      ' and len(line) > 10:
                pass
            elif line[0:6] == '      ':
                # print 'terminator line'
                lat = float(line[14:16]) + (float(line[17:19]) / \
                            60 + float(line[19:21]) / (100.0 * 60))
                lon = -float(line[21:24]) + -(float(line[25:27]) / \
                             60.0 + float(line[27:29]) / (100.0 * 60))
                pnt = kml.newpoint()
                pnt.name = str(cou)
                pnt.coords = [(lon, lat)]
                cou += 1
        kml.save(outname)

############## HypoDD, HypoInverse and NonLinLoc Functions ################

def writeKMLFromHypDD(hypreloc='hypoDD.reloc', outname='hypo.kml'):
    """
    Uses simplekml to create a KML file (used by Google Earth, 
    Google Maps, etc) of the results from hypoDD
    """
    points = np.array(np.genfromtxt(hypreloc))
    kml = simplekml.Kml(open=1)
    for a in points:
        pnt = kml.newpoint()
        pnt.name = str(int(a[0]))
        pnt.coords = [(a[2], a[1])]
    kml.save(outname)

def writeHypoDDStationInput(stakey, fileName='station.dat', useElevations=True,
                            inFt=False):
    """
    Write the station input file for hypoDD (station.dat)

    Parameters
    ---------
    stakey : str or DataFrame
        Path to station key or instance of DataFrame with station key loaded
    fileName : str
        Path to the output file
    useElevations : boolean
        If true also print elevations
    inFt : boolean
        If true elevations in station key are in ft, convert to meters
    """
    if isinstance(stakey, str):
        stakey = read_key(stakey, key_type='station')
    fil = open(fileName, 'wb')
    conFact = 0.3048 if inFt else 1  # ft to meters if needed
    for num, row in stakey.iterrows():
        line = '%s %.6f %.6f' % (
            row.NETWORK + '.' + row.STATION, row.LAT, row.LON)
        if useElevations:
            line = line + ' %.2f' % row.ELEVATION * conFact
        fil.write(line + '\n')
    fil.close()

def writeHypoDDEventInput(temkey, fileName='event.dat'):
    """
    Write a hypoDD event file (event.dat)
    Parameters
    ----------
    temkey : str or pandas DataFrame
        If str then path to template key, else loaded template key
    """
    if isinstance(temkey, str):
        temkey = read_key(temkey, key_type='template')
    fil = open(fileName, 'wb')
    reqZeros = int(np.ceil(np.log10(len(temkey))))
    fomatstr = '{:0' + "{:d}".format(reqZeros) + 'd}'
    for num, row in temkey.iterrows():
        utc = obspy.UTCDateTime(row.TIME)
        DATE = '%04d%02d%02d' % (
            int(utc.year), int(utc.month), int(utc.day))
        TIME = '%02d%02d%04d' % (int(utc.hour), int(
            utc.minute), int(utc.second * 100))
        mag = row.MAG if row.MAG > -20 else 0.0
        ID = fomatstr.format(num)
        linea = (DATE + ', ' + TIME + ', ' + '{:04f}, '.format(row.LAT) + 
                '{:04f}, '.format(row.LON) + '{:02f}, '.format(row.DEPTH))
        lineb = '{:02f}, '.format(mag) + '0.0, 0.0, 0.0, ' + ID
        fil.write(linea + lineb + '\n')
    fil.close()

def writeKMLFromEQSearchSum(eqsum='eqsrchsum', outname='eqsearch.kml'):
    """
    Write a KML from the eqsearch sum file (produced by the University 
    of Utah seismograph stations code EQsearch)

    Parameters
    -------------
    eqsum : str
        eqsearch sum. file
    outname : str
        name of the kml file
    Notes 
    -------------
    Code assimes any year code above 50 belongs to 1900, and any year code
    less than 50 belongs to 2000 (since eqsrchsum is not y2k compliant)
    """
    clspecs = [(0, 2), (2, 4), (4, 6), (7, 9), (9, 11), (12, 17),
               (18, 20), (21, 26), (27, 30), (31, 36), (37, 43), (45, 50)]
    names = ['year', 'mo', 'day', 'hr', 'min', 'sec', 'latdeg', 'latmin',
             'londeg', 'lonmin', 'dep', 'mag']
    df = pd.read_fwf(eqsum, colspecs=clspecs, header=None, names=names)
    year = ['19%02d' % x if x > 50 else '20%02d' % x for x in df['year']]
    month = ['%02d' % x for x in df['mo']]
    day = ['%02d' % x for x in df['day']]
    hr = ['%02d' % x for x in df['hr']]
    minute = ['%02d' % x for x in df['min']]
    second = ['%05.02f' % x for x in df['sec']]
    TIME = ['%s-%s-%sT%s-%s-%s' % (x1, x2, x3, x4, x5, x6) 
            for x1, x2, x3, x4, x5, x6 in zip(
            year, month, day, hr, minute, second)]
    Lat = df['latdeg'].values + df['latmin'].values / 60.0
    Lon = -df['londeg'].values - df['lonmin'].values / 60.0

    kml = simplekml.Kml(open=1)
    for T, Lat, Lon in zip(TIME, Lat, Lon):
        pnt = kml.newpoint()
        pnt.name = str(T)
        pnt.coords = [(Lon, Lat)]
    kml.save(outname)

def writeHypoFromDict(TTdict, phase='P', output='all.phases'):
    """ Function to write a hyp phase input file based on dictionary or 
    list of dictionaries where station name are keys and the timestamps 
    are values
    """
    if isinstance(TTdict, dict):
        TTdict = [TTdict]  # make iterable
    if isinstance(TTdict, pd.core.series.Series):
        TTdict = TTdict.tolist()
    if not isinstance(TTdict, list):
        msg = ('TTdict type not understood, must be python dictionary or '
               'list of dictionaries')
        detex.log(__name__, msg, level='error')
    with open(output, 'wb') as out:
        out.write('\n')  # write intial end character
        for a in TTdict:
            if len(a) > 3:
                for key in a.keys():
                    line = _makeSHypStationLine(key, 'ZENZ', 'TA', a[key], 'P')
                    out.write(line)
                termline = _makeHypTermLine(a)
                out.write(termline)
                out.write('\n')


def _makeHypTermLine(TTdict):
    mintime = obspy.core.UTCDateTime(np.min(np.array(TTdict.values())))
    space = ' '
    hhmmssss = mintime.formatIRISWebService().replace(
        '-', '').replace('T', '').replace(':', '').replace('.', '')[8:16]
    lat, latminute, lon, lonminute = ' ', ' ', ' ', ' '
    trialdepth = '  400'
    endline = "{:<6}{:<8}{:<3}{:<4}{:<4}{:<4}{:<5}\n".format(
        space, hhmmssss, lat, latminute, lon, lonminute, trialdepth)
    return endline


def _makeSHypStationLine(sta, cha, net, ts, pha):
    Ptime = obspy.core.UTCDateTime(ts)
    dstr = Ptime.formatIRISWebService().replace('-','').replace('T','')
    dstr = dstr.replace(':', '').replace('.', '')
    YYYYMMDDHHMM = dstr[0:12]
    ssss = dstr[12:16]
    end = '01'
    ty = ' %s 0' % pha
    line = "{:<5}{:<3}{:<5}{:<3}{:<13}{:<80}{:<2}\n".format(
        sta, net, cha, ty, YYYYMMDDHHMM, ssss, end)
    return line
        
def readSum(sumfile):
    """
    read a sum file from hyp2000 and return DataFrame with info loaded

    Parameters
    Read a sum file from hyp2000 and return lat, long, depth, mag, and RMS and
    TSTMP as pd dataframe
    WARNING : Assumes western hemisphere
    """
    lines = [line.rstrip('\n') for line in open(sumfile)]
    cols = ['Lat', 'Lon', 'DateString', 'Dep', 'RMS', 'ELAz', 'HozError',
            'VertError']
    DF = pd.DataFrame(index=range(len(lines)), columns=cols)

    for a, l in enumerate(lines):
        DF.Lat[a] = (float(l[16:18]) + (float(l[19:21].replace(' ', '0')) +
                     float(l[21:23].replace(' ', '0'))/100)/60)

        DF.Lon[a] = (-float(l[23:26]) - (float(l[27:29].replace(' ', '0')) +
                     float(l[29:31].replace(' ', '0')) / 100) / 60)

        DF.DateString[a] = (l[0:4] + '-' + l[4:6] + '-' + l[6:8] + 'T' + 
                            l[8:10] + '-' + l[10:12] + '-' + l[12:14] +
                            '.' + l[14:16])

        DF.Dep[a] = (float(l[31:34].replace(' ', '0').replace(
            '-', '0')) + float(l[34:36].replace(' ', '0')) / 100)
            
        DF.RMS[a] = (float(l[48:50].replace(' ', '0')) +
            float(l[50:52].replace(' ', '0')) / 100)
            
        DF.HozError[a] = (float(l[85:87].replace(' ', '0')) +
            float(l[87:89].replace(' ', '0')) / 100.0)
            
        DF.VertError[a] = (float(l[89:91].replace(' ', '0')) + 
            float(l[91:93].replace(' ', '0')) / 100.0)
    return DF        
        
################# Read/Create detex key files ###################




def read_key(dfkey, key_type='template'):
    """
    Read a template key csv and perform checks for required columns
    Parameters
    ---------
    dfkey : str or pandas DataFrame
        A path to the template key csv or the DataFrame itself
    key_type : str
        "template" for template key or "station" for station key
    Returns
    --------
    A pandas DataFrame if required columns exist, else raise Exception

    """
    # key types and required columns
    req_temkey = set(['TIME', 'NAME', 'LAT', 'LON', 'MAG', 'DEPTH'])
    req_stakey = set(['NETWORK', 'STATION', 'STARTTIME', 'ENDTIME', 'LAT', 
                      'LON', 'ELEVATION', 'CHANNELS'])
    key_types = ['template', 'station']
    req_columns = {'template': req_temkey, 'station': req_stakey}
    
    if key_type not in key_types:
        msg = "unsported key type, supported types are %s" % (key_types)
        detex.log(__name__, msg, level='error')

    if isinstance(dfkey, str):
        if not os.path.exists(dfkey):
            msg = '%s does not exists, check path' % dfkey
            detex.log(__name__, msg, level='error')
        else:
            df = pd.read_csv(dfkey)
    elif isinstance(dfkey, pd.DataFrame):
        df = dfkey
    else:
        msg = 'Data type of dfkey not understood'
        detex.log(__name__, msg, level='error')

    # Check required columns
    if not req_columns[key_type].issubset(df.columns):
        msg = ('Required columns not in %s, required columns for %s key are %s'
               % (df, key_type, req_columns))
        detex.log(__name__, msg, level='error')

    tdf = df.loc[:, list(req_columns[key_type])]
    condition = [all([x != '' for item, x in row.iteritems()])
                 for num, row in tdf.iterrows()]
    df = df[condition]
    df.sort(columns=list(req_columns[key_type]), inplace=True)
    df.reset_index(drop=True, inplace=True)

    # specific operations for various key types
    if key_type == 'station':
        df['STATION'] = [str(x) for x in df['STATION']]
        df['NETWORK'] = [str(x) for x in df['NETWORK']]
    return df
    
    
def writeTemplateKeyFromEQSearchSum(eq='eqsrchsum', oname='eqTemplateKey.csv'):
    """
    Write a template key from the eqsearch sum file (produced by the 
    University of Utah seismograph stations code EQsearch)

    Parameters
    -------------
    eq : str
        eqsearch sum. file
    oname : str
        name of the template key
    Notes 
    -------------
    Code assimes any year code above 50 belongs to 1900, and any year code
    less than 50 belongs to 2000 (since eqsrchsum is not y2k compliant)
   """
    clspecs = [(0, 2), (2, 4), (4, 6), (7, 9), (9, 11), (12, 17),
               (18, 20), (21, 26), (27, 30), (31, 36), (37, 43), (45, 50)]
    names = ['year', 'mo', 'day', 'hr', 'min', 'sec', 'latdeg', 'latmin',
             'londeg', 'lonmin', 'dep', 'mag']
    df = pd.read_fwf(eq, colspecs=clspecs, header=None, names=names)
    year = ['19%02d' % x if x > 50 else '20%02d' % x for x in df['year']]
    month = ['%02d' % x for x in df['mo']]
    day = ['%02d' % x for x in df['day']]
    hr = ['%02d' % x for x in df['hr']]
    minute = ['%02d' % x for x in df['min']]
    second = ['%05.02f' % x for x in df['sec']]
    TIME = [
        '%s-%s-%sT%s-%s-%s' %
        (x1, x2, x3, x4, x5, x6) for x1, x2, x3, x4, x5, x6 in zip(
            year, month, day, hr, minute, second)]
    Lat = df['latdeg'].values + df['latmin'].values / 60.0
    Lon = -df['londeg'].values - df['lonmin'].values / 60.0

    DF = pd.DataFrame()
    DF['TIME'] = TIME
    DF['NAME'] = TIME
    DF['LAT'] = Lat
    DF['LON'] = Lon
    DF['MAG'] = df['mag']
    DF['DEPTH'] = df['dep']
    DF.to_csv(oname)





def saveSQLite(DF, CorDB, Tablename, silent=True):  
    """
    Basic function to save pandas dataframe to SQL
    
    Parameters 
    -------------
    DF : pandas DataFrame
        The data frame instance to save
    CorDB : str
        Path to the database
    Tablename : str
        Name of the table to which DF will be saved
    silent : bool 
        If True will suppress the any messages from database writing
    """

    with connect(CorDB, detect_types=PARSE_DECLTYPES) as conn:
        if os.path.exists(CorDB):
            detex.pandas_dbms.write_frame(
                DF, Tablename, con=conn, flavor='sqlite', if_exists='append')
        else:
            detex.pandas_dbms.write_frame(
                DF, Tablename, con=conn, flavor='sqlite', if_exists='fail')

def loadSQLite(corDB, tableName, sql=None, readExcpetion=False, silent=True,
               convertNumeric=True):
    """
    Function to load sqlite database

    Parameters
    ----------
    corDB : str
        Path to the database
    tablename : str
        Table to load from sqlite database
    sql : str
        sql arguments to pass directly to database query

    Returns
    -------
    A pandas dataframe with loaded table or None if DB or table does not exist
    """
    try:
        if sql is None:
            sql = 'SELECT %s FROM %s' % ('*', tableName)
        with connect(corDB, detect_types=PARSE_DECLTYPES) as con:
            df = psql.read_sql(sql, con)
            if convertNumeric:
                df = df.convert_objects(
                    convert_dates=False, convert_numeric=True)
    except:
        msg = 'failed to load %s in %s with sql=%s' % (corDB, tableName, sql)
        detex.log(__name__, msg, level='warn', pri=not silent)
        df = None
        if readExcpetion:
            raise Exception
    return df

def loadClusters(filename='clust.pkl'): 
    """
    Function that uses pandas.read_pickle to load a pickled cluster
    (instance of detex.subspace.ClusterStream)
    Parameters
    ----------
    filename : str
        Path to the saved cluster isntance
    Returns
    ----------
    An instance of detex.subspace.ClusterStream 
    """
    cl = pd.read_pickle(filename)
    if not isinstance(cl, detex.subspace.ClusterStream):
        msg = '%s is not a ClusterStream instance' % filename
        detex.log(__name__, msg, level='error')
    return cl
    
def loadSubSpace(filename='subspace.pkl'): 
    """
    Function that uses pandas.read_pickle to load a pickled subspace
    (instance of detex.subspace.SubSpaceStream)
    Parameters
    ----------
    filename : str
        Path to the saved subspace instance
    Returns
    ----------
    An instance of detex.subspace.SubSpaceStream 
    """
    ss = pd.read_pickle(filename)
    if not isinstance(ss, detex.subspace.SubSpaceStream):
        msg = '%s is not a SubSpaceStream instance' % filename
        detex.log(__name__, msg, level='error')
    return ss

###################### Data processing functions ###########################

def get_number_channels(st):
    """
    Take an obspy stream and get the number of unique channels in stream
    (stream must have only one station)
    """
    if len(set([x.stats.station for x in st])) > 1:
        msg = 'function only takes streams with exactly 1 station'
        detex.log(__name__, msg, level='error')
    nc = len(list(set([x.stats.channel for x in st])))
    return nc

############################## Phase Picker ############################

def pickPhases(fetch='EventWaveForms', templatekey='TemplateKey.csv', 
               stationkey='StationKey.csv', pickFile='PhasePicks.csv'):
    """
    Uses streamPicks to parse the templates and allow user to manually pick
    phases for events. Only P,S, Pend, and Send are supported phases under 
    the current GUI, but other phases can be manually input to this format.

    Parameters
    -------------
    EveDir : str
        Input to detex.getdata.quickFetch, defaults to using the default 
        directory structure for 
    templatekey : str or pandas DataFrame
        Path to the template key or template key loaded in DataFrame
    stationkey : str or pandas DataFrame
        Path to the station key or station key loaded in DataFrame
    pickFile : str
        Path to newly created csv containing phase times. If the file already
        exists it will be read so that picks already made do not have to be
        re-picked
        
    Notes
    ----------
    Required columns are : TimeStamp, Station, Event, Phase
    Station field is net.sta (eg TA.M17A)
    """
    if isinstance(templatekey, str):
        temkey = pd.read_csv(templatekey)
    if isinstance(stationkey, str):
        stakey = pd.read_csv(stationkey)

    cols = ['TimeStamp', 'Station', 'Event', 'Phase']
    fetcher = detex.getdata.quickFetch(fetch)
    
    ets = {} # events to skip picking on

    # load pickfile if it exists
    if os.path.exists(pickFile):  
        DF = pd.read_csv(pickFile)
        if len(DF) < 1:  # if empty then delete
            os.remove(pickFile)
            DF = pd.DataFrame(columns=cols)
        else:
            for ind, row in DF.iterrows():
                if not row.Station in ets:
                    ets[row.Station] = []
                ets[row.Station].append(row.Event)
    else:
        DF = pd.DataFrame(columns=cols)
    #detex.deb(temkey)
    for st, event in fetcher.getTemData(temkey, stakey, skipDict=ets):
        reload(detex.streamPick)
        # print event, st[0].stats.station
        Pks = None  # needed so OS X doesn't crash
        Pks = detex.streamPick.streamPick(st)
        tdict = {}
        saveit = 0  # saveflag
        for b in Pks._picks:
            if b:
                tdict[b.phase_hint] = b.time.timestamp
                saveit = 1
        if saveit:
            for key in tdict.keys():
                stmp = tdict[key]
                sta = str(st[0].stats.network + '.' + st[0].stats.station)
                di = {'TimeStamp': stmp, 'Station': sta, 'Event': event, 
                      'Phase': key}
                DF = DF.append(pd.Series(di), ignore_index=True)
        if not Pks.KeepGoing:
            msg = 'Exiting picking GUI, progress saved in %s' % pickFile
            detex.log(__name__, msg, level='info', pri=True)
            DF.sort(columns=['Station', 'Event'], inplace=True)
            DF.reset_index(drop=True, inplace=True)
            DF.to_csv(pickFile, index=False)
            return
    DF.sort(columns=['Station', 'Event'], inplace=True)
    DF.reset_index(drop=True, inplace=True)
    DF.to_csv(pickFile, index=False)
