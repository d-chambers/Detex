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
import sip
import time
import PyQt4
import sys

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
        template key loaded with readKey function with key_type='template'
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
    Write a KML file from a station key

    Parameters
    -------------
    DF : str or pandas Dataframe
        If str then the path to the station key. If dataframe then
        station key loaded with readKey function with key_type='template'
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

############## HypoDD Functions ################

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
        stakey = readKey(stakey, key_type='station')
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
        temkey = readKey(temkey, key_type='template')
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

############## Hypoinverse Functions ################
def makeHypoInversePhaseFile(phases, evekey, outname, fix=0, usePhases=['P'],
                  fixFirstStation=False):
    """ 
    Write a hypoinverse phase file used by hypoinverse, format defined
    on page 113 of the manual for version 1.39, using the phase file created
    by detex.util.pickPhases. Format for this file is free csv with the 
    following fields: TimeStamp (epoch time in seconds), Station (net.sta),
    Event (unique event id, usually datestring, must be the same in 
    templatekey), Phase (phase identifier, generally only P or S)
    
    Parameters
    ---------
    phases : pandas dataframe or path to csv
        Phase input from AssociatePhases script
    evekey : pandas dataframe or csv
        Event info
    outname : str
        File name (path) to write
    fix : int
        if fix==0 nothing is fixed, fix==1 depths are fixed, fix==2 
        hypocenters fixed, fix==3 hypocenters and origin time fixed
    usePhases : list
        List of phases to use, all other phases will be skipped
    fixFirstStation : bool
        If True do not set any lat, lon, or depth on each terminator
        line. HypoInverse will then use the first hit station as the 
        starting location with some reasonable depth (a few kilometers)
    
    Note
    --------
    Assumes the channel to be ZENZ for writing file. 
    """
    phases = readKey(phases, key_type='phases')
    evekey = readKey(evekey, key_type='template')
    with open(outname,'wb') as phafil:
        phafil.write('\n')
        for eveind, everow in evekey.iterrows(): # Loop events
            phas = phases[phases.Event==everow.NAME]
            if len(phas) < 1: # go to next event if no phase info
                continue
            for phaind, pha in phas.iterrows():
                stmp = obspy.UTCDateTime(pha.TimeStamp)
                phase = pha.Phase.upper()
                net = pha.Station.split('.')[0]
                sta = pha.Station.split('.')[1] 
                chan = pha.Channel
                # make sure all chans/stations have expected lengths
                _checkLens(net, chan, sta)
                if phase not in usePhases:
                    continue
                line = _makeSHypStationLine(sta, chan, net, stmp, phase) 
                phafil.write(line)
            el = _makeHypTermLine(pha, everow, fix, fixFirstStation)
            phafil.write(el)
            phafil.write('\n')            

def _checkLens(net, chan, sta):
    """
    Check max lens of net (2), chan(3), and sta (5)
    """
    if len(net) > 2:
        msg = 'network code must be 2 characters or less, %s is not' % net
        detex.log(__name__, msg, level='error')  
    if len(chan) > 3:
        msg = 'channel code must be 3 characters or less, %s is not' % chan
        detex.log(__name__, msg, level='error') 
    if len(sta) > 5:
        msg = 'station code must be 5 characters or less, %s is not' % sta
        detex.log(__name__, msg, level='error')  
                  
def _makeSHypStationLine(sta, cha, net, ts, pha):
    Ptime = obspy.core.UTCDateTime(ts)
    datestring = _killChars(Ptime.formatIRISWebService())
    YYYYMMDDHHMM = datestring[0:12]
    secs = float(datestring[12:])
    ssss = '%5.2f' % secs
    end = '01'
    ty = '%s 0' % pha
    line = "{:<5}{:<4}{:<5}{:<3}{:<12}{:<80}{:<2}\n".format(
                        sta, net, cha, ty, YYYYMMDDHHMM, ssss, end)
    return line
    
def _makeHypTermLine(pha, everow, fix, fixFirstStation):
    space=' '
    if fix == 0:
        fixchar = ' '
    elif fix == 1:
        fixchar = '-'
    elif fix == 2:
        fixchar = 'X'
    elif fix == 3:
        fixchar = 'O'
    UTC=obspy.core.UTCDateTime(everow.TIME)
    iws = UTC.formatIRISWebService()
    hhmmssss = _killChars(iws)[8:16]
    if fixFirstStation:
        lat, latmin, latchar = ' ', ' ', ' '
        lon, lonmin, lonchar = ' ', ' ', ' '
        dep = ' '
    else:
        lat, latmin, latchar = _returnLat(everow.LAT)
        lon, lonmin, lonchar = _returnLon(everow.LON)
        dep = '%05.2f' % everow.DEPTH
    endline="{:<6}{:<8}{:<3}{:<4}{:<4}{:<4}{:<5}{:<1}\n".format(
    space, hhmmssss, lat, latmin, lon, lonmin, dep, fixchar)
    return endline 

def makeHypoInverseStationFile(stationKey, outname):
    """
    Make a hypoinverse station file as defined in station data format #2, 
    p. 30 of hyp1.41 user's manual
    
    Parameters
    ------------
    stationKey : str or DataFrame
        Path to station key csv or loaded dataframe with required columns
    outname : str
        The file to write
    """
    with open(outname,'wb') as stafil:
        stakey = readKey(stationKey, key_type='station')
        for ind, srow in stakey.iterrows():
            net = srow.NETWORK
            sta = srow.STATION
            chans = srow.CHANNELS
            latd, latm, latc = _returnLat(srow.LAT, degPre=4)
            lond, lonm, lonc = _returnLon(srow.LON, degPre=4)
            ele = '%4d' % srow.ELEVATION
            for chan in chans.split('-'):
                li = _makeInvStaLine(net, sta, chan, lond, lonm, lonc, latd, 
                                latm, latc, ele)
                stafil.write(li)

def _makeInvStaLine(net, sta, chan, lond, lonm, lonc, latd, 
                    latm, latc, ele):
    chco = ' ' # optional 1 letter chan code
    fstr = "{:<6}{:<3}{:<1}{:<5}{:<3}{:<7}{:<1}{:<4}{:<7}{:<1}{:<4}"
    sto = fstr.format(sta, net, chco, chan,latd, latm, latc, lond, 
                      lonm, lonc, ele)
    ends = '5.0  P  0.00  0.00  0.00  0.00 0  0.00--'
    return "{:<86}".format(sto + ends) + os.linesep
        
def readHypo2000Sum(sumfile):
    """
    read a sum file from hyp2000 and return DataFrame with info loaded into,
    you guessed it, a DataFrame

    Parameters
    --------------
    sumfile : str
        Path to the summary file to read
        
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

def readHypo71Sum(sumfile):
    """
    Read a summary file from hypoinverse in the y2k compliant hypo71 
    format
    
    Parameters
    ----------
    sumfile : str
        Path the the sum file
        
    Returns
    -----------
    DataFrame populated with sumfile info
    """
    fw = [(0,20), (19,22), (22,23), (23,28), (28,32), (32,33), (33,38), 
          (38,45), (52,55), (55,59), (59,64), (64,69), (69,74), (74,79) ]
    cols = ['ds', 'latd', 'latc', 'latm', 'lond', 'lonc', 'lonm', 'depth', 
            'numphase', 'azgap', 'stadist', 'rms','horerr', 'vererr']
    toDrop = ['ds', 'latd', 'latc', 'latm', 'lond', 'lonc', 'lonm']
    df = pd.read_fwf(sumfile, colspecs=fw, names=cols )
    
    latmul = [1 if x else -1 for x in df['latc'].isnull()]
    df['lat'] = np.multiply((df['latd'] + df['latm']/60.), latmul)
    lonmul = [1 if x else -1 for x in df['lonc'].isnull()]
    df['lon'] = np.multiply((df['lond'] + df['lonm']/60.), lonmul)
    utcs = [obspy.UTCDateTime(x.replace(' ','')) for x in df.ds]
    irisws = [x.formatIRISWebService().replace(':','-') for x in utcs]
    times = [x.timestamp for x in utcs]
    names = [x.split('.')[0] for x in irisws]
    df['times'] = times
    df['names'] = names
    df.drop(toDrop, axis=1, inplace=True)
    return df


########## NonLinLoc Functions ##############

def writePhaseNLL(phases, evekey, NLLoc_dir, useP=True, useS=True):
    """ 
    Write a y2k complient phase file used by hypoinverse 2000, format defined
    on page 113 of the manual for version 1.39
    Parameters
    ---------
    phases : pandas dataframe or path to csv
        Phase input from AssociatePhases script
    evekey : pandas dataframe or csv
        Event info
    outname : str
        File name (path) to write
    useP : bool
        If true write P phases
    useS : bool
        If true write S phases
    """
    if not isinstance(phases, pd.DataFrame):
        phases = pd.read_csv(phases)
    if not isinstance(evekey, pd.DataFrame):
        evekey = pd.read_csv(evekey)
    
    ## Split files if more than 100 events
    
    
    for eveind, everow in evekey.iterrows(): # Loop events
    
        on = everow.NAME.split('.')[0].replace('-','').replace('T','') + '.p'
        outpath = os.path.join(NLLoc_dir, on)
        with open(outpath,'wb') as phafil:
            phas = phases[phases.event==everow.NAME]
            if len(phas) < 1: # got to nect event if no phase info
                continue
            for phaind, pha in phas.iterrows():
                Ppick = obspy.UTCDateTime(pha.ptime)
                Spick = obspy.UTCDateTime(pha.stime)
                if Ppick is not None and Ppick > 0 and useP:
                    line= _makeNLLine(pha, everow, 'P')
                    phafil.write(line)
                if Spick is not None and Spick > 0 and useS:
                    line= _makeNLLine(pha, everow, 'S')
                    phafil.write(line)
            phafil.write('\n')            

def _makeNLLine(pha, everow, phase):
    if phase == 'P':
        utc= obspy.UTCDateTime(pha.ptime)
    elif phase == 'S':
        utc = obspy.UTCDateTime(pha.stime)
    staname = '%-6s' % pha.station
    inst = '%-4s' % '?'
    comp = '%-4s' % '?'
    ponset = '%-1s' % '?'
    pdes = '%-6s' % phase
    fmot = '%-1s' % '?'
    ymd = '%04d%02d%02d' % (utc.year, utc.month, utc.day)
    hm = '%02d%02d' % (utc.hour, utc.minute)
    sec = '%07.4f' % (float(utc.second) + utc.microsecond/1000000.)
    err = '%-3s' % 'GAU'
    #errMag = '%9.2e' % ermag
    errMag = '%-9s' % '.01'
    codadur = '%9.2e' % -1
    amp = '%9.2e' % -1
    per = '%9.2e' % -1
    oustr = ' '.join([staname, inst, comp, ponset, pdes, fmot, ymd, hm, sec, 
                      err, errMag, codadur, amp, per])
    return oustr + '\n'


     
        
################# Read/Create detex key files ###################




def readKey(dfkey, key_type='template'):
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
    req_phases = set(['TimeStamp', 'Event', 'Station', 'Phase'])
    req_columns = {'template': req_temkey, 'station': req_stakey, 
                   'phases':req_phases}
    key_types = req_columns.keys()
    
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
               % (df.columns, key_type, req_columns[key_type]))
        detex.log(__name__, msg, level='error')

    tdf = df.loc[:, list(req_columns[key_type])]
    condition = [all([x != '' for item, x in row.iteritems()])
                 for num, row in tdf.iterrows()]
    df = df[condition]
    
    # TODO if column TIME is utcDateTime object sorting fails, fix this
    df.sort_values(by=list(req_columns[key_type]), inplace=True)
    df.reset_index(drop=True, inplace=True)

    # specific operations for various key types
    if key_type == 'station':
        df['STATION'] = [str(x) for x in df['STATION']]
        df['NETWORK'] = [str(x) for x in df['NETWORK']]
    return df

def inventory2StationKey(inv, starttime, endtime, fileName=None):
    """
    Function to create a station key from an obspy station inventory
    
    Parameters
    ----------
    inv : an obspy.station.inventory.Inventory instance
        The inventory to use to create the station key, level of inventory
        must be at least "channel"
    starttime : obspy.UTCDateTime instance
        The start time to be written to station key
    endtime : obspy.UTCDateTime instance
        The end time to be written to the station key
    fileName : None or str
        If str then path to file to save (as csv), default name is 
        StationKey.csv
    Returns
    -------
    A pandas DataFrame of the station key

    """
    # input checks
    if not isinstance(inv, obspy.station.inventory.Inventory):
        msg = 'inv must be an obspy Inventory instance'
        detex.log(__name__, msg, level='error')
    if not isinstance(starttime, obspy.UTCDateTime):
        msg = 'starttime must be an obspy.UTCDateTime instance'
        detex.log(__name__, msg, level='error')
    if not isinstance(endtime, obspy.UTCDateTime):
        msg = 'endtime must be an obspy.UTCDateTime instance'
        detex.log(__name__, msg, level='error')      
    if starttime >= endtime:
        msg = 'starttime must be less than endtime'
        detex.log(__name__, msg, level='error')
    contents = inv.get_contents() # get contents
    if len(contents['channels']) < 1:
        msg = ('Either no channels were found or inventory level is not at '
                'least "channel", try recreating the inventory using '
                'level="channel"')
        detex.log(__name__, msg, level='error')
    # Init DataFrame
    cols = ['NETWORK', 'STATION', 'STARTTIME', 'ENDTIME', 
            'LAT', 'LON', 'ELEVATION', 'CHANNELS']
    
    df = pd.DataFrame(index=range(len(contents['stations'])), columns=cols)
    # Iter inv
    count = 0
    stime = str(starttime).split('.')[0].replace(':', '-')
    etime = str(endtime).split('.')[0].replace(':', '-')
    for net in inv:
        nc = net.code
        for sta in net:
            lat = sta.latitude
            lon = sta.longitude
            ele = sta.elevation
            sc = sta.code
            chanlist = []
            for chan in sta.channels:
                chanlist.append(chan.code)
            chanlist.sort()
            cs = '-'.join(chanlist)
            dat = np.array([nc, sc, stime, etime, lat, lon, ele, cs])
            df.loc[count] = dat
            count += 1
    if isinstance(fileName, str):
        df.to_csv(fileName)
    return df

def templateKey2Catalog(temkey='TemplateKey.csv', picks=None):
    """
    Function to convert a templatekey and optionally a phase picks file to 
    an obspy catalog instance
    
    Parameters
    -----------
    temeky : str, pd.DataFrame
        The standard template key (or path to it)
    picks : str, pd.DataFrame
        A picks file in same format as created by pickPhases
    
    Returns
    ---------
    An Obspy.Catalog object
    """
    temkey = readKey(temkey, "template")
    picks = readKey(picks, 'phases')
    cat = obspy.core.event.Catalog()
    for ind, row in temkey.iterrows():
        cat.events.append(_getEvents(row, picks))
    return cat
        
def _getEvents(row, picks):
    eve = obspy.core.event.Event()
    eve.magnitudes = _getMagnitudes(row)
    eve.origins = _getOrigins(row)
    eve.picks = _getPicks(row, picks)
    return eve
    
def _getMagnitudes(row):
    mag = obspy.core.event.Magnitude()
    mag.mag = row.MAG
    if 'MTYPE' in row.index:
        mag.magnitude_type = row.MTYPE
    return [mag]

def _getOrigins(row):
    lat = row.LAT
    lon = row.LON
    dep = row.DEPTH
    ori = obspy.core.event.Origin
    ori.lattitude = lat
    ori.longitude = lon
    ori.depth = dep
    return [ori]
    
def _getPicks(row, picks):
    phases = []
    if picks is None:
        return phases
    phs = picks[picks.Event==row.NAME]
    for phind, ph in phs.iterrows():
        phases.append(_getPick(row, ph))
    return phases
    
def _getPick(row, ph):
    pick = obspy.core.event.Pick()
    #pick.waveform_id = _getWFID(ph)
#    import ipdb
#    ipdb.set_trace()
    pick.time = obspy.UTCDateTime(ph.TimeStamp)
    pick.phase_hint = ph.Phase
    return pick
    
#def _getWFID(ph):
#    net, sta = ph.Station.split('.')
#    if 'Channel' in ph.index:
#        chan = ph.Channel
#    else:
#        chan = ''
#    loc = ''
#    seedid = '%s.%s.%s.%s' % (net, sta, loc, chan)
#    return obspy.core.event.WaveformStreamID(seed_string=seedid)
    
    
    
        
    
    
    
def EQSearch2TemplateKey(eq='eqsrchsum', oname='eqTemplateKey.csv'):
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
    return DF

def catalog2Templatekey(cat, fileName=None):
    """
    Function to get build the Detex required file TemplateKey.csv 
    from an obspy catalog object

    Parameters
    -----------
    catalog : instance of obspy.core.event.Catalog
        The catalog to use in the template key creation
    filename : str or None
        If None no file is saved, if str then path to the template key to
        save. Default is StationKey.csv.
    Returns
    --------
    A pandas DataFrame of the template key information found in the 
    catalog object

    Notes
    -------
    obspy catalog object docs at:
    http://docs.obspy.org/packages/autogen/obspy.fdsn.client.Client.get_events
    .html#obspy.fdsn.client.Client.get_events
    """
    if not isinstance(cat, obspy.core.event.Catalog):
        msg = 'input is not an obspy catalog object'
        detex.log(__name__, msg, level='error')
    cols = ['NAME', 'TIME', 'LAT', 'LON', 'DEPTH', 'MAG',
            'MTYPE', 'CONTRIBUTOR']
    df = pd.DataFrame(index=range(len(cat)), columns=cols)
    

    for evenum, event in enumerate(cat):
        if not event.origins:
            msg = ("Event '%s' does not have an origin" % 
            str(event.resource_id))
            detex.log(__name__, msg, level='debug')
            continue
        if not event.magnitudes:
            msg = ("Event '%s' does not have a magnitude" % str(
                event.resource_id))
            detex.log(__name__, msg, level='debug')
        origin = event.preferred_origin() or event.origins[0]
        lat = origin.latitude
        lon = origin.longitude
        dep = origin.depth / 1000.0
        time = origin.time.formatIRISWebService().replace(':', '-')
        name = time.split('.')[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        mag = magnitude.mag
        magty = magnitude.magnitude_type
        auth = origin.creation_info.author
        dat = np.array([name, time, lat, lon, dep, mag, magty, auth])
        df.loc[evenum] = dat
    if isinstance(fileName, str):
        df.to_csv(fileName)
    return df

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
   # try:
    if sql is None:
        sql = 'SELECT %s FROM %s' % ('*', tableName)
    with connect(corDB, detect_types=PARSE_DECLTYPES) as con:
        try:
            df = psql.read_sql(sql, con)
        except pd.io.sql.DatabaseError:
            msg = "Table %s not found in %s" % (tableName, corDB)
            detex.log(__name__, msg, level='warning', pri=True)
            return None
        if convertNumeric:
            for item, ser in df.iteritems():
                try:
                    serConverted = pd.to_numeric(ser)
                    df[item] = serConverted
                except ValueError:
                    pass
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

def readLog(logpath='detex_log.log'):
    """
    Read the standard detex log into a dataframe. Columns are: Time, Mod,
    Level, and Msg.
    
    Parameters
    -------------
    logpath : str
        Path the the log file
    
    Returns
    -----------
    DataFrame with log info
    """
    df = pd.read_table(logpath, names=['Time','Mod','Level','Msg'])           
    return df
    
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
               stationkey='StationKey.csv', pickFile='PhasePicks.csv',
               skipIfExists=True, **kwargs):
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
    skipIfExists : bool 
        If True skip any events/stations that already have any phase picks
    kwargs passed to quickFetch, see detex.getdata.quickFetch for details
    Notes
    ----------
    Required columns are : TimeStamp, Station, Event, Phase
    Station field is net.sta (eg TA.M17A)
    """
    temkey = readKey(templatekey, key_type='template')
    stakey = readKey(stationkey, key_type='station')

    cols = ['TimeStamp', 'Station', 'Event', 'Phase', 'Channel', 'Seconds']
    fetcher = detex.getdata.quickFetch(fetch, **kwargs)
    
    ets = {} # events to skip picking on
    count = 0
    
    # must init the PyQt app outside of the loop or else it kills python
    qApp = PyQt4.QtGui.QApplication(sys.argv)

    # load pickfile if it exists
    if os.path.exists(pickFile):  
        DF = pd.read_csv(pickFile)
        if len(DF) < 1:  # if empty then delete
            os.remove(pickFile)
            DF = pd.DataFrame(columns=cols)
        else:
            if skipIfExists:
                for ind, row in DF.iterrows():
                    if not row.Station in ets:
                        ets[row.Station] = []
                    ets[row.Station].append(row.Event)
    else:
        DF = pd.DataFrame(columns=cols)
    for st, event in fetcher.getTemData(temkey, stakey, skipDict=ets):
        if st is None or len(st) < 1: # skip if no data returned
            continue
        count += 1
        #reload(detex.streamPick)

        Pks = None  # needed so OS X doesn't crash
        Pks = detex.streamPick.streamPick(st, ap=qApp)

        tdict = {}
        saveit = 0  # saveflag

        for b in Pks._picks:
            if b:
                tstamp = b['time'].timestamp
                chan = b['waveform_id']['channel_code']
                tdict[b.phase_hint] = [tstamp, chan] 
                saveit = 1
        if saveit:
            for key in tdict.keys():
                stmp = tdict[key][0]
                chan = tdict[key][1]
                secs = '%3.5f' % stmp
                sta = str(st[0].stats.network + '.' + st[0].stats.station)
                di = {'TimeStamp': stmp, 'Station': sta, 'Event': event, 
                      'Phase': key, 'Channel':chan, 'Seconds':secs}
                DF = DF.append(pd.Series(di), ignore_index=True)
        if not Pks.KeepGoing:
            msg = 'Exiting picking GUI, progress saved in %s' % pickFile
            detex.log(__name__, msg, level='info', pri=True)
            DF.sort_values(by=['Station', 'Event'], inplace=True)
            DF.reset_index(drop=True, inplace=True)
            DF.to_csv(pickFile, index=False)
            return
        if count % 10 == 0: # save every 10 phase picks
            DF.sort_values(by=['Station', 'Event'], inplace=True)
            DF.reset_index(drop=True, inplace=True)
            DF.to_csv(pickFile, index=False)
    DF.sort_values(by=['Station', 'Event'], inplace=True)
    DF.reset_index(drop=True, inplace=True)
    DF.to_csv(pickFile, index=False)
    
############### Misc functions
    
def _killChars(string, charstokill=['-', 'T', ':']):
    for ctk in charstokill:
        string = string.replace(ctk, '')
    return string

def _returnLat(lat, degPre=1):
    """
    functon to take lattitude and return lattitude in hypo inverse format
    lat degrees, lat decimal minutes with degPre precision after decimal
    """
    if lat < 0:
        lat = abs(lat)
        cha = 'S'
    else:
        cha = 'N'
    #cha = ' '
    # take decimal degrees spit out degrees decimal minutes
    latds = "{:<2}".format(int(lat))
    latms = ('%4.' + (('%d') % degPre) + 'f') % ((lat % int(lat)) * 60)
    return latds, latms, cha
    
def _returnLon(lon, degPre=1):
    """
    functon to take longitude and return longitude in hypo inverse format
    lon degrees, lon decimal minutes with degPre precision after decimal
    """
    if lon < 0:
        lon = abs(lon)
        cha = 'W'
    else:
        cha = 'E'
        #cha = ' '
    # take decimal degrees spit out degrees decimal minutes
    londs = "{:<3}".format(int(lon))
    lonms = ('%4.' + (('%d') % degPre) + 'f') % ((lon % int(lon)) * 60)
    return londs, lonms, cha
