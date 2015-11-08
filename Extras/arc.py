# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:48:55 2014

@author: Derrick
"""
import numpy as np
import obspy
import os
import glob
import pandas as pd
import detex


def _arcOrTime(st):  # Function to convert string time to obspy UTC object
    UTCob = obspy.core.UTCDateTime(st[0:14])
    UTCob = UTCob + float(st[14:16]) / 100.0
    return UTCob


def _arcLatLong(stLat, stLong):
    stLat = stLat.split()
    stLong = stLong.split()
    # divide by 100 to get to decimal minutes, 60 to degrees
    lat = float(stLat[0]) + float(stLat[1]) / (60 * 100.0)
    lon = float(stLong[0]) + float(stLong[1]) / (60 * 100.0)
    return lat, lon


def _arcDivide100(st):
    depth = float(st) / 100.0
    return depth


def _arcMag(st):
    mag = float(st.replace(' ', '0')) / 100.0
    return mag


def readArc(afile):
    with open(afile, "r") as myfile:
        arc = myfile.read().replace('$', '')
    arc = arc.split('\n')
    oTime = _arcOrTime(arc[0][0:16])
    Time = obspy.core.UTCDateTime(oTime).formatIRISWebService()
    lat, lon = _arcLatLong(arc[0][16:23], arc[0][23:31])
    depth = _arcDivide100(arc[0].split()[3])
    mag = _arcMag(arc[0][70:73])
    herr = _arcDivide100(arc[0][85:89])
    verr = _arcDivide100(arc[0][89:93])

    return([Time, oTime.timestamp, lat, lon, depth, mag, herr, verr])


def parseArcDir(arcdir='NF_Arcs'):  # Function to parse through standard Arc directory
    detex.util.checkExists(arcdir)  # make sure arcdir exists
    init = 1
    hd = os.getcwd()
    if not os.path.isdir(os.path.join(hd, arcdir)):
        print (arcdir + ' does not exist in ' + hd)
    else:
        years = glob.glob(os.path.join(hd, arcdir, '*'))
        for a in years:
            months = glob.glob(os.path.join(a, '*'))
            for b in months:
                files = glob.glob(os.path.join(b, '*'))
                for c in files:
                    try:
                        if init == 1:
                            Dvect = [readArc(c)]
                            init = 0
                        else:
                            Dvect = Dvect + [readArc(c)]
                    except:
                        print (c + ' failed')

        return pd.DataFrame(Dvect, columns=['Time', 'STMP', 'Lat', 'Lon', 'Depth', 'Mag', 'HorErr', 'VerErr'])


def createArcDB(arcDir='NF_Arcs', arcdb='Arc.db', tableName='arc'):
    detex.util.checkExists([arcDir])
    detex.util.DoldDB(arcdb)
    Ar = parseArcDir(arcDir)
    detex.util.saveSQLite(Ar, arcdb, tableName)


def readArcDB(arcdb='Arc.DB', tablename='arc'):
    detex.util.checkExists(arcdb)
    Df = detex.util.loadSQLite(arcdb, tablename)
    return Df
