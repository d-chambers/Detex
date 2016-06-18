#!/usr/local/anaconda/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 14:16:09 2015

@author: Derrick
"""

import glob
import os
import shutil
import sys

import numpy as np
import obspy
import pandas as pd

# get start and stop times



# stations = glob.glob(os.path.join(conDir,'*'))
countSoFar = 0  # This global variable used to accurately keep progress bar.


def divideIntoHours(utc1, utc2):
    """
    Function to take two utc date time objects and create a generator to yield
    all time in between by hour
    Inputs can be any obspy readable format
    """
    utc1 = obspy.UTCDateTime(utc1)
    utc2 = obspy.UTCDateTime(utc2)

    # convert to time stamps (epoch time)
    ts1 = utc1.timestamp - utc1.timestamp % 3600
    ts2 = utc2.timestamp - utc2.timestamp % 3600
    t = ts1
    while t <= ts2:
        yield obspy.UTCDateTime(t)  # yield a value
        t += 3600  # add an hour


def makePath(conDir, starow, utc):
    """
    Function to make a path corresponding to current station and time
    """
    year = '%04d' % utc.year
    julday = '%03d' % utc.julday
    hour = '%02d' % utc.hour
    stanet = starow.NETWORK + '.' + starow.STATION  # NET.STATION format
    path = os.path.join(conDir, stanet, year, julday, stanet + '.' +
                        year + '-' + julday + 'T' + hour + '*')
    return path


def checkQuality(stPath):
    """
    load a path to an obspy trace and check quality
    """
    st = obspy.read(stPath)
    lengthStream = len(st)
    gaps = st.getGaps()
    gapsum = np.sum([x[-2] for x in gaps])
    starttime = min([x.stats.starttime.timestamp for x in st])
    endtime = max([x.stats.endtime.timestamp for x in st])
    duration = endtime - starttime
    if len(gaps) > 0:
        hasGaps = True
    else:
        hasGaps = False
    exists = True
    outDict = {'Exists': exists, 'HasGaps': hasGaps, 'Length': lengthStream,
               'Gaps': gapsum, 'Timestamp': utc, 'Duration': duration}
    return outDict


def _move_files2trash(fil, newDir):
    pathlist = fil.split(os.path.sep)
    newpathlist = pathlist[:]  # slice to make copy
    newpathlist[0] = newDir
    dirPath = os.path.join(*newpathlist[:-1])
    if not os.path.exists(dirPath):  # if the directory isn't there make it
        os.makedirs(dirPath)
    shutil.move(os.path.join(*pathlist), os.path.join(*newpathlist))  # move file


# This will create and update the progress bar
def _progress_bar(total, fileMoveCount):
    global countSoFar
    countSoFar += 1
    width = 25
    percent = float((float(countSoFar) / float(total)) * 100.0)
    completed = int(percent)
    totalLeft = 100
    completedAmount = int(completed / (float(totalLeft) / float(width)))
    spaceAmount = int((float(totalLeft) - float(completed)) /
                      (float(totalLeft) / float(width)))

    for i in xrange(width):
        sys.stdout.write("\r[" + "=" * completedAmount + " " * spaceAmount +
                         "]" + str(round(float(percent), 2)) + "%" + " " + str(countSoFar) +
                         "/" + str(total) + "  Bad Files:" + str(fileMoveCount) + " ")
        sys.stdout.flush()


# This will count the number of files in the folder structure
def _file_count(con_dir):
    print("Counting files for progress bar..."),
    sys.stdout.flush()
    count = 0
    for subdir, dirs, files in os.walk(con_dir):
        for file in files:
            count = count + 1
    print("DONE\n")
    return (count)


## Check continuity of Continuous Data Directory

# Required inputs
def check_data_quality(con_dir='ContinuousWaveForms',
                       eve_dir='EventsWaveForms',
                       stakey='StationKey.csv',
                       temkey='EventKey.csv',
                       move_files=False,
                       write_files=True,
                       bad_files_name='BadContinousWaveForms.txt',
                       bad_files_dir='BadContinuousWaveForms',
                       max_gap_duration=1,
                       minDuration=3570):
    # Init dataframe
    columns = ['Exists', 'HasGaps', 'Length', 'Gaps', 'Timestamp', 'Duration']
    df = pd.DataFrame(columns=columns)
    # Parameters for moving files
    file_move_count = 0  # This keeps a count of how many files were moved

    # read station/template keys
    if isinstance(stakey, str):
        stakey = pd.read_csv(stakey)
    elif not isinstance(stakey, pd.DataFrame):
        raise Exception('stakey must be string or DataFrame')
    if isinstance(temkey, str):
        temkey = pd.read_csv(temkey)
    elif not isinstance(temkey, pd.DataFrame):
        raise Exception('temkey must be string or DataFrame')

    print
    "\nWrite to File = " + str(write_files)
    print
    "Move files = " + str(move_files) + "\n"
    counted_files = _file_count(con_dir)
    print
    "Beginning Data Quality check..."

    for stanum, starow in stakey.iterrows():  # iter through station info
        utcGenerator = divideIntoHours(starow.STARTTIME, starow.ENDTIME)
        for utc in utcGenerator:
            utcpath = makePath(con_dir, starow, utc)
            fil = glob.glob(utcpath)
            _progress_bar(counted_files, file_move_count)
            if len(fil) > 0:
                qualDict = checkQuality(fil[0])
                df.loc[len(df)] = pd.Series(qualDict)
                if (move_files or write_files):
                    gaps = qualDict['Gaps']
                    duration = qualDict['Duration']
                    if gaps > max_gap_duration or duration < minDuration:
                        file_move_count = file_move_count + 1
                        if move_files:
                            _move_files2trash(fil[0], bad_files_dir)
                        if write_files:
                            f = open(bad_files_name, 'a')
                            f.write(str(fil[0]) + '\n')
                            f.close()
            elif len(fil) > 1:
                print
                'More than one file found for station hour pair'
                sys.exit(1)
            else:
                qualDict = {'Exists': False, 'HasGaps': False, 'Length': 0,
                            'Duration': 0, 'Gaps': [], 'Path': fil,
                            'Timestamp': utc.timestamp}
                df.loc[len(df)] = pd.Series(qualDict)
    if (write_files):
        f = open(bad_files_name, 'w')
        f.write("Total Files Checked:" + str(counted_files) +
                "  Total number of bad files:" + str(file_move_count) + '\n')
        f.close()
    print
    "Quality check complete"
    return df
