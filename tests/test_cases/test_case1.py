# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:36:59 2016

@author: derrick
Module for running test casses
"""

###### imports and such
from __future__ import print_function, unicode_literals, division, absolute_import
from six import string_types
import pytest
import os
import detex
import obspy
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple

import pdb

# mark entire module
pytestmark = [pytest.mark.test_case, pytest.mark.case1]

# define named tuples used by fixtures

########## globals, functions and determine if this module should be skipped

# define path to directory that contains keys and phases
test_case = 'Case1'
directory_path = os.path.join(pytest.test_directory, 'test_cases', test_case)
log_path = os.path.join(directory_path, 'detex.log')
subspace_database = 'SubSpace.db'


# The following are dictionaries that define the input params for each function 
# call of the test cases 

# dict of parameters for continuous data fetcher used for get data call
get_data_fet_args = {'method':'iris', 'client':obspy.clients.fdsn}
# dict of parameters for the make data directories call
make_data_args = {'timeBeforeOrigin':60, 'timeAfterOrigin':240, 'secBuf':120,
                  'conDatDuration':3600, 'getContinuous':True, 
                  'getTemplates':True, 'removeResponse':True, 'opType':'VEL',
                  'prefilt':[0.05, 0.1, 15, 20]}
# dict of params for data fetcher for continuous data
con_data_args = {}
# dict of params for event data fetcher
eve_data_args = {} 
# dict for create cluster args
create_cluster_args = {}
create_cluster_args2 = {'enforceOrigin': True}
# dict for create subspace args
create_subspace_args = {}
# attach phases params
attach_pick_time_args = {'defaultDuration':30}
# params for SVD 
svd_params = {'conDatNum':100, 'threshold':None, 'normalize':False, 
              'useSingles':False, 'validateWaveforms':True}
# params for running detections
detection_params = {'utcStart':None, 'utcEnd':None, 'subspaceDB':subspace_database,
                    'delOldCorrs':True, 'calcHist':True, 'useSubSpaces':True,
                    'useSingles':False, 'estimateMags':True, 'fillZeros':False}
# params for running detResults
results_params = {'ss_associateBuffer':1, 'sg_associateBuffer':2.5, 
                  'requiredNumStations':2, 'veriBuffer':60*10, 'ssDB':subspace_database,
                  'reduceDets':True, 'Pf':False, 'stations':None, 'starttime':None,
                  'endtime':None, 'fetch':'ContinuousWaveForms'}
# params for write detections
write_detections_params = {'onlyVerified':False, 'minDS':False, 'minMag':False, 
                           'eventDir':'EventWaveForms', 'updateTemKey':False, 
                           'timeBeforeOrigin':1*60, 
                           'timeAfterOrigin':4*60, 'waveFormat':"mseed"}


# mark skip on module if required command line not passed or other case run
opt = pytest.config.getoption('--test_case')
if bool(opt):
    if isinstance(opt, string_types):
        if opt in os.path.basename(__file__):
            run_module = True
    else:
        run_module = True
else:
     run_module = False
# skip if required
reason = "test cases not used or this one not selected"
pytestmark = pytest.mark.skipif(not run_module, reason=reason)
#pytestmark = pytest.mark(test_case)
########### detex workflow setup fixtures and tests
# These fixtures run the basic tutorial workflow

####### Test inputs
# change working directory and then go back
@pytest.yield_fixture(scope='session')
def cd_into_case_dir():
    here = os.getcwd()
    there = directory_path
    os.chdir(there)
    yield
    os.chdir(here)
    
# get paths to keys and directory, return namedtuple else fail downstream tests
@pytest.fixture(scope='session')
def case_paths(cd_into_case_dir):
    """
    return the paths to the key files for detex and current test directory in 
    """
    # init case directory
    cols = ['temkey', 'stakey', 'phases', 'condir', 'evedir', 'casedir', 'verify']
    CasePaths = namedtuple('CasePaths', cols)
#    test_dirs = request.config.getoption("--test_direcotory")
    temkey_path = os.path.join(directory_path, 'TemplateKey.csv')
    stakey_path = os.path.join(directory_path, 'StationKey.csv')
    phase_path = os.path.join(directory_path, 'PhasePicks.csv')
    condir_path = os.path.join(directory_path, 'ContinuousWaveForms')
    evedir_path = os.path.join(directory_path, 'EventWaveForms')
    verifile_path = os.path.join(directory_path, 'veriFile.csv')
    case_paths = CasePaths(temkey_path, stakey_path, phase_path, condir_path,
                           evedir_path, directory_path, verifile_path)
    if all([os.path.exists(x) for x in [temkey_path, stakey_path, phase_path]]):
        return case_paths
    else: # if not all keys are found fail the rest of the tests
        msg = ("StationKey.csv, TemplateKey.csv, and PhasePicks.csv not all" 
                " found in %s, failing dependant tests" % directory_path)
        pytest.fail(msg)

# Set the logger and delete it when finished
@pytest.yield_fixture(scope="module")
def set_logger(case_paths):
    detex.setLogger(fileName=log_path)
    yield log_path
    os.remove(log_path)

class TestLog():
    # make sure a log was created
    def test_log_exists(self, set_logger):
        assert os.path.exists(set_logger)

# Load keys into memory
@pytest.fixture(scope="module")
def detex_keys(case_paths):
    DetexKeys = namedtuple('DetexKeys', ['temkey', 'stakey', 'phases'])
    temkey = detex.util.readKey(case_paths.temkey, 'template')
    stakey = detex.util.readKey(case_paths.stakey, 'station')
    phases = detex.util.readKey(case_paths.phases, 'phases')
    detex_keys = DetexKeys(temkey, stakey, phases)
    return detex_keys
    
# class for input file tests
class TestKeys():
    # test that all keys are DFs
    def test_key_types(self, detex_keys):
        assert all(isinstance(x, pd.DataFrame) for x in detex_keys)
    # test that all lat/lon/time values make sense om stakey
    def test_lat_lon_time_stakey(self, detex_keys):
        stakey = detex_keys.stakey
        assert (abs(stakey.LAT) <= 90).all()
        assert (abs(stakey.LON) <= 180).all()
        assert all([obspy.UTCDateTime(x) for x in stakey.STARTTIME])
        assert all([obspy.UTCDateTime(x) for x in stakey.ENDTIME])
        assert detex.util.req_stakey.issubset(set(stakey.columns))
    # test that all lat/lon values make sense in template key
    def test_lat_lon_stakey(self, detex_keys):
        temkey = detex_keys.temkey
        assert (abs(temkey.LAT) <= 90).all()
        assert (abs(temkey.LON) <= 180).all()
        assert all([obspy.UTCDateTime(x) for x in temkey.TIME])    
        assert detex.util.req_temkey.issubset(set(temkey.columns))
    # test the phase picks
    def test_phase_picks(self, detex_keys):
        picks = detex_keys.phases
        assert detex.util.req_phases.issubset(set(picks.columns))
        assert all([obspy.UTCDateTime(x) for x in picks.TimeStamp]) 

####### Test get data

# init data fetcher of choice for downloading continuous data
@pytest.fixture(scope="module")
def data_fetcher(detex_keys):
    fet = detex.getdata.DataFetcher(**get_data_fet_args)
    return fet

# Make data dirs, (eg download data from IRIS)
@pytest.fixture(scope="module")
def make_data_dirs(detex_keys, data_fetcher, case_paths):
    condir, evedir = case_paths.condir, case_paths.evedir
    fet = data_fetcher
    temkey, stakey, phases = detex_keys
    detex.getdata.makeDataDirectories(templateKey=temkey, stationKey=stakey,
                                      fetch=fet, conDir=condir, 
                                      templateDir=evedir, **make_data_args)
    # dont delete when finished gitignore will keep it from being pushed  

# tests for the makedirectories part
class TestMakeDataDirs():
    # test the data fetcher has required attrs
    def test_data_fetcher(self, data_fetcher):
        for attr in ['getTemData', 'getConData', 'getStream']:
            assert hasattr(data_fetcher, attr)
    # test that the new directories have been created
    def test_exists_directories(self, make_data_dirs, case_paths):
        assert os.path.exists(case_paths.condir)
        assert os.path.exists(case_paths.evedir)
        
        
######### Test directory data fetchers
                                    
# return continuous data fetcher from continuous data file
@pytest.fixture(scope="module")
def continuous_data_fetcher(case_paths):
    path = case_paths.condir
    fet = detex.getdata.DataFetcher(method='dir', directoryName=path, 
                                    **con_data_args)
    return fet

# return event data fetcher from event waveforms file
@pytest.fixture(scope="module")
def event_data_fetcher(case_paths):
    path = case_paths.evedir
    fet = detex.getdata.DataFetcher(method='dir', directoryName=path, 
                                    **eve_data_args)
    return fet

# test the directory fetchers
class TestDirectoryFetchers():
    # test fetcher has required attrs
    def test_continuous(self, continuous_data_fetcher):
        for attr in ['getTemData', 'getConData', 'getStream']:
            assert hasattr(continuous_data_fetcher, attr)      
    # test fetcher has required attrs
    def test_event(self, event_data_fetcher):
        for attr in ['getTemData', 'getConData', 'getStream']:
            assert hasattr(event_data_fetcher, attr)  
            
############ Test cluster

## Create Cluster
@pytest.fixture(scope="module")
def create_cluster(make_data_dirs, event_data_fetcher):
    cl = detex.createCluster(fetch_arg=event_data_fetcher,
                             **create_cluster_args)
    return cl

# any actions to perform on cluster go here
@pytest.fixture(scope="module")
def modify_cluster(create_cluster):
    cl = create_cluster
    cl.updateReqCC(.55)
    cl['TA.M17A'].updateReqCC(.38)
    return cl





# tests for the cluster object
class TestCluster():
    # test types
    def test_type(self, modify_cluster):
        assert isinstance(modify_cluster, detex.subspace.ClusterStream)
    # test that the number of clusters is correct
    def test_cluster_number(self, modify_cluster):
        cl = modify_cluster
        assert len(cl) == 2 # test there are two stations
        for c in cl:
            assert len(c) == 4 # make sure there are exactly 4 clusters

@pytest.fixture(scope='module')
def dendrogram(modify_cluster):
    save_name = 'dendro_test.pdf'
    modify_cluster.dendro(show=False, saveName=save_name)
    return save_name

# test that dendro doesn't raise
class TestDendrogram():
    # test dendro doesn't raise
    def test_dendro(self, dendrogram):
        assert os.path.exists(dendrogram)

# save cluster
@pytest.yield_fixture(scope="module")
def save_cluster(modify_cluster):
    cl = modify_cluster
    cl.write()
    yield cl.filename
    if os.path.exists(cl.filename):
        os.remove(cl.filename)

# any actions to perform on cluster go here
@pytest.fixture(scope="module")
def load_cluster(save_cluster):
    cl = detex.loadClusters(save_cluster)
    return cl
        
class TestLoadCluster():
    # test load cluster
    def test_type(self, load_cluster):
        assert isinstance(load_cluster, detex.subspace.ClusterStream)

# create a cluster with fill
@pytest.fixture(scope="module")
def create_cluster2(make_data_dirs, event_data_fetcher):
    cl = detex.createCluster(fetch_arg=event_data_fetcher,
                             **create_cluster_args2)
    return cl

@pytest.yield_fixture(scope='module')
def hypoDD_output(create_cluster2):
    filename = 'dt.cc'
    cl = create_cluster2
    cl.writeSimpleHypoDDInput(fileName=filename)
    yield filename
    if os.path.exists(filename):
        os.remove(filename)

class TestWriteHypoDD():
    # test output
    def test_exists_output(self, hypoDD_output):
        assert os.path.exists(hypoDD_output)

############# Test Subspace

# create the subspace object
@pytest.fixture(scope='module')
def create_subspace(modify_cluster, continuous_data_fetcher):
    cl = modify_cluster
    ss = detex.createSubSpace(clust=cl, conDatFetcher=continuous_data_fetcher,
                              **create_subspace_args)
    return ss

# attach picktimes found in the phase file
@pytest.fixture(scope='module')
def attach_pick_times(create_subspace, case_paths):
    ss = create_subspace
    ss.attachPickTimes(pksFile=case_paths.phases, **attach_pick_time_args)
    return ss

# perform the svd 
@pytest.fixture(scope='module')
def perform_svd(attach_pick_times):
    ss = attach_pick_times
    ss.SVD(**svd_params)
    return ss

# perform any additional modifications
@pytest.fixture(scope='module')
def modified_subspace(perform_svd):
    ss = perform_svd
    return ss

# test the subspaces created    
class TestSubspace():
    # test that the type is right
    def test_type(self, create_subspace):
        assert isinstance(create_subspace, detex.subspace.SubSpace)
    # test stations in key are found in subspace cluster
    def test_stations(self, create_subspace, detex_keys):
        ss = create_subspace
        stakey = detex_keys.stakey
        stations_in_key = set(stakey.NETWORK + '.' + stakey.STATION)
        stations_in_ss = set(ss.stations)
        assert stations_in_key == stations_in_ss
    # test that the subspace dict is a dict of dataframes
    def test_subspace_dict(self, create_subspace):
        ss = create_subspace
        assert isinstance(ss.subspaces, dict)
        for key, item in ss.subspaces.items():
            assert isinstance(item, pd.DataFrame)
            assert key in ss.stations
    # test that pick times attached to subspace
    def test_attach_picktimes_subspace(self, attach_pick_times):
        ss = attach_pick_times
        for key, df in ss.subspaces.items():
            assert not df.SVDdefined.all()
            for ind, row in df.iterrows():
                assert row.SampleTrims
    # test that SVD is called and basis vectors are defined
    def test_SVD(self, modified_subspace):
        ss = modified_subspace
        for key, df in ss.subspaces.items():
            assert df.SVDdefined.all()
            for ind, row in df.iterrows():
                assert isinstance(row.NumBasis, int) 
                assert isinstance(row.Threshold, float)

# save subspace 
@pytest.fixture(scope='module')
def save_subspace(modified_subspace):
    filename = 'subspace.pkl'    
    ss = modified_subspace
    ss.write(filename)
    return filename

# load a subspace from the given filename
@pytest.fixture(scope='module')
def load_subspace(cd_into_case_dir):#save_subspace):
    save_subspace = 'subspace.pkl'
    ss = detex.loadSubSpace(save_subspace)    
    return ss


# test that the subspace can be laoded and saved
class TestLoadSubspace():
    # load subspace
    def test_load_subspace(self, load_subspace):
        isinstance(load_subspace, detex.subspace.SubSpace)
    

# run detections
@pytest.fixture(scope='module')
def run_detections(load_subspace):
    ss = load_subspace
    ss.detex(**detection_params)
    return ss

class TestDetections():
    def test_detections(self, run_detections):
        assert os.path.exists(subspace_database)

# load results
@pytest.fixture(scope='module')
def results(case_paths, run_detections):
    res = detex.results.detResults(veriFile=case_paths.verify, **results_params)
    return res

# load results
@pytest.fixture(scope='module')
def verifile(case_paths):#, run_detections):
    df = pd.read_csv(case_paths.verify)
    return df
    
# test that the expected results were returned
class TestResults():
    def test_results(self, results, verifile):
        res = results
        # test that all detections are vefified
        assert (len(res.Dets) + len(res.Autos)) == len(res.Vers)
        assert len(res.Vers) == len(verifile)

# write detections 
@pytest.yield_fixture(scope='module')
def write_detections(results):
    file_name = 'temp.csv'
    results.writeDetections(temkeyPath=file_name, **write_detections_params)
    yield file_name
    if os.path.exists(file_name):
        os.remove(file_name)

class TestWriteDetections():
    def test_write_detections(self, write_detections, results):
        assert os.path.exists(write_detections)
        pdb.set_trace()

## test write detection functionality
#class TestWriteDetections():
#    def test_
    




























