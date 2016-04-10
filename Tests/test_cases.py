# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:36:59 2016

@author: derrick
Module for running test casses
"""

import pytest
import os
import detex
from collections import namedtuple
case = pytest.mark.skipif(not pytest.config.getoption("--case_tests"), 
                          reason="dont run test cases")

CasePaths = namedtuple('CasePaths', ['stakey', 'temkey', 'phases', 'dir_path'])

########### detex workflow setup fixtures
# These fixtures run the basic tutorial workflow
@pytest.fixture(scope='session')
def case_paths(test_directory):
    """
    return the paths to the key files for detex and current test directory in 
    """
#    test_dirs = request.config.getoption("--test_direcotory")
    temkey_path = os.path.join(test_directory, 'TemplateKey.csv')
    stakey_path = os.path.join(test_directory, 'StationKey.csv')
    phase_path = os.path.join(test_directory, 'PhasePicks.csv')
    
    if all([os.path.exists(x) for x in [temkey_path, stakey_path, phase_path]]):
        return temkey_path, stakey_path, phase_path
    else: # if not all keys are found fail the rest of the tests
        msg = ("StationKey.csv, TemplateKey.csv, and PhasePicks.csv not all" 
                " found in %s, failing dependant tests" % test_directory)
        pytest.fail(msg)

# return the paths to the data directories for detex
@pytest.fixture(scope='session')
def data_directory_paths(test_directory):
    condir_path = os.path.join(test_directory, 'ContinuousWaveForms')
    evedir_path = os.path.join(test_directory, 'EventWaveForms')
    return condir_path, evedir_path

# Set the logger and delete it when finished
@pytest.yield_fixture(scope="session")
def set_logger():
    detex.setLogger(fileName=log_name)
    yield log_name
    os.remove(log_name)

# Load keys into memory
@pytest.fixture(scope="session")
def load_keys(key_paths):
    temkey_path, stakey_path, phase_path = key_paths
    temkey = detex.util.readKey(temkey_path, 'template')
    stakey = detex.util.readKey(stakey_path, 'station')
    phases = detex.util.readKey(phase_path, 'phase')
    return temkey, stakey, phases

# Make data dirs, (IE download data from IRIS)
@pytest.yield_fixture(scope="session")
def make_data_dirs(load_keys):
    temkey, stakey, phases = load_keys
    detex.getdata.makeDataDirectories(templateKey=temkey, stationKey=stakey)
    # dont delete when finished gitignore will keep it from being pushed  

# return continuous data fetcher from continuous data file
@pytest.fixture(scope="session")
def continuous_data_fetcher_condir(data_directory_paths):
    condir = data_directory_paths[0]
    fet = detex.getdata.DataFetcher(method='dir', directoryName=condir)
    return fet

# return event data fetcher from event waveforms file
@pytest.fixture(scope="session")
def event_data_fetcher_evedir(data_directory_paths):
    evedir = data_directory_paths[1]
    fet = detex.getdata.DataFetcher(method='dir', directoryName=evedir)
    return fet

## Create Cluster
@pytest.fixture(scope="session")
def create_cluster(make_data_dirs, event_data_fetcher_evedir):
    cl = detex.createCluster(fetch_arg=event_data_fetcher_evedir)
    return cl

@pytest.fixture(scope='session')
def create_subspace(create_cluster, continuous_data_fetcher):
    cl = create_cluster
    ss = detex.createSubSpace(clust=cl, conDatFetcher=continuous_data_fetcher)
    return ss
    
@pytest.fixture(scope='session')
def attach_pick_times(create_subspace):
    ss = create_subspace
    ss.attachPickTimes()
    return cl
