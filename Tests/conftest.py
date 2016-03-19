# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:40:56 2016

@author: derrick
config. file for tests
"""
# python 2 and 3 compatibility imports
from __future__ import print_function, absolute_import, unicode_literals
from __future__ import with_statement, nested_scopes, generators, division

import pytest
import detex
import os
import shutil


##### metas
# allow the setting of the test directory (case tests)
def pytest_addoption(parser):
    parser.addoption("--test_directory", action="append", 
                     default=os.path.join('test_cases', 'Test1'),
                     help="A directory to run the tests on, should have keys")

# alow use of general tests or case tests
def pytest_addoption(parser):
    parser.addoption("--case_tests", action="store_true", default=True,
        help="run the test cases define by --test_directory (or defualt)")

def pytest_addoption2(parser):
    parser.addoption("--general_tests", action="store_true", default=True,
        help="run the general tests")


#### markers 

case = pytest.mark.skipif(
       not pytest.config.getoption("--case_tests"),
       reason="dont run case tests")

general = pytest.mark.skipif(
          not pytest.config.getoption("--general_tests"),
          reason="dont run general tests")



###### A few globals
log_name = 'detex_test.log'
#
#test_dir = os.path.join('test_cases', 'Test1')




########### detex workflow setup fixtures
# return the paths to the key files for detex
@pytest.fixture(scope='session')
def key_paths(test_directory):
    temkey_path = os.path.join(test_directory, 'TemplateKey.csv')
    stakey_path = os.path.join(test_directory, 'StationKey.csv')
    phase_path = os.path.join(test_directory, 'PhasePicks.csv')
    return temkey_path, stakey_path, phase_path

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



# Create Subpace

        
