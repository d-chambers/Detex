# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:40:56 2016

@author: derrick
config. file for tests
"""
import pytest
import detex
import os
import shutil


###### A few globals
log_name = 'detex_test.log'
test_dir = os.path.join('test_cases', 'Test1')


########### detex workflow setup fixtures
@pytest.fixture(scope="session")
def set_logger():
    detex.setLogger(fileName=log_name)
    yield log_name
    os.remove(log_name)

@pytest.fixture(scope="session")
def load_keys(test_dir):
    temkey = detex.util.readKey('TemplateKey.csv', 'template')
    stakey = detex.util.readKey('StationKey.csv', 'station')
    phases = detex.util.readKey('PhasePicks.csv', 'phase')
    return temkey, stakey, phases

@pytest.fixture(scope="session")
def make_data_dirs(load_keys):
    temkey, stakey, phases = load_keys
    detex.getdata.makeDataDirectories(templateKey=temkey, stationKey=stakey)
    yield # as clean up remove the directories just created
    shutil.rmtree(detex.getdata.eveDirDefault)
    shutil.rmtree(detex.getdata.conDirDefault)

@pytest.fixture(scope="session")
def create_cluster(make_data_dirs):
    cl = detex.createCluster()
        
