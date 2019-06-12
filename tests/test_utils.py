# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 11:45:19 2016

@author: derrick
tests for utils
"""
from __future__ import absolute_import, unicode_literals, division, print_function

import pytest
import detex
import os
import obspy
import glob
from pathlib2 import Path
from collections import namedtuple

here = Path(__file__).parent

# set paths for data
test_data = here / 'test_data' / 'test_utils'
temkeys = [str(x) for x in (test_data / 'template_keys').glob('*')]
stakeys = [str(x) for x in (test_data / 'station_keys').glob('*')]
picks = [str(x) for x in (test_data / 'picks').glob('*')]
# make sure some files were found, require stakeys and tempkeys to be equal
assert len(temkeys) and len(temkeys) == len(stakeys) == len(picks)

file_sets = zip(temkeys, stakeys, picks)

# serve the keys file paths in a name tupple
@pytest.fixture(scope='session', params=file_sets)
def key_files(request):
    nt = namedtuple('KeyFiles', 'temkey, stakey, picks')
    return nt(*request.param)

# read in the key files and serve named tuple of their classes
@pytest.fixture(scope='session')
def key_Dfs(key_files):
    nt = namedtuple('KeyDfs', 'temkey, stakey, picks')
    temkey = detex.util.readKey(key_files.temkey, 'template')
    stakey = detex.util.readKey(key_files.stakey, 'station')
    picks = detex.util.readKey(key_files.picks, 'phases')
    return nt(temkey, stakey, picks)

# convert template key to catalog
@pytest.fixture(scope='session')
def temkey2catalog(key_Dfs):
    temkey, picks = key_Dfs.temkey, key_Dfs.picks
    return detex.util.templateKey2Catalog(temkey, picks=picks)

# class to test functionality of template key to catalog function
class Test_templateKey2Catalog:
    def test_cat_type(self, temkey2catalog):
        cat = temkey2catalog
        assert isinstance(cat, obspy.core.event.Catalog)



































