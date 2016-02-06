# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 11:45:19 2016

@author: derrick
tests for utils
"""
import pytest
import detex
import os
import obspy
####### Constants

intro_data = os.path.join('test_cases', 'Test1')
intro_temkey = os.path.join(intro_data, 'TemplateKey.csv')
intro_stakey = os.path.join(intro_data, 'StationKey.csv')
intro_picks = os.path.join(intro_data, 'PhasePicks.csv')


@pytest.fixture(scope='module')
def load_intro_template_key():
    temkey = detex.util.readKey(intro_temkey, 'template')
    return temkey

@pytest.fixture(scope='module')
def load_intro_station_key():
    stakey = detex.util.readKey(intro_stakey, 'station')
    return stakey

@pytest.fixture(scope='module')
def load_intro_picks():
    picks = detex.util.readKey(intro_picks, 'phases')
    return picks

@pytest.fixture(scope='module')
@pytest.mark.parametrize('temkey, picks', [(load_intro_template_key, 
                                           load_intro_picks)])
def run_templateKey2Catalog(load_intro_template_key, load_intro_picks):
    temkey = load_intro_template_key
    picks = load_intro_picks
    return detex.util.templateKey2Catalog(temkey, picks=picks)


class Test_templateKey2Catalog:
    
    def test_cat_type(self, run_templateKey2Catalog):
        cat = run_templateKey2Catalog
        assert isinstance(cat, obspy.core.event.Catalog)



































