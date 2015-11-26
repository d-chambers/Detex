# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:56:54 2015

@author: derrick
# A test used by pytests to make sure the tutorial works
"""
import detex
import os
import pandas as pd
class TestGetData:
    detex.getdata.makeDataDirectories()
    con_dat_dir = 'ContinuousWaveForms'
    
    def test_data_directories(con_dat_dir):
        assert os.path.exists(con_dat_dir)
        
    index_path = os.path.join(con_dat_dir, '.index.db')
    
    def test_index_path(index_path):
        assert os.path.exists(index_path)
        
    def test_index_readable():
        index_path = os.path.join(con_dat_dir, '.index.db')
        ind = detex.util.loadSQLite(index_path, 'ind')
        assert isinstance(ind, pd.DataFrame)
        
#    def test_filecount():
#        df = detex.util.
