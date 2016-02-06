# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:13:28 2015

@author: derrick
Tests for __init__ module
"""
import detex
import os

def test_setLogger(set_logger): # check log was created
    log_name = set_logger
    assert os.path.exist(log_name)
    
