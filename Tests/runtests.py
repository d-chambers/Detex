# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:27:35 2015

@author: derrick
Script to run test suite for detex
"""
import os
import glob
import imp

# loop through each test directory and look for examply one file "test.py"
import detex #import detex module
for tdir in glob.glob(os.path.join('Tests', '*')):
    pyrun = glob.glob(os.path.join(tdir, 'test*.py'))
    if len(pyrun) > 0:
        modname = os.path.basename(pyrun[0]).split('.')[0]
        mod = imp.load_source(modname, pyrun[0])
        mod.main()
    
