# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:40:56 2016

@author: derrick
config. file for tests
"""
# python 2 and 3 compatibility imports
from __future__ import print_function, absolute_import, unicode_literals, division

import pytest
import detex
import os
import shutil
import glob

# global
default_test_cases_directory = 'test_cases'
default_test_cases = glob.glob(os.path.join(default_test_cases_directory, '*'))


###### metas
## allow the setting of the test directory (case tests)
#def pytest_addoption(parser):
#    parser.addoption("--test_directory", action="store", 
#                     default=','.join(default_test_cases),
#                     help=("A directory to run the tests on, should have keys "
#                            "and pick files. Can pass multiple paths seperated"
#                            " by ,"))

# alow use of general tests or case tests
def pytest_addoption(parser):
    parser.addoption("--test_case", action="store_true", default=True,
        help=("run the test case(s). Pass True to run all, or name of test case"
                "to run only one, e.g. 'Case2' would only run case 2"))
        
    parser.addoption("--general_tests", action="store_true", default=True,
        help="run the general tests")

# parametrize input from test directory (lets multiple test directoies be used)
#def pytest_generate_tests(metafunc):
#    if 'test_directory' in metafunc.fixturenames:
#        metafunc.parametrize("test_directory",
#                             metafunc.config.option.test_directory.split(','))

###### A few globals
#log_name = 'detex_test.log'
#
#test_dir = os.path.join('test_cases', 'Test1')







# Create Subpace

        
