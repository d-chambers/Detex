# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:40:56 2016

@author: derrick
config. file for tests
"""
# python 2 and 3 compatibility imports
from __future__ import print_function, absolute_import, unicode_literals, division

import pytest
import os
import glob
import sys

##### add paths so that all the detex dependents can simply be in the same dir
## add detex path to start of python path (doesnt test installed version)
pypo_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, pypo_path)

## add other paths that might be in the same dir as pypo 
git_path = os.path.dirname(pypo_path)
def add_paths(git_path):
    """
    iter through paths and add all to python path for session
    """
    repos = glob.glob(os.path.join(git_path, '*'))
    for repo in repos:
        sys.path.append(os.path.abspath(repo))
add_paths(git_path)
# import detex here to ensure the package from the repo is imported
import detex

# add test directory to namespace for solid path in other tests 
def pytest_namespace():
    return {'test_directory': os.path.abspath(os.path.dirname(__file__))}

# global
#default_test_cases_directory = 'test_cases'
#default_test_cases = glob.glob(os.path.join(default_test_cases_directory, '*'))


###### metas

# alow use of general tests or case tests
def pytest_addoption(parser):
    parser.addoption("--test_case", action="store_true", default=True,
        help=("run the test case(s). Pass True to run all, or name of test case"
                "to run only one, e.g. '2' would only run case 2"))
        
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

        
