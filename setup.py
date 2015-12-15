# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:11:57 2015

@author: Derrick
"""

from setuptools import setup, find_packages
#from codecs import open
#from os import path


setup(
    name='detex',

    version = '1.0.5',

    description = 'A package for performing subspace and correlation detections on seismic data',
    
    # The project's main homepage.
    url = 'https://github.com/d-chambers/detex',

    # Author details
    author = 'Derrick Chambers',
    author_email = 'djachambeador@gmail.com',

    # Liscense
    license = 'MIT',

    classifiers = [

        'Development Status :: 4 - Beta',
        'Intended Audience :: Geo-scientists',
        'Topic :: Earthquake detections',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords = 'seismology signal detection',
    packages = find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires = ['peppercorn', 'obspy', 'basemap', 'numpy', 
                        'pandas >= 0.17.0', 'scipy', 'matplotlib',
                        'multiprocessing', 'glob2'],
)
