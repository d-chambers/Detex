# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:11:57 2015

@author: Derrick
"""

from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
#with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
#    long_description = f.read()

setup(
    name='detex',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.1',

    description='A package for performing subspace and correlation detections on seismic data',
    # The project's main homepage.
    url='https://github.com/DerrickChambers/detex',

    # Author details
    author='Derrick Chambers',
    author_email='derrick.chambers@comcast.net',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[

        'Development Status :: 3 - Alpha',
        'Intended Audience :: Geo-scientists',
        'Topic :: Earthquake detections',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='seismology signal detection',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['peppercorn','obspy','basemap','numpy','pandas','scipy','matplotlib','joblib','multiprocessing'],
)
