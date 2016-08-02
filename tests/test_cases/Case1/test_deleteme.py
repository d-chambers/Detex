"""
simple test to add functionality to detections, if you are reading
this delete this file
"""
import obspy
import os
import pdb
import sys
import contextlib

# add gits detex to path
detex_path = '/media/Research/Gits/detex'
sys.path.insert(0, detex_path)
import detex
assert detex_path in detex.__file__

# define paths
base_dir = 'Case1'
subspace_path = 'subspace.pkl'
picktime_path = 'PhasePicks.csv'
# assert os.path.exists(subspace_path) and os.path.exists(picktime_path)

# set through detex workflow
ss_old = detex.loadSubSpace(subspace_path)
ss = detex.createSubSpace(clust=ss_old.clusters)
ss.attachPickTimes(picktime_path)
ss.SVD(threshold=.5)
ss.detex(ssDB='test.db')