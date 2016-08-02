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
detex_paths = ['/media/Data/Gits/detex', '/media/Research/Gits/detex']
for dtp in detex_paths:
    sys.path.insert(0, dtp)
import detex
assert '/Gits' in detex.__file__

# define paths
base_dir = 'Case1'
subspace_path = 'subspace.pkl'
picktime_path = 'PhasePicks.csv'
# assert os.path.exists(subspace_path) and os.path.exists(picktime_path)

#### step through detex workflow
# ss_old = detex.loadSubSpace(subspace_path)
# ss = detex.createSubSpace(clust=ss_old.clusters)
# ss.attachPickTimes(picktime_path)
# ss.SVD(threshold=.5)
# ss.detex(ssDB='test.db')

### get results
res = detex.results.detResults(requiredNumStations=2)
pdb.set_trace()