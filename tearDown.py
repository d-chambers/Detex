# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 06:22:20 2015

@author: derrick
clean up all detex_log.log files, all pickled detex classes and detete
continuous and waveform data directory files
"""
import os
import shutil
# define files to delete
files_to_kill = ['detex_log.log', 'clust.pkl', 'SubSpace.db', 'subspace.pkl']
# define directories to delete
dirs_to_kill = ['ContinuousWaveForms', 'EventWaveForms', 'detex.egg-info', 
                'dist', 'build', '.pynb_checkpoints', 'DetectedEvents']
def tear_down():
    for root, dirs, files in os.walk("."):
        path = os.path.normpath(root)
        for fi in files:
            if fi in files_to_kill:
                os.remove(os.path.join(path, fi))
        for di in dirs:
            if di in dirs_to_kill:
                shutil.rmtree(os.path.join(path, di))

if __name__ == "__main__":
    tear_down()