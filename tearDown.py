# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 06:22:20 2015

@author: derrick
clean up all detex_log.log files, all pickled detex classes and detete
continuous and waveform data directory files
"""
import os
import shutil
import sys
import glob

# define files to delete
files_to_kill = ['detex_log.log', 'clust.pkl', 'SubSpace.db', 'subspace.pkl']
# define directories to delete
dirs_to_kill = ['ContinuousWaveForms', 'EventWaveForms', 'detex.egg-info', 
                'dist', 'build', '.ipynb_checkpoints', 'DetectedEvents']
            
######### Get paths
def _get_detex_paths():
    # go up one level to avoid importing detex in repo
    cwd = os.getcwd()
    detex_dir = os.path.join(cwd, 'detex')
    up_one = cwd.split(os.path.sep)[1]
    up_one = os.path.dirname(cwd)
    up_two = os.path.dirname(up_one)
    os.chdir(up_two)
    print os.getcwd()
    try: # make sure new instance of detex is imported
        reload(detex)
    except NameError:
        import detex
    detex_path = os.path.dirname(detex.__file__)
    print detex_dir, detex_path
    return cwd, detex_dir, detex_path

############### Bump version functions
def bump_version(cwd, detex_dir, detex_path):
    """
    bump up the version number in __init__ and setup
    """
    version, init_line, setup_line = find_version(detex_dir, cwd)
    verint = [int(x) for x in version.split('.')]
    msg = 'Bump version? y, [n]'
    resp1 = get_user_input(msg)
    if resp1 not in ['y', 'n', '']:
        raise Exception ("input not understood, use 'y', 'n' or enter'")
    if resp1 == 'y':
        msg = 'Bump which? major, minor, [micro]'
        resp2 = get_user_input(msg)
        if resp2 == '' or 'micro':
            verint[2] += 1
        elif resp2 == 'minor':
            verint[1] += 1
        elif resp2 == 'major':
            verint[0] += 1 
        else:
            raise Exception("Input must be: 'major', 'minor', or 'micro'")
        version2 = '.'.join([str(x) for x in verint])
        _replace_str(detex_dir, cwd, init_line, setup_line, version, version2)

def find_version(detex_dir, cwd):
    # Get version indicated in setup.py
    setup = os.path.join(cwd, 'setup.py')
    setup_version_str = _find_line('version =', setup)
    setup_version = setup_version_str.split("'")[1]
    
    # get version in init
    init = os.path.join(detex_dir, '__init__.py')
    init_version_str = _find_line('__version__', init)
    init_version = init_version_str.split("'")[1]
    if not init_version == setup_version:
        raise Exception ("versions not equal in __init__.py and setup.py")
    return init_version, init_version_str, setup_version_str

def _find_line(string, file_name):
    """
    find the first occurence of a string in a file
    """
    
    with open(file_name, 'r') as fi:
        for line in fi:
            if string in line:
                return line

def _replace_str(detex_dir, cwd, init_line, setup_line, version, version2):
    setup = os.path.join(cwd, 'setup.py')
    _replace_line(setup_line, setup_line.replace(version, version2), setup)
    
    init = os.path.join(detex_dir, '__init__.py')
    _replace_line(init_line, init_line.replace(version, version2), init)

def _replace_line(old_line, new_line, file_name):
    with open(file_name, 'r') as fi1:
        with open('temp.txt', 'w') as fi2:
            for lin in fi1:
                if old_line in lin:
                    fi2.write(new_line)
                else:
                    fi2.write(lin)
    shutil.copy('temp.txt', file_name)
    os.remove('temp.txt')

########### Pull Detex from site packages
def pull_detex(cwd, detex_dir, detex_path):
    """
    Function to pull detex from the import path (generally in site packages)
    """
    
    msg = 'pull detex from %s and delete old copy? [y] or n' % detex_dir
    response = get_user_input(msg)
    if response == 'y' or response == '':
        files_to_copy = glob.glob(os.path.join(detex_path, '*.py'))
        for fi in files_to_copy:
            shutil.copy(fi, detex_dir)

########### Tear down
def tear_down():
    for root, dirs, files in os.walk("."):
        path = os.path.normpath(root)
        for fi in files:
            if fi in files_to_kill:
                os.remove(os.path.join(path, fi))
        for di in dirs:
            if di in dirs_to_kill:
                shutil.rmtree(os.path.join(path, di))

######### user inputs
def get_user_input(msg):
    py3 = sys.version_info[0] > 2
    if py3:
        response = input(msg + '\n')
    else:
        response = raw_input(msg + '\n')
    return response
    
################ Go!
if __name__ == "__main__":
    cwd, detex_dir, detex_path = _get_detex_paths()
    bump_version(cwd, detex_dir, detex_path)
    pull_detex(cwd, detex_dir, detex_path)
    tear_down()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
