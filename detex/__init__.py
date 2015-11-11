# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#   Author: Derrick Chambers
#------------------------------------------------------------------------------
"""
deTex: A Python Toolbox for running subspace detections.
==================================================================
"""
# General imports
import os
import glob
import inspect
import logging
import logging.handlers
import sys

# Detex imports
import getdata
import util
import subspace
import fas
import construct
import results
import streamPick
import pandas_dbms
import version

# import all modules in detex directory
#modules = glob.glob(os.path.dirname(__file__)+"/*.py")
#__all__ = [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f)]


# Imports for lazy people (ie make detex.createCluster callable) 
from construct import createCluster, createSubSpace


#detResults=results.detResults

maxsize = 100 * 1024*1024 # max size log file can be in bytes (100 mb defualt)
verbose = True # set to false to avoid printing to screen
makeLog = True # set to false to not make log file

## Configure logger to be used across all of Detex
def setLogger(makeLog=True,filename='detex_log.log'):
    """
    Function to set up the logger used across Detex
    
    Parameters
    ----------
    makeLog : boolean
        If True write log events to file
    filename : str
        Path to log file to be created
    """
    reload(logging) # reload to reconfigure default ipython log
    cwd=os.getcwd()
    
    fil = os.path.join(cwd,filename)
    fh = logging.FileHandler(fil)
    fh.setLevel(logging.DEBUG)
    fmat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmat)
    fh.setFormatter(formatter)    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger
    

#define basic logging function

def log(name, msg, level='info', pri=False, close=False, e=Exception):
    """
    Function to log important events as detex runs
    Parameters
    ----------
    name : the __name__ statement
        should always be set to __name__ from the log call. This will enable
        inspect to trace back to where the call initially came from and 
        record it in the log
    msg : str
        A message to log
    level : str
        level of event (info, debug, warning, critical, or error)
    pri : bool
        If true print msg to screen without entire log info
    close : bool
        If true close the logger so the log file can be opened
    e : Exception class
        If level == "error" and Exception is raised, e is the type of 
        exception
    """
    # get name of function that called log
    if not verbose:
        pri = False
    cfun = inspect.getouterframes(inspect.currentframe())[1][0].f_code.co_name 
    log = logging.getLogger(name+'.'+cfun)
    if level == 'info' and makeLog:
        log.info(msg)
    elif level == 'debug' and makeLog:
        log.debug(msg)
    elif (level == 'warning' or level == 'warn') and makeLog :
        log.warning(msg)
    elif level == 'critical' and makeLog:
        log.critical(msg)
    elif level == 'error':
        if makeLog:
            log.error(msg)
        closeLogger()
        raise e(msg)
    else:
        if makeLog:
            raise Exception('level input not understood, acceptable values are' 
                        ' "debug","info","warning","error","critical"')
    if pri:
        print msg
    if close and makeLog: #close logger
        closeLogger()
    
def closeLogger():
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
        
def deb(varlist):
    global de
    de = varlist
    sys.exit(1)
if makeLog:
    logger=setLogger()
    logger.info('Imported Detex')





