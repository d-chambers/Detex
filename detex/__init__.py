# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#   Author: Derrick Chambers
#------------------------------------------------------------------------------
"""
deTex: A Python Toolbox for running subspace detections.
==================================================================
"""
import os
import glob
import inspect
import logging
import logging.handlers
import sys

# import all modules in detex directory
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f)]
from detex import *

# Set shortcuts for lazy people 

#getAllData=getdata.getAllData

createCluster = construct.createCluster

createSubSpace = construct.createSubSpace

#detResults=results.detResults

maxsize = 100 * 1024*1024 # max size log file can be in bytes (100 mb defualt)

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
    reload(logging) # reload to reconfigure default ipython logging behavior
    cwd=os.getcwd()
    
    fil = os.path.join(cwd,filename)
    fh = logging.FileHandler(fil)
    fh.setLevel(logging.DEBUG)
    fmat = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmat)    
    
    fh.setFormatter(formatter)    
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    
    # create file handler which logs even debug messages

    logger.addHandler(fh)
    # create console handler with a higher log level
#    ch = logging.StreamHandler()
#    ch.setLevel(logging.WARNING)
#    ch.setFormatter(formatter)
#    logger.addHandler(ch)
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
    cfun = inspect.getouterframes(inspect.currentframe())[1][0].f_code.co_name 
    log = logging.getLogger(name+'.'+cfun)
    if level == 'info':
        log.info(msg)
    elif level == 'debug':
        log.debug(msg)
    elif level == 'warning' or level == 'warn':
        log.warning(msg)
    elif level == 'critical':
        log.critical(msg)
    elif level == 'error':
        log.error(msg)
        closeLogger()
        raise e(msg)
    else:
        raise Exception('level input not understood, acceptable values are' 
                        '"debug","info","warning","error","critical"')
    if pri:
        print msg
    if close: #close logger
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

logger=setLogger()
logger.info('Imported Detex')





