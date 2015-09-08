# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#   Author: Derrick Chambers
#------------------------------------------------------------------------------
"""
deTex: A Python Toolbox for running subspace detections.
==================================================================
"""
#imports
import getdata, results, util, arc, pandas_dbms, streamPick, Picks, subspace, ANF, version, inspect,os,logging

# Set shortcuts for lazy people 

getAllData=getdata.getAllData

createCluster=subspace.createCluster

createSubSpace=subspace.createSubSpace

detResults=results.detResults

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
    reload(logging) #reload to reconfigure default ipython logging behavior (hopefully this doesnt screw up anything)
    cwd=os.getcwd()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create file handler which logs even debug messages
    if makeLog:
        fh = logging.FileHandler(os.path.join(cwd,filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
    

#define basic logging function

def log(name,msg,level='info',pri=False):
    """
    Wrapper to log various events in detex. By default only events with level "warning" and above are printed to screen
    """
    cfun=inspect.getouterframes(inspect.currentframe())[1][0].f_code.co_name # get name of function that called this one
    log=logging.getLogger(name+'.'+cfun)
    if level=='info':
        log.info(msg)
    elif level=='debug':
        log.debug(msg)
    elif level=='warning' or 'warn':
        log.warning(msg)
    elif level=='critical':
        log.critical(msg)
    elif level=='error':
        log.error(msg)
    else:
        raise Exception('level input not understood, acceptable values are "debug","info","warning","error","critical"')
    if pri:
        print msg

logger=setLogger()
logger.info('Imported Detex')





