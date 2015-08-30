# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:57:38 2015

@author: Derrick

Symple module to handle all of Detex logging

"""

import logging, os
reload(logging) #reload to reconfigure default ipython logging behavior (hopefully this doesnt screw up anything)
cwd=os.getcwd()

## Configure logger to be used across all of Detex

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(cwd,'detex.log'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

logger.info('Imported Detex')


