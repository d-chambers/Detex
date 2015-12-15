# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:58:02 2015

@author: derrick
test bug in subsample extrap that can cause misalignment
"""

import detex
# no subsamp extrap
cl1 = detex.createCluster(subSamp=False)
ss1 = detex.createSubSpace(clust=cl1)
ss1.attachPickTimes()
ss1.SVD(threshold=.1)
ss1.plotFracEnergy()

# subsamp extrap
cl2 = detex.createCluster(subSamp=False)
ss2 = detex.createSubSpace(clust=cl1)
ss2.attachPickTimes()
ss2.SVD(threshold=.1)
ss2.plotFracEnergy()