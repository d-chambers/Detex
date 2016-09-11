# -*- coding: utf-8 -*-
"""
Created on Tue Jul 08 21:24:18 2014

@author: Derrick
Module containing import detex classes
"""
# python 2 and 3 compatibility imports
from __future__ import print_function, absolute_import, unicode_literals, division

import json
import numbers
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
import scipy
from six import string_types

import detex

try:  # python 2/3 compat
    import cPickle
except ImportError:
    import pickle as cPickle
import itertools
import copy
import colorsys
from struct import pack
import PyQt4
import sys
from obspy import UTCDateTime as UTC

from scipy.cluster.hierarchy import dendrogram, fcluster
from detex.detect import _SSDetex

pd.options.mode.chained_assignment = None  # mute setting copy warning


# warnings.filterwarnings('error') #uncomment this to make all warnings errors

# lines for backward compat.


class ClusterStream(object):
    """
    A container for multiple cluster objects, should only be called with
    detex.construct.createCluster
    """

    def __init__(self, trdf, temkey, stakey, fetcher, eventList, ccReq, filt,
                 decimate, trim, fileName, eventsOnAllStations, enforceOrigin):
        self.__dict__.update(locals())  # Instantiate all input variables
        self.ccReq = None  # set to None because it can vary between stations
        self.clusters = [0] * len(trdf)
        self.stalist = trdf.Station.values.tolist()  # station lists 
        self.stalist2 = [x.split('.')[1] for x in self.stalist]
        self.filename = fileName
        self.eventCodes = self._makeCodes()
        for num, row in trdf.iterrows():
            if not eventsOnAllStations:
                evlist = row.Events
            else:
                evlist = eventList
            self.clusters[num] = Cluster(self, row.Station, temkey, evlist,
                                         row.Link, ccReq, filt, decimate, trim,
                                         row.CCs)

    def writeSimpleHypoDDInput(self, fileName='dt.cc', coef=1, minCC=.35):
        """
        Create a hypoDD cross correlation file (EG dt.cc), assuming the lag 
        times are pure S times (should be true if S amplitude is dominant)

        Parameters
        ----------
        fileName : str
            THe path to the new file to be created
        coef : float or int
            The exponential coefficient to apply to the correlation
           coefficient when creating file,usefull to down-weight lower cc
            values
        minCC : float
            The

       """
        if not self.enforceOrigin:
            msg = ('Sample Lags are not meaningful unless origin times are '
                   'enforced on each waveform. re-run detex.subspace.'
                   'createCluster with enforceOrigin=True')
            detex.log(__name__, msg, level='error')
        fil = open(fileName, 'wb')
        # required number of zeros for numbering all events
        reqZeros = int(np.ceil(np.log10(len(self.temkey))))
        for num1, everow1 in self.temkey.iterrows():
            for num2, everow2 in self.temkey.iterrows():
                if num1 >= num2:  # if autocors or redundant pair then skip
                    continue
                ev1, ev2 = everow1.NAME, everow2.NAME
                header = self._makeHeader(num1, num2, reqZeros)
                count = 0
                for sta in self.stalist:  # iter through each station
                    Clu = self[sta]
                    try:
                        # find station specific index for event1
                        ind1 = np.where(np.array(Clu.key) == ev1)[0][0]
                        ind2 = np.where(np.array(Clu.key) == ev2)[0][0]
                    except IndexError:  # if either event is not in index
                        msg = ('%s or %s not found on station %s' %
                               (ev1, ev2, sta))
                        detex.log(__name__, msg, level='warning', pri=True)
                        continue
                    # get data specific to this station
                    trdf = self.trdf[self.trdf.Station == sta].iloc[0]
                    sr1 = trdf.Stats[ev1]['sampling_rate']
                    sr2 = trdf.Stats[ev2]['sampling_rate']
                    if sr1 != sr2:
                        msg = 'Samp. rates not equal on %s and %s' % (ev1, ev2)
                        detex.log(__name__, msg, level='error')
                    else:
                        sr = sr1
                    Nc1, Nc2 = trdf.Stats[ev1]['Nc'], trdf.Stats[ev2]['Nc']
                    if Nc1 != Nc2:
                        msg = ('Num. of channels not equal for %s and %s on %s'
                               % (ev1, ev2))
                        detex.log(__name__, msg, level='warning', pri=True)
                        continue
                    else:
                        Nc = Nc1
                    cc = trdf.CCs[ind2][ind1]  # get cc value
                    if np.isnan(cc):  # get other part of symetric matrix
                        try:
                            cc = trdf.CCs[ind1][ind2]
                        except KeyError:
                            msg = ('%s - %s pair not in CCs matrix' %
                                   (ev1, ev2))
                            detex.log(__name__, msg, level='warning', pri=True)
                            continue
                        if np.isnan(cc):  # second pass required
                            msg = ('%s - %s pair returning NaN' %
                                   (ev1, ev2))
                            detex.log(__name__, msg, level='warning', pri=True)
                            continue
                    if cc < minCC:
                        continue
                    lagsamps = trdf.Lags[ind2][ind1]
                    subsamps = trdf.Subsamp[ind2][ind1]
                    if np.isnan(lagsamps):  # if lag from other end of mat
                        lagsamps = -trdf.Lags[ind1][ind2]
                        subsamps = trdf.Subsamp[ind1][ind2]
                    lags = lagsamps / (sr * Nc) + subsamps
                    obsline = self._makeObsLine(sta, lags, cc ** coef)
                    if isinstance(obsline, string_types):
                        count += 1
                        if count == 1:
                            fil.write(header + '\n')
                        fil.write(obsline + '\n')
        fil.close()

    def _makeObsLine(self, sta, dt, cc, pha='S'):
        line = '%s %0.4f %0.4f %s' % (sta, dt, cc, pha)
        return line

    def _makeHeader(self, num1, num2, reqZeros):
        fomatstr = '{:0' + "{:d}".format(reqZeros) + 'd}'
        # assume cross corr and cat origins are identical
        head = '# ' + fomatstr.format(num1) + \
               ' ' + fomatstr.format(num2) + ' ' + '0.0'
        return head

    def _makeCodes(self):
        evcodes = {}
        for num, row in self.temkey.iterrows():
            evcodes[num] = row.NAME
        return evcodes

    def updateReqCC(self, reqCC):
        """
        Updates the required correlation coefficient for clusters to form on
        all stations or individual stations. 
        Parameters
        --------------
        reqCC : float (between 0 and 1), or dict of reference keys and floats
            if reqCC is a float the required correlation coeficient for 
            clusters to form will be set to reqCC on all stations. 
            If dict keys must be indicies for each cluster object (IE net.sta,
            sta, or int index) and values are the reqCC for that station.
        Notes 
        ---------------
        The Cluster class also have a similar method that can be more 
        intuitive to use, as in the tutorial
        """
        if isinstance(reqCC, float):
            if reqCC < 0 or reqCC > 1:
                msg = 'reqCC must be between 0 and 1'
                detex.log(__name__, msg, level='error')
            for cl in self.clusters:
                cl.updateReqCC(reqCC)
        elif isinstance(reqCC, dict):
            for key in reqCC.keys():
                self[key].updateReqCC(reqCC[key])
        elif isinstance(reqCC, list):
            for num, cc in enumerate(reqCC):
                self[num].updateReqCC(cc)

    def printAtr(self):  # print out basic attributes used to make cluster
        for cl in self.clusters:
            cl.printAtr()

    def dendro(self, **kwargs):
        """
        Create dendrograms for each station
        """
        for cl in self.clusters:
            cl.dendro(**kwargs)

    def simMatrix(self, groupClusts=False, savename=False, returnMat=False,
                  **kwargs):
        """
        Function to create similarity matrix of each event pair

        Parameters
        -------
        groupClusts : bool
            If True order by clusters on the simmatrix with the singletons 
            coming last
        savename : str or False
            If not False, a path used by plt.savefig to save the current 
            figure. The extension is necesary for specifying format. 
            See plt.savefig for details
        returnMat : bool
            If true return the similarity matrix
        """
        out = []
        for cl in self.clusters:
            dout = cl.simMatrix(groupClusts, savename, returnMat, **kwargs)
            out.append(dout)

    def plotEvents(self, projection='merc', plotSingles=True, **kwargs):
        """
        Plot the event locations for each station using basemap. Calls the 
        plotEvents method of the Cluster class, see its docs for accepted 
        kwargs.

        Parameters
        ---------
        projection : str
            The pojection type to pass to basemap
        plotSingles : bool
            If True also plot the singletons (events that dont cluster)
        
        Notes
        -------
        
        kwargs are passed to basemap
        
        If no working installation of basemap is found an ImportError will 
        be raised. See the following URL for tips on installing it:
        http://matplotlib.org/basemap/users/installing.html, good luck!
        """
        for cl in self.clusters:
            cl.plotEvents(projection, plotSingles, **kwargs)

    def write(self):  # uses pickle to write class to disk
        """
        Write instance to file (name is the filename attribute)
        """
        msg = 'writing ClusterStream instance as %s' % self.filename
        detex.log(__name__, msg, level='info', pri=True)
        cPickle.dump(self, open(self.filename, 'wb'))

    def __getitem__(self, key):  # allows indexing of children Cluster objects
        if isinstance(key, int):
            return self.clusters[key]
        elif isinstance(key, string_types):
            if len(key.split('.')) == 1:
                return self.clusters[self.stalist2.index(key)]
            elif len(key.split('.')) == 2:
                return self.clusters[self.stalist.index(key)]
        else:
            msg = ('indexer must either be an int or str of sta.net or sta'
                   ' you passed %s' % key)
            detex.log(__name__, msg, level='error')

    def __len__(self):
        return len(self.clusters)

    def __repr__(self):
        outstr = 'SSClusterStream with %d stations ' % (len(self.stalist))
        return outstr


class Cluster(object):
    def __init__(self, clustStream, station, temkey, eventList, link, ccReq,
                 filt, decimate, trim, DFcc):

        # instantiate a few needed varaibles (not all to save space)
        self.link = link
        self.DFcc = DFcc
        self.station = station
        self.temkey = temkey
        self.key = eventList
        self.updateReqCC(ccReq)
        self.trim = trim
        self.decimate = decimate
        self.nonClustColor = '0.6'  # use a grey of 0.6 for singletons

    def updateReqCC(self, newccReq):
        """
        Function to update the required correlation coeficient for 
        this station
        Parameters
        -------------
        newccReq : float (between 0 and 1)
            Required correlation coef
        """
        if newccReq < 0. or newccReq > 1.:
            msg = 'Parameter ccReq must be between 0 and 1'
            detex.log(__name__, msg, level='error')
        self.ccReq = newccReq
        self.dflink, serclus = self._makeDFLINK(truncate=False)
        # get events that actually cluster (filter out singletons)
        dfcl = self.dflink[self.dflink.disSim <= 1 - self.ccReq]
        # sort putting highest links in cluster on top
        dfcl.sort_values(by='disSim', inplace=True, ascending=False)
        dfcl.reset_index(inplace=True, drop=True)
        dftemp = dfcl.copy()
        clustlinks = {}
        clustEvents = {}
        clnum = 0
        while len(dftemp) > 0:
            ser = dftemp.iloc[0]
            ndf = dftemp[[set(x).issubset(ser.II) for x in dftemp.II]]
            clustlinks[clnum] = ndf.clust
            valset = set([y for x in ndf.II.values for y in x])
            clustEvents[clnum] = list(valset)
            dftemp = dftemp[~dftemp.index.isin(ndf.index)]
            clnum += 1
        self.clustlinks = clustlinks
        self.clusts = [[self.key[y] for y in clustEvents[x]]
                       for x in clustEvents.keys()]
        keyset = set(self.key)
        clustset = set([y for x in self.clusts for y in x])
        self.singles = list(keyset.difference(clustset))
        self.clustcount = np.sum([len(x) for x in self.clusts])
        self.clustColors = self._getColors(len(self.clusts))
        msg = ('ccReq for station %s updated to ccReq=%1.3f' %
               (self.station, newccReq))
        detex.log(__name__, msg, level='info', pri=True)

    def _getColors(self, numClusts):
        """
        See if there are enough defualt colors for the clusters, if not
        Generate N unique colors (that probably dont look good together)
        """
        clustColorsDefault = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        # if there are enough default python colors use them
        if numClusts <= len(clustColorsDefault):
            return clustColorsDefault[:numClusts]
        else:  # if not generaete N unique colors
            colors = []
            for i in np.arange(0., 360., 360. / numClusts):
                hue = i / 360.
                lightness = (50 + np.random.rand() * 10) / 100.
                saturation = (90 + np.random.rand() * 10) / 100.
                cvect = colorsys.hls_to_rgb(hue, lightness, saturation)
                rgb = [int(x * 255) for x in cvect]
                # covnert to hex code
                colors.append('#' + pack("BBB", *rgb).encode('hex'))
            return colors

    def _makeColorDict(self, clustColors, nonClustColor):
        if len(self.clusts) < 1:
            colorsequence = clustColors
        # if not enough colors repeat color matrix
        elif float(len(clustColors)) / len(self.clusts) < 1:
            colorsequence = clustColors * \
                            int(np.ceil((float(len(self.clusts)) / len(clustColors))))
        else:
            colorsequence = clustColors
        # unitialize color list with default color
        color_list = [nonClustColor] * 3 * len(self.dflink)
        for a in range(len(self.clusts)):
            for b in self.clustlinks[a]:
                color_list[int(b)] = colorsequence[a]
        return color_list

    def _makeDFLINK(self, truncate=True):  # make the link dataframe
        N = len(self.link)
        # append cluster numbers to link array
        link = np.append(self.link, np.arange(N + 1, N + N + 1).reshape(N, 1), 1)
        if truncate:  # truncate after required coeficient
            linkup = link[link[:, 2] <= 1 - self.ccReq]
        else:
            linkup = link
        T = fcluster(link[:, 0:4], 1 - self.ccReq, criterion='distance')
        serclus = pd.Series(T)

        clusdict = pd.Series([np.array([x]) for x in np.arange(
            0, N + 1)], index=np.arange(0, N + 1))
        for a in range(len(linkup)):
            clusdict[int(linkup[a, 4])] = np.append(
                clusdict[int(linkup[a, 0])], clusdict[int(linkup[a, 1])])
        columns = ['i1', 'i2', 'disSim', 'num', 'clust']
        dflink = pd.DataFrame(linkup, columns=columns)
        if len(dflink) > 0:
            dflink['II'] = list
        else:
            msg = 'No events cluster with corr coef = %1.3f' % self.ccReq
            detex.log(__name__, msg, level='info', pri=True)
        for a in dflink.iterrows():  # enumerate cluster contents
            ar1 = list(np.array(clusdict[int(a[1].i1)]))
            ar2 = list(np.array(clusdict[int(a[1].i2)]))
            dflink['II'][a[0]] = ar1 + ar2
        return dflink, serclus

    # creates a basic dendrogram plot
    def dendro(self, hideEventLabels=True, show=True, saveName=False,
               legend=True, return_axis=False, color_list=None, **kwargs):
        """
        Function to plot dendrograms of the clusters

        Parameters
        -----
        hideEventLabels : bool
            turns x axis labeling on/off. Better set to false 
            if many events are in event pool
        show : bool
            If true call plt.show
        saveName : str or False
            path to save figure. Extention denotes format. See plt.savefig 
            for details
        legend : bool
            If true plot a legend on the side of the dendrogram
        return_axis : bool
            If True return axis for more plotting
        color_list : None or list of matplotlib colors
            Colors to plot dendrogram
        Note 
        ----------
        kwargs are passed to scipy.cluster.hierarchy.dendrogram, see docs
        for acceptable arguments and descriptions
        """
        # Get color schemes
        if color_list is None:
            clust_colors = self.clustColors
        else:
            clit = itertools.cycle(color_list)
            clen = len(self.clustColors)
            clust_colors = list(itertools.islice(clit, 0, clen))

        color_list = self._makeColorDict(clust_colors, self.nonClustColor)

        for a in range(len(self.clusts)):
            plt.plot([], [], '-', color=self.clustColors[a])
        plt.plot([], [], '-', color=self.nonClustColor)
        dendrogram(self.link, color_threshold=1 - self.ccReq, count_sort=True,
                   link_color_func=lambda x: color_list[x], **kwargs)
        ax = plt.gca()
        if legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend([str(x) for x in range(1, len(self.clusts) + 1)] +
                      ['N/A'], loc='center left', bbox_to_anchor=(1, .5),
                      title='Clusters')
        ax.set_ylim([0, 1])
        if hideEventLabels:
            ax.set_xticks([])
        plt.xlabel('Events')
        plt.ylabel('Dissimilarity')
        plt.title(self.station)
        if return_axis:
            return ax
        elif saveName:
            plt.savefig(saveName, **kwargs)
        elif show:
            plt.show()

    def plotEvents(self, projection='merc', plotSingles=True, **kwargs):
        """
        Plot the event locations for each station using basemap. Calls the 
        plotEvents method of the Cluster class, see its docs for accepted 
        kwargs.

        Parameters
        ---------
        projection : str
            The pojection type to pass to basemap
        plotSingles : bool
            If True also plot the singletons (events that dont cluster)
        
        Notes
        -------
        kwargs are passed to basemap
        If no working installation of basemap is found an ImportError will 
        be raised. See the following URL for tips on installing it:
        http://matplotlib.org/basemap/users/installing.html, good luck!
        """
        # TODO make dot size scale with magnitudes
        # make sure basemap is installed
        try:
            from mpl_toolkits.basemap import Basemap
        except ImportError:
            msg = 'mpl_toolskits basemap not installed, cant plot'
            detex.log(__name__, msg, level='error', e=ImportError)
        # init figures and get limits
        fig_map, emap, horrange = self._init_map(Basemap, projection, kwargs)
        zmin, zmax, zscale = self._get_z_scaling(horrange)
        fig_lat = self._init_profile_figs(zmin, zmax, zscale)
        fig_lon = self._init_profile_figs(zmin, zmax, zscale)
        # seperate singletons from clustered events
        cl_dfs, sing_df = self._get_singletons_and_clusters()
        self._plot_map_view(emap, fig_map, horrange, cl_dfs, sing_df)
        self._plot_profile_view(zmin, zmax, zscale, fig_lat, fig_lon, cl_dfs,
                                sing_df, emap)

    def _init_map(self, Basemap, projection, kwargs):
        """
        Function to setup the map figure with basemap returns the 
        figure instance and basemap instance and horizontal range of plot
        """
        map_fig = plt.figure()

        # get map bounds       
        latmin = self.temkey.LAT.min()
        latmax = self.temkey.LAT.max()
        lonmin = self.temkey.LON.min()
        lonmax = self.temkey.LON.max()
        # create buffers so there is a slight border with no events around map
        latbuff = abs((latmax - latmin) * 0.1)
        lonbuff = abs((lonmax - lonmin) * 0.1)
        # get the total horizontal distance of plot in km
        totalxdist = obspy.core.util.geodetics.gps2DistAzimuth(
            latmin, lonmin, latmin, lonmax)[0] / 1000
        # init projection
        emap = Basemap(projection=projection,
                       lat_0=np.mean([latmin, latmax]),
                       lon_0=np.mean([lonmin, lonmax]),
                       resolution='h',
                       area_thresh=0.1,
                       llcrnrlon=lonmin - lonbuff,
                       llcrnrlat=latmin - latbuff,
                       urcrnrlon=lonmax + lonbuff,
                       urcrnrlat=latmax + latbuff,
                       **kwargs)
        # draw scale
        emap.drawmapscale(lonmin, latmin, lonmin, latmin, totalxdist / 4.5)

        # get limits in projection
        xmax, xmin, ymax, ymin = emap.xmax, emap.xmin, emap.ymax, emap.ymin
        horrange = max((xmax - xmin), (ymax - ymin))  # horizontal range

        # get maximum degree distance for setting scalable ticks
        latdi, londi = [abs(latmax - latmin), abs(lonmax - lonmin)]
        maxdeg = max(latdi, londi)
        parallels = np.arange(0., 80, maxdeg / 4)
        emap.drawparallels(parallels, labels=[1, 0, 0, 1])
        meridians = np.arange(10., 360., maxdeg / 4)
        mers = emap.drawmeridians(meridians, labels=[1, 0, 0, 1])
        for m in mers:  # rotate meridian labels
            try:
                mers[m][1][0].set_rotation(90)
            except:
                pass

        plt.title('Clusters on %s' % self.station)
        return map_fig, emap, horrange

    def _init_profile_figs(self, zmin, zmax, zscale):
        """
        init figs for plotting the profiles of the events
        """
        # init profile figures
        profile_fig = plt.figure()
        z1 = zmin * zscale
        z2 = zmax * zscale
        tickfor = ['%0.1f' % x1 for x1 in np.linspace(zmin, zmax, 10)]
        plt.yticks(np.linspace(z1, z2, 10), tickfor)
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.ylabel('Depth (km)')
        return profile_fig

    def _get_z_scaling(self, horrange):
        """
        Return z limits and scale factors
        """
        zmin, zmax = self.temkey.DEPTH.min(), self.temkey.DEPTH.max()
        zscale = horrange / (zmax - zmin)
        return zmin, zmax, zscale

    def _get_singletons_and_clusters(self):
        """        
        get dataframes of clustered events and singletons
        Note: cl_dfs is a list of dfs whereas sing_df is just a df
        """
        cl_dfs = [self.temkey[self.temkey.NAME.isin(x)] for x in self.clusts]
        sing_df = self.temkey[self.temkey.NAME.isin([x for x in self.singles])]
        return cl_dfs, sing_df

    def _plot_map_view(self, emap, map_fig, horrange, cl_dfs, sing_df):
        """
        plot the map figure
        """
        plt.figure(map_fig.number)  # set to map figure
        # plot singles
        x, y = emap(sing_df.LON.values, sing_df.LAT.values)
        emap.plot(x, y, '.', color=self.nonClustColor, ms=6.0)
        for clnum, cl in enumerate(cl_dfs):
            x, y = emap(cl.LON.values, cl.LAT.values)
            emap.plot(x, y, '.', color=self.clustColors[clnum])

    def _plot_profile_view(self, zmin, zmax, zscale, fig_lat, fig_lon, cl_df,
                           sing_df, emap):
        """
        plot the profile view
        """
        x_sing, y_sing = emap(sing_df.LON.values, sing_df.LAT.values)
        # plot singletons
        nccolor = self.nonClustColor
        plt.figure(fig_lon.number)
        plt.plot(x_sing, sing_df.DEPTH * zscale, '.', color=nccolor, ms=6.0)
        plt.xlabel('Longitude')
        plt.figure(fig_lat.number)
        plt.plot(y_sing, sing_df.DEPTH * zscale, '.', color=nccolor, ms=6.0)
        plt.xlabel('Latitude')
        # plot clusters
        for clnum, cl in enumerate(cl_df):
            ccolor = self.clustColors[clnum]
            x, y = emap(cl.LON.values, cl.LAT.values)
            plt.figure(fig_lon.number)
            plt.plot(x, cl.DEPTH * zscale, '.', color=ccolor)
            plt.figure(fig_lat.number)
            plt.plot(y, cl.DEPTH * zscale, '.', color=ccolor)
        # set buffers so nothing plots right on edge
        for fig in [fig_lat, fig_lon]:
            plt.figure(fig.number)
            xlim = plt.xlim()
            xdist = abs(max(xlim) - min(xlim))
            plt.xlim(xlim[0] - xdist * .1, xlim[1] + xdist * .1)
            ylim = plt.ylim()
            ydist = abs(max(xlim) - min(xlim))
            plt.ylim(ylim[0] - ydist * .1, ylim[1] + ydist * .1)

    def simMatrix(self, groupClusts=False, savename=None, returnMat=False,
                  return_ax=False, plot_title=True, **kwargs):
        """
        Function to create basic similarity matrix of the values 
        in the cluster object

        Parameters
        -------
        groupClusts : boolean
            If True order by clusters on the simmatrix with the 
            singletons coming last
        savename : str or None
            If str, a path used by plt.savefig to save the current
            figure. The extension is necessary for specifying format. See
            plt.savefig for details
        returnMat : boolean
            If true return the similarity matrix
        return_ax : bool
            If True return the axis the figure is plotted in
        plot_title : bool, str
            If True plot the title  as the station, if str plot str
            for the title

        kwargs are passed to plt.figure and plt.save
        """
        if groupClusts:  # if grouping clusters together
            clusts = copy.deepcopy(self.clusts)  # get cluster list
            clusts.append(self.singles)  # add singles list at end
            eveOrder = list(itertools.chain.from_iterable(clusts))
            indmask = {
                num: list(self.key).index(eve) for num,
                                                   eve in enumerate(eveOrder)}  # create a mask for the order
        else:
            # blank index mask if not
            indmask = {x: x for x in range(len(self.key))}
        plt.figure(**kwargs)
        le = self.DFcc.columns.values.max()
        mat = np.zeros((le + 1, le + 1))
        for a in range(le + 1):
            for b in range(le + 1):
                if a == b:
                    mat[a, b] = 1
                else:
                    # new a and b coords based on mask
                    a1, b1 = indmask[a], indmask[b]
                    gi = max(a1, b1)
                    li = min(a1, b1)
                    mat[a, b] = self.DFcc.loc[li, gi]
                    mat[b, a] = self.DFcc.loc[li, gi]

        # cmap = mpl.colors.LinearSegmentedColormap.from_list(
        #     'my_colormap', ['blue', 'red'], 256)
        cmap = plt.cm.YlOrRd
        img = plt.imshow(
            mat,
            interpolation='nearest',
            cmap=cmap,
            origin='upper',
            vmin=0,
            vmax=1)
        plt.clim(0, 1)
        plt.grid(True, color='white')
        plt.colorbar(img, cmap=cmap, label='Correlation Coefficient')
        if plot_title:
            if isinstance(plot_title, string_types):
                plt.title(plot_title)
            else:
                plt.title(self.station)
        plt.xlabel('Event Number')
        plt.ylabel('Event Number')

        if savename is not None:
            plt.savefig(savename, bbox_inches='tight', **kwargs)
        if returnMat:
            return mat

    def write(self):  # uses pickle to write class to disk
        cPickle.dump(self, open(self.filename, 'wb'))

    def printAtr(self):  # print out basic attributes used to make cluster
        print('%s Cluster' % self.station)
        print('%d Events cluster out of %d' %
              (self.clustcount, len(self.singles) + self.clustcount))
        print('Total number of clusters = %d' % len(self.clusts))
        print('Required Cross Correlation Coeficient = %.3f' % self.ccReq)

    def __getitem__(self, index):  # allow indexing
        return self.clusts[index]

    def __iter__(self):  # make class iterable
        return iter(self.clusts)

    def __len__(self):
        return len(self.clusts)


# def __repr__(self):
#        self.printAtr()
#        return ''


class SubSpace(object):
    """ Class used to hold subspaces for detector
    Holds both subspaces (as defined from the SScluster object) and 
    single event clusters, or singles
    """

    def __init__(self, singlesDict, subSpaceDict, cl, dtype, Pf, cfetcher):
        self.cfetcher = cfetcher
        self.clusters = cl
        self.subspaces = subSpaceDict
        self.singles = singlesDict
        self.singletons = singlesDict
        self.dtype = dtype
        self.Pf = Pf
        self.ssStations = self.subspaces.keys()
        self.singStations = self.singles.keys()
        self.Stations = list(set(self.ssStations) | set(self.singStations))
        self.Stations.sort()
        self._stakey2 = {x: x for x in self.ssStations}
        self._stakey1 = {x.split('.')[1]: x for x in self.ssStations}
        self.pick_times = None

    ################################ Validate Cluster functions

    def validateClusters(self):
        """
        Method to check for misaligned waveforms and discard those that no 
        longer meet the required correlation coeficient for each cluster. 
        See Issue 25 (www.github.com/d-chambers/detex) for why this might 
        be useful.
        """
        msg = 'Validating aligned (and trimmed) waveforms in each cluster'
        detex.log(__name__, msg, level='info', pri=True)
        for sta in self.subspaces.keys():
            subs = self.subspaces[sta]
            c = self.clusters[sta]
            ccreq = c.ccReq
            for clustNum, row in subs.iterrows():
                stKeys = row.SampleTrims.keys()
                # get trim times if defined
                if 'Starttime' in stKeys and 'Endtime' in stKeys:
                    start = row.SampleTrims['Starttime']
                    stop = row.SampleTrims['Endtime']
                else:
                    start = 0
                    stop = -1
                for ev1num, ev1 in enumerate(row.Events[:-1]):
                    ccs = []  # blank list for storing ccs of aligned WFs
                    for ev2 in row.Events[ev1num + 1:]:
                        t = row.AlignedTD[ev1][start: stop]
                        s = row.AlignedTD[ev2][start: stop]
                        maxcc = detex.construct.fast_normcorr(t, s)
                        ccs.append(maxcc)
                    if len(ccs) > 0 and max(ccs) < ccreq:
                        msg = (('%s fails validation check or is ill-aligned '
                                'on station %s, removing') % (ev1, row.Station))
                        detex.log(__name__, msg, pri=True)
                        self._removeEvent(sta, ev1, clustNum)
        msg = 'Finished validateCluster call'
        detex.log(__name__, msg, level='info', pri=True)

    def _removeEvent(self, sta, event, clustNum):
        """
        Function to remove an event from a SubSpace instance
        """
        # remove from eventList
        srow = self.subspaces[sta].loc[clustNum]
        srow.Events.remove(event)
        srow.AlignedTD.pop(event, None)

    ################################ SVD Functions

    def SVD(self, selectCriteria=2, selectValue=0.9, conDatNum=100,
            threshold=None, normalize=False, useSingles=True,
            validateWaveforms=True, backupThreshold=None, **kwargs):
        """
        Function to perform SVD on the alligned waveforms and select which 
        of the SVD basis are to be used in event detection. Also assigns 
        a detection threshold to each subspace-station pair.

        Parameters
        ----------------

        selctionCriteria : int, selectValue : number
            selectCriteria is the method for selecting which basis vectors 
            will be used as detectors. selectValue depends on selectCriteria
            Valid options are:

            0 - using the given Pf, find number of dimensions to maximize 
            detection probability !!! NOT YET IMPLIMENTED!!!
                    selectValue - Not used
                    (Need to find a way to use the doubly-non central F 
                    distribution in python)

            1 - Failed implementation, not supported

            2 - select basis number based on an average fractional signal 
            energy captured (see Figure 8 of Harris 2006). Then calculate
            an empirical distribution of the detection statistic by running
            each subspace over random continuous data with no high amplitude
            signals (see getFAS method). A beta distribution is then fit to
            the data and the DS value that sets the probability of false 
            detection to the Pf defined in the subspace instance is selected
            as the threshold.            
                    selectValue - Average fractional energy captured, 
                    can range from 0 (use no basis vectors) to 1
                    (use all basis vectors). A value between 0.75 and 0.95
                    is recommended. 
                    
            3 - select basis number based on an average fractional signal 
            energy captured (see Figure 8 of Harris 2006).
            Then set detection threshold to a percentage of the minimum 
            fractional energy captured. This method is a bit quick and dirty
            but ensures all events in the waveform pool will be detected.
                select value is a fraction representing the fraction of 
                the minum fractional energy captured (between 0 and 1).

            4 - use a user defined number of basis vectors, beginning with the 
            most significant (Barrett and Beroza 2014 use first two basis
            vectors as an "empirical" subspace detector). Then use the same 
            technique in method one to set threshold
                    selectValue - can range from 0 to number of events in 
                    subspace, if selectValue is greater than number of events 
                    all events are used

        conDatNum : int
            The number of continuous data chunks to use to estimate the 
            effective dimension of the signal space or to estimate the null
            distribution. Used if selectCriteria == 1,2,4
        threshold : float or None
            Used to set each subspace at a user defined threshold. If any 
            value is set it overrides any of the previously defined methods and
            avoids estimating the effective dimension of representation or 
            distribution of the null space. Can be useful if problems arise
            in the false alarm statistic calculation
        normalize : bool
            If true normalize the amplitude of all the training events before 
            preforming the SVD. Keeps higher amplitude events from dominating
            the SVD vectors but can over emphasize noise. Haris 2006 recomends
            using normalization but the personal experience of the author has
            found normalization can increase the detector's propensity to 
            return false detections.
        useSingles : bool
            If True also calculate the thresholds for singles
        validateWaveforms : bool
            If True call the validateClusters method before the performing SVD
            to make sure each trimed aligned waveform still meets the
            required correlation coeficient. Any waveforms that do not will
            be discarded. 
        backupThreshold : None or float
            A backup threshold to use if approximation fails. Typically,
            using the default detex settings, a reasonable value would be
            0.25
            
        kwargs are passed to the getFAS call (if used)
        """

        # make sure user defined options are kosher
        self._checkSelection(selectCriteria, selectValue, threshold)
        # Iterate through all subspaces defined by stations
        for station in self.ssStations:
            for ind, row in self.subspaces[station].iterrows():
                self.subspaces[station].UsedSVDKeys[ind] = []
                svdDict = {}  # initialize dict to put SVD vectors in
                keys = sorted(row.Events)
                arr, basisLength = self._trimGroups(ind, row, keys, station)
                if basisLength == 0:
                    msg = (('subspace %d on %s is failing alignment and '
                            'trimming, deleting it') % (ind, station))
                    detex.log(__name__, msg, level='warn')
                    self._drop_subspace(station, ind)
                    continue
                if normalize:
                    arr = np.array([x / np.linalg.norm(x) for x in arr])
                tparr = np.transpose(arr)
                # perform SVD
                U, s, Vh = scipy.linalg.svd(tparr, full_matrices=False)
                # make dict with sing. value as key and sing. vector as value
                for einum, eival in enumerate(s):
                    svdDict[eival] = U[:, einum]
                # asign Parameters back to subspace dataframes
                self.subspaces[station].SVD[ind] = svdDict  # assign SVD
                fracEnergy = self._getFracEnergy(ind, row, svdDict, U)

                usedBasis = self._getUsedBasis(ind, row, svdDict, fracEnergy,
                                               selectCriteria, selectValue)
                # Add fracEnergy and SVD keys (sing. vals) to main DataFrames
                self.subspaces[station].FracEnergy[ind] = fracEnergy
                self.subspaces[station].UsedSVDKeys[ind] = usedBasis
                self.subspaces[station].SVDdefined[ind] = True
                numBas = len(self.subspaces[station].UsedSVDKeys[ind])
                self.subspaces[station].NumBasis[ind] = numBas
        if len(self.ssStations) > 0:
            self._setThresholds(selectCriteria, selectValue, conDatNum,
                                threshold, basisLength, backupThreshold, kwargs)
        if len(self.singStations) > 0 and useSingles:
            self.setSinglesThresholds(conDatNum=conDatNum, threshold=threshold,
                                      backupThreshold=backupThreshold,
                                      kwargs=kwargs)

    def _drop_subspace(self, station, ssnum):
        """
        Drop a subspace that is misbehaving
        """
        space = self.subspaces[station]
        self.subspaces[station] = space[space.index != int(ssnum)]

    def _trimGroups(self, ind, row, keys, station):
        """
        function to get trimed subspaces if trim times are defined, and 
        return an array of the aligned waveforms for the SVD to act on
        """
        stkeys = row.SampleTrims.keys()
        aliTD = row.AlignedTD
        if 'Starttime' in stkeys and 'Endtime' in stkeys:
            stim = row.SampleTrims['Starttime']
            etim = row.SampleTrims['Endtime']
            if stim < 0:  # make sure stim is not less than 0
                stim = 0
            Arr = np.vstack([aliTD[x][stim:etim] -
                             np.mean(aliTD[x][stim:etim]) for x in keys])
            basisLength = Arr.shape[1]
        else:
            msg = ('No trim times for %s and station %s, try running '
                   'pickTimes or attachPickTimes' % (row.Name, station))

            detex.log(__name__, msg, level='warn', pri=True)
            Arr = np.vstack([aliTD[x] - np.mean(aliTD[x]) for x in keys])
            basisLength = Arr.shape[1]
        return Arr, basisLength

    def _checkSelection(self, selectCriteria, selectValue, threshold):
        """
        Make sure all user defined values are kosher for SVD call
        """
        if selectCriteria in [1, 2, 3]:
            if selectValue > 1 or selectValue < 0:
                msg = ('When selectCriteria==%d selectValue must be a float'
                       ' between 0 and 1' % selectCriteria)
                detex.log(__name__, msg, level='error', e=ValueError)
        elif selectCriteria == 4:
            if selectValue < 0 or not isinstance(selectValue, int):
                msg = ('When selectCriteria==3 selectValue must be an'
                       'integer greater than 0')
                detex.log(__name__, msg, level='error', e=ValueError)
        else:
            msg = 'selectCriteria of %s is not supported' % selectCriteria
            detex.log(__name__, msg, level='error')

        if threshold is not None:
            if not isinstance(threshold, numbers.Number) or threshold < 0:
                msg = 'Unsupported type for threshold, must be None or float'
                detex.log(__name__, msg, level='error', e=ValueError)

    def _getFracEnergy(self, ind, row, svdDict, U):
        """
        calculates the % energy capture for each stubspace for each possible
        dimension of rep. (up to # of events that go into the subspace)
        """
        fracDict = {}
        keys = row.Events
        svales = svdDict.keys()
        svales.sort(reverse=True)
        stkeys = row.SampleTrims.keys()  # dict defining sample trims
        for key in keys:
            aliTD = row.AlignedTD[key]  # aligned waveform for event key
            if 'Starttime' in stkeys and 'Endtime' in stkeys:
                start = row.SampleTrims['Starttime']  # start of trim in samps
                end = row.SampleTrims['Endtime']  # end of trim in samps
                aliwf = aliTD[start: end]
            else:
                aliwf = aliTD
            Ut = np.transpose(U)  # transpose of basis vects
            # normalized dot product (mat. mult.) 
            normUtAliwf = scipy.dot(Ut, aliwf) / scipy.linalg.norm(aliwf)
            # add 0% energy capture for dim of 0
            repvect = np.insert(np.square(normUtAliwf), 0, 0)
            # cumul. energy captured for increasing dim. reps
            cumrepvect = [np.sum(repvect[:x + 1]) for x in range(len(repvect))]
            fracDict[key] = cumrepvect  # add cumul. to keys
        # get average and min energy capture, append value to dict
        fracDict['Average'] = np.average([fracDict[x] for x in keys], axis=0)
        fracDict['Minimum'] = np.min([fracDict[x] for x in keys], axis=0)
        return (fracDict)

    def _getUsedBasis(self, ind, row, svdDict, cumFracEnergy,
                      selectCriteria, selectValue):
        """
        function to populate  the keys of the selected SVD basis vectors
        """
        keys = svdDict.keys()
        keys.sort(reverse=True)
        if selectCriteria in [1, 2, 3]:
            # make sure last element is exactly 1
            cumFracEnergy['Average'][-1] = 1.00
            ndim = np.argmax(cumFracEnergy['Average'] >= selectValue)
            selKeys = keys[:ndim]  # selected keys
        if selectCriteria == 4:
            selKeys = keys[:selectValue + 1]
        return selKeys

    def _setThresholds(self, selectCriteria, selectValue, conDatNum,
                       threshold, basisLength, backupThreshold, kwargs={}):
        if threshold > 0:
            for station in self.ssStations:
                subspa = self.subspaces[station]
                for ind, row in subspa.iterrows():
                    self.subspaces[station].Threshold[ind] = threshold

        elif selectCriteria == 1:
            msg = 'selectCriteria 1 currently not supported'
            detex.log(__name__, msg, level='error', e=ValueError)

        elif selectCriteria in [2, 4]:
            # call getFAS to estimate null space dist.
            self.getFAS(conDatNum, **kwargs)
            for station in self.ssStations:
                subspa = self.subspaces[station]
                for ind, row in subspa.iterrows():
                    beta_a, beta_b = row.FAS['betadist'][0:2]
                    # get threshold from beta dist. 
                    # TODO consider implementing other dist. options as well
                    th = scipy.stats.beta.isf(self.Pf, beta_a, beta_b, 0, 1)
                    if th > .9:
                        th, Pftemp = self._approxThld(beta_a, beta_b, station,
                                                      row, self.Pf, 1000, 3,
                                                      backupThreshold)

                        msg = ('Scipy.stats.beta.isf failed with pf=%e, '
                               'approximated threshold to %f with a Pf of %e '
                               'for station %s %s using forward grid search' %
                               (self.Pf, th, Pftemp, station, row.Name))
                        detex.log(__name__, msg, level='warning')
                    self.subspaces[station].Threshold[ind] = th

        elif selectCriteria == 3:
            for station in self.ssStations:
                subspa = self.subspaces[station]
                for ind, row in subspa.iterrows():
                    th = row.FracEnergy['Minimum'][row.NumBasis] * selectValue
                    self.subspaces[station].Threshold[ind] = th

    def setSinglesThresholds(self, conDatNum=50, recalc=False,
                             threshold=None, backupThreshold=None, **kwargs):
        """
        Set thresholds for the singletons (unclustered events) by fitting 
        a beta distribution to estimation of null space

        Parameters
        ----------
        condatNum : int
            The number of continuous data chunks to use to fit PDF
        recalc : boolean
            If true recalculate the the False Alarm Statistics
        threshold : None or float between 0 and 1
            If number, don't call getFAS simply use given threshold
        backupThreshold : None or float
            If approximate a threshold fails then use backupThreshold. If None
            then raise. 
        Note 
        ----------
        Any singles without pick times will not be used. In this way singles 
        can be rejected 
        """
        for sta in self.singStations:
            sing = self.singles[sta]  # singles on station
            sampTrims = self.singles[sta].SampleTrims
            self.singles[sta].Name = ['SG%d' % x for x in range(len(sing))]
            # get singles that have phase picks
            singsAccepted = sing[[len(x.keys()) > 0 for x in sampTrims]]
            self.singles[sta] = singsAccepted
            self.singles[sta].reset_index(inplace=True, drop=True)
        if threshold is None:
            # get empirical dist unless manual threshold is passed
            self.getFAS(conDatNum, useSingles=True,
                        useSubSpaces=False, **kwargs)
        for sta in self.singStations:
            for ind, row in self.singles[sta].iterrows():
                if len(row.SampleTrims.keys()) < 1:  # skip singles with no pick times
                    continue
                if threshold:
                    th = threshold
                else:
                    beta_a, beta_b = row.FAS[0]['betadist'][0:2]
                    th = scipy.stats.beta.isf(self.Pf, beta_a, beta_b, 0, 1)
                    if th > .9:
                        th, Pftemp = self._approxThld(beta_a, beta_b, sta,
                                                      row, self.Pf, 1000, 3,
                                                      backupThreshold)
                        msg = ('Scipy.stats.beta.isf failed with pf=%e, '
                               'approximated threshold to %f with a Pf of %e '
                               'for station %s %s using forward grid search' %
                               (self.Pf, th, Pftemp, sta, row.Name))
                        detex.log(__name__, msg, level='warning')
                self.singles[sta]['Threshold'][ind] = th

    def _approxThld(self, beta_a, beta_b, sta, row, target, numint, numloops,
                    backupThreshold):
        """
        Because scipy.stats.beta.isf can break, if it returns a value near 1 
        when this is obviously wrong initialize grid search algorithm to get 
        close to desired threshold using forward problem which seems to work
        where inverse fails See this bug report: 
        https://github.com/scipy/scipy/issues/4677        
        """
        startVal, stopVal = 0, 1
        loops = 0
        while loops < numloops:
            Xs = np.linspace(startVal, stopVal, numint)
            pfs = np.array([scipy.stats.beta.sf(x, beta_a, beta_b) for x in Xs])
            resids = abs(pfs - target)
            minind = resids.argmin()
            if minind == 0 or minind == numint - 1:
                msg1 = (('Grid search for threshold failing for %s on %s, '
                         'set it manually or use default') % (sta, row.name))
                msg2 = (('Grid search for threshold failing for %s on %s, '
                         'using backup %.2f') % (sta, row.name, backupThreshold))
                if backupThreshold is None:
                    detex.log(__name__, msg1, level='error', e=ValueError)
                else:
                    detex.log(__name__, msg2, level='warn', pri=True)
                    return backupThreshold, target
            bestPf = pfs[minind]
            bestX = Xs[minind]
            startVal, stopVal = Xs[minind - 1], Xs[minind + 1]
            loops += 1
        return bestX, bestPf

    ########################### Visualization Methods

    def plotThresholds(self, conDatNum, xlim=[-.01, .5], **kwargs):
        """
        Function to sample the continuous data and plot the thresholds 
        calculated with the SVD call with a histogram of detex's best 
        estimate of the null space (see getFAS for more details)

        Parameters
        ------
        conDatNum : int
            The number of continuous data chunks to use in the sampling,
            duration of chunks defined in data fetcher
        xlim : list (number, number)
            The x limits on the plot (often it is useful to zoom in around 0)
            
        **kwargs are passed to the getFAS call
        """

        self.getFAS(conDatNum, **kwargs)
        count = 0
        for station in self.ssStations:
            for ind, row in self.subspaces[station].iterrows():
                beta_a, beta_b = row.FAS['betadist'][0:2]
                plt.figure(count)
                plt.subplot(2, 1, 1)
                bins = np.mean(
                    [row.FAS['bins'][1:], row.FAS['bins'][:-1]], axis=0)
                plt.plot(bins, row.FAS['hist'])
                plt.title('Station %s %s' % (station, row.Name))

                plt.axvline(row.Threshold, color='g')
                beta = scipy.stats.beta.pdf(bins, beta_a, beta_b)
                plt.plot(bins, beta * (max(row.FAS['hist']) / max(beta)), 'k')
                plt.title('%s station %s' % (row.Name, row.Station))
                plt.xlim(xlim)
                plt.ylabel('Count')

                plt.subplot(2, 1, 2)
                bins = np.mean(
                    [row.FAS['bins'][1:], row.FAS['bins'][:-1]], axis=0)
                plt.plot(bins, row.FAS['hist'])
                plt.axvline(row.Threshold, color='g')
                plt.plot(bins, beta * (max(row.FAS['hist']) / max(beta)), 'k')
                plt.xlabel('Detection Statistic')
                plt.ylabel('Count')
                plt.semilogy()
                plt.ylim(ymin=10 ** -1)
                plt.xlim(xlim)
                count += 1

    def plotFracEnergy(self, plot_station=None, plot_ss=None):
        """
        Method to plot the fractional energy captured of by the subspace for
        various dimensions of rep. Each event is plotted as a grey dotted
        line, the average as a red solid line, and the chosen degree of rep.
        is plotted as a solid green vertical line.
        Similar to Harris 2006 Fig 8

        Parameters
        ----------
        plot_station : str or None
            If str only plot given station
        plot_ss : str or None
            If str the subspace to plot, else plot all (example name: 'SS0')

        Returns
        -------

        """
        for a, station in enumerate(self.ssStations):
            if plot_station and plot_station not in station:
                continue
            f = plt.figure(a + 1)
            # size fig if only one ss is being plotted or multiple
            if not plot_ss:
                f.set_figheight(1.85 * len(self.subspaces[station]))
            for ind, row in self.subspaces[station].iterrows():
                print('%s: %d' % (row.Name, len(row.AlignedTD)))
                if plot_ss:
                    if row.Name != plot_ss:
                        continue
                else:
                    plt.subplot(len(self.subspaces[station]), 1, ind + 1)
                if not isinstance(row.FracEnergy, dict):
                    msg = 'fractional energy not defiend, call SVD'
                    detex.log(__name__, msg, level='error')
                for event in row.Events:
                    plt.plot(row.FracEnergy[event], '--', color='0.6')
                plt.plot(row.FracEnergy['Average'], 'r')
                plt.axvline(row.NumBasis, 0, 1, color='g')
                plt.ylim([0, 1.1])
                plt.title('Station %s, %s' % (row.Station, row.Name))
            f.subplots_adjust(hspace=.4)
            f.text(0.5, 0.06, 'Dimension of Representation', ha='center')
            f.text(0.04, 0.5, 'Fraction of Energy Captured',
                   va='center', rotation='vertical')
            plt.show()

    def plotAlignedEvents(self, plot_station=None, min_number_events=None):
        """
        Plots the aligned events for each station in each cluster.
        Will trim waveforms if trim times (by pickTimes or attachPickTimes)
        are defined.

        Parameters
        ----------
        plot_station : None or str
            If str the station id (net.sta) to plot
        min_number_events : None or int
            If int the subspace must have this many events or more to plot

        Returns
        -------

        """
        for a, station in enumerate(self.ssStations):
            if plot_station is not None:
                if plot_station not in station:
                    continue
            for ind, row in self.subspaces[station].iterrows():
                plt.figure(figsize=[10, .9 * len(row.Events)])
                # f.set_figheight(1.85 * len(row.Events))
                # plt.subplot(len(self.subspaces[station]), 1, ind + 1)
                events = row.Events
                stKeys = row.SampleTrims.keys()  # sample trim keys
                for evenum, eve in enumerate(events):
                    # plt.subplot(len(self.subspaces[station]), 1, evenum + 1)
                    aliTD = row.AlignedTD[eve]  # aligned wf for event eve
                    if 'Starttime' in stKeys and 'Endtime' in stKeys:
                        start = row.SampleTrims['Starttime']
                        stop = row.SampleTrims['Endtime']
                        aliwf = aliTD[start: stop]
                    else:
                        aliwf = row.AlignedTD[eve]
                    plt.plot(aliwf / (2 * max(aliwf)) + 1.5 * evenum, c='k')
                    plt.xlim([0, len(aliwf)])
                plt.ylim(-1, 1.5 * evenum + 1)
                plt.xticks([])
                plt.yticks([])
                plt.title('Station %s, %s, %d events' % (station, row.Name, len(events)))
                plt.show()

    def plotBasisVectors(self, onlyused=False):
        """
        Plots the basis vectors selected after performing the SVD
        If SVD has not been called will throw error

        Parameters
        ------------
        onlyUsed : bool
            If true only the selected basis vectors will be plotted. See
            SVD for how detex selects basis vectors.
            If false all will be plotted (used in blue, unused in red)
        """
        if not self.subspaces.values()[0].iloc[0].SVDdefined:
            msg = 'SVD not performed, call SVD before plotting basis vectors'
            detex.log(__name__, msg, level='error')
        for subnum, station in enumerate(self.ssStations):
            subsp = self.subspaces[station]

            for ind, row in subsp.iterrows():
                num_wfs = len(row.UsedSVDKeys) if onlyused else len(row.SVD)
                keyz = row.SVD.keys()
                keyz.sort(reverse=True)
                keyz = keyz[:num_wfs]
                plt.figure(figsize=[10, .9 * num_wfs])
                for keynum, key in enumerate(keyz):
                    wf = row.SVD[key] / (2 * max(row.SVD[key])) - 1.5 * keynum
                    c = 'b' if keynum < len(row.UsedSVDKeys) else '.5'
                    plt.plot(wf, c=c)
                plt.ylim(-1.5 * keynum - 1, 1)
                plt.yticks([])
                plt.xticks([])
                plt.title('%s station %s' % (row.Name, row.Station))

    def plotOffsetTimes(self):
        """
        Function to loop through each station/subspace pair and make 
        histograms of offset times
        """
        count = 1
        for station in self.ssStations:
            for ind, row in self.subspaces[station].iterrows():
                if len(row.SampleTrims.keys()) < 1:
                    msg = 'subspaces must be trimmed before plotting offsets'
                    detex.log(__name__, msg, level='error')
                plt.figure(count)
                keys = row.Events
                offsets = [row.Stats[x]['offset'] for x in keys]
                plt.hist(offsets)
                plt.title('%s %s' % (row.Station, row.Name))
                plt.figure(count + 1)
                numEvs = len(row.Events)
                ranmin = np.zeros(numEvs)
                ranmax = np.zeros(numEvs)
                orsamps = np.zeros(numEvs)
                for evenum, eve in enumerate(row.Events):
                    tem = self.clusters.temkey[
                        self.clusters.temkey.NAME == eve].iloc[0]
                    condat = row.AlignedTD[
                                 eve] / max(2 * abs(row.AlignedTD[eve])) + evenum + 1
                    Nc, Sr = row.Stats[eve]['Nc'], row.Stats[
                        eve]['sampling_rate']
                    starTime = row.Stats[eve]['starttime']
                    ortime = obspy.core.UTCDateTime(tem.TIME).timestamp
                    orsamps[evenum] = row.SampleTrims[
                                          'Starttime'] - (starTime - ortime) * Nc * Sr
                    plt.plot(condat, 'k')
                    plt.axvline(row.SampleTrims['Starttime'], c='g')
                    plt.plot(orsamps[evenum], evenum + 1, 'r*')
                    ran = row.SampleTrims['Endtime'] - orsamps[evenum]
                    ranmin[evenum] = orsamps[evenum] - ran * .1
                    ranmax[evenum] = row.SampleTrims['Endtime'] + ran * .1
                plt.xlim(int(min(ranmin)), int(max(ranmax)))
                plt.axvline(min(orsamps), c='r')
                plt.axvline(max(orsamps), c='r')
                count += 2

    ############################# Pick Times functions 
    def pickTimes(self, duration=30, traceLimit=15, repick=False,
                  subspace=True, singles=True):
        """
        Calls a modified version of obspyck (https://github.com/megies/obspyck)
        , a GUI for picking phases, so user can manually select start times 
        (trim) of unclustered and clustered events.
        Triming down each waveform group to only include event phases, 
        and not pre and post event noise, will significantly decrease the 
        runtime for the subspace detection (called with detex method). 
        Trimming is required for singletons as any singletons without trim
        times will not be used as detectors).
        Parameters
        --------------
        duration : real number
            the time after the first pick (in seconds) to trim waveforms.
            The fact that the streams are multiplexed is taken into account.
            If None is passed then the last pick will be used as the end time
            for truncating waveforms.
        traceLimit : int
            Limits the number of traces that will show up to be manually 
            picked to traceLimit events. Avoids bogging down and/or killing 
            the GUI with too many events.
        repick : boolean
            If true repick times that already have sample trim times, else
            only pick those that do not. 
        subspace : boolean
            If true pick subspaces
        singles : boolean
            If true pick singletons
        """
        qApp = PyQt4.QtGui.QApplication(sys.argv)
        if subspace:
            self._pickTimes(self.subspaces, duration, traceLimit,
                            qApp, repick=repick)
        if singles:
            self._pickTimes(self.singles, duration, traceLimit, qApp,
                            issubspace=False, repick=repick)

    def _pickTimes(self, trdfDict, duration, traceLimit, qApp,
                   issubspace=True, repick=False):
        """
        Function to initate GUI for picking, called by pickTimes
        """
        for sta in trdfDict.keys():
            for ind, row in trdfDict[sta].iterrows():
                if not row.SampleTrims or repick:  # if not picked or repick
                    # Make a modified obspy stream to pass to streamPick
                    st = self._makeOpStream(ind, row, traceLimit)
                    Pks = None  # This is needed or it crashes OS X
                    Pks = detex.streamPick.streamPick(st, ap=qApp)
                    d1 = {}
                    for b in Pks._picks:
                        if b:  # if any picks made
                            d1[b.phase_hint] = b.time.timestamp
                    if len(d1.keys()) > 0:  # if any picks made
                        # get sample rate and number of chans
                        sr = row.Stats[row.Events[0]]['sampling_rate']
                        Nc = row.Stats[row.Events[0]]['Nc']
                        # get sample divisible by NC to keep traces aligned
                        fp = int(min(d1.values()))  # first picked phase
                        d1['Starttime'] = fp - fp % Nc
                        # if duration paramenter is defined (it is usually
                        # better to leave it defined)
                        stime = d1['Starttime']
                        if duration:

                            etime = stime + int(duration * sr * Nc)
                            d1['Endtime'] = etime
                            d1['DurationSeconds'] = duration
                        else:
                            etime = int(max(d1.values()))
                            d1['Endtime'] = etime
                            dursecs = (etime - stime) / (sr * Nc)
                            d1['DurationSeconds'] = dursecs
                        trdfDict[sta].SampleTrims[ind] = d1
                        for event in row.Events:  # update starttimes
                            sspa = trdfDict[sta]
                            stimeOld = sspa.Stats[ind][event]['starttime']
                            # get updated start time
                            stN = stimeOld + d1['Starttime'] / (Nc * sr)
                            ot = trdfDict[sta].Stats[ind][event]['origintime']
                            offset = stN - ot
                            trdfDict[sta].Stats[ind][event]['starttime'] = stN
                            trdfDict[sta].Stats[ind][event]['offset'] = offset
                    if not Pks.KeepGoing:
                        msg = 'aborting picking, progress saved'
                        detex.log(__name__, msg, pri=1)
                        return None
            self._updateOffsets()

    def _makeOpStream(self, ind, row, traceLimit):
        """
        Make an obspy stream of the multiplexed data stored in main detex 
        DataFrame
        """
        st = obspy.core.Stream()
        count = 0
        if 'AlignedTD' in row:  # if this is a subspace
            for key in row.Events:
                if count < traceLimit:
                    tr = obspy.core.Trace(data=row.AlignedTD[key])
                    tr.stats.channel = key
                    tr.stats.network = row.Name
                    tr.stats.station = row.Station
                    st += tr
                    count += 1
            return st
        else:  # if this is a single event
            for key in row.Events:
                tr = obspy.core.Trace(data=row.MPtd[key])
                tr.stats.channel = key
                tr.stats.station = row.Station
                st += tr
            return st

    def _updateOffsets(self):
        """
        Calculate offset (predicted origin times), throw out extreme 
        outliers using median and median scaling
        """
        for sta in self.subspaces.keys():
            for num, row in self.subspaces[sta].iterrows():
                keys = row.Stats.keys()
                offsets = [row.Stats[x]['offset'] for x in keys]
                self.subspaces[sta].Offsets[
                    num] = self._getOffsets(np.array(offsets))
        for sta in self.singles.keys():
            for num, row in self.singles[sta].iterrows():
                keys = row.Stats.keys()
                offsets = [row.Stats[x]['offset'] for x in keys]
                self.singles[sta].Offsets[
                    num] = self._getOffsets(np.array(offsets))

    def attachPickTimes(self, pksFile='PhasePicks.csv', overwrite=False,
                        function=np.median, defaultDuration=30, **kwargs):
        """
        Rather than picking times manually attach a file (either csv or pkl 
        of pandas dataframe) with pick times. Pick time file must have the 
        following fields:  TimeStamp, Station, Event, Phase. 
        This file can be created by detex.util.pickPhases. If trims are 
        already defined attachPickTimes will not override.
        ----------
        pksFile : str
            Path to the input file (either csv or pickle)
        overwrite : bool
            If True overwrite current pick times event if already defined
        function : callable (reducing function)
            Describes how to handle selecting a common pick time for 
            subspace groups (each event in a subspace cannot be treated 
            independently as the entire group is aligned to maximize 
            similarity). Does not apply for singles.
            mean - Trims the group to the sample corresponding to the 
                average of the first arriving phase
            median - Trims the group to the sample corresponding to the 
                median of the first arriving phase
            max - trim to max value of start times for group
            min - trim to min value of end times for group
        defaultDuration : int or None
            if Int, the default duration (in seconds) to trim the signal to 
            starting from the first arrival in pksFile for each event or 
            subspace group. If None, then durations are defined by first 
            arriving phase (start) and last arriving phase (stop) for each
            event
        """
        try:  # read pksFile
            pks = pd.read_csv(pksFile)
        except Exception:
            try:
                pks = pd.read_pickle(pksFile)
            except Exception:
                msg = ('%s does not exist, or it is not a pkl or csv file'
                       % pksFile)
                detex.log(__name__, msg, level='error')
        finally: # trim down picks to only include events in temkey
            temkey = self.clusters.temkey
            pks = pks[pks.Event.isin(temkey.NAME)]
        # get travel times
        self._attach_travel_times(temkey, pks)
        self.pick_times = pks
        # loop through each station in cluster, get singles and subspaces
        for cl in self.clusters:
            sta = cl.station  # current station
            # Attach singles
            if sta in self.singles.keys():
                for ind, row in self.singles[sta].iterrows():
                    if len(row.SampleTrims.keys()) > 0 and not overwrite:
                        continue  # skip if sampletrims already defined
                    # get phases that apply to current event and station
                    con1 = pks.Event.isin(row.Events)
                    con2 = pks.Station == sta
                    pk = pks[(con1) & (con2)]
                    # add origin times
                    eves, starttimes, Nc, Sr = self._getStats(row)
                    if len(pk) > 0:
                        trims = self._getSampTrim(eves, starttimes, Nc, Sr, pk,
                                                  defaultDuration, function, sta,
                                                  ind, self.singles[sta], row)
                        if isinstance(trims, dict):
                            self.singles[sta].SampleTrims[ind] = trims
                self._updateOffsets()
            # Attach Subspaces
            if sta in self.subspaces.keys():
                for ind, row in self.subspaces[sta].iterrows():
                    if len(row.SampleTrims.keys()) > 0 and not overwrite:
                        continue  # skip if sampletrims already defined
                    # phases that apply to current event and station
                    con1 = pks.Event.isin(row.Events)
                    con2 = pks.Station == sta
                    pk = pks[(con1) & (con2)]
                    # add origin times
                    eves, starttimes, Nc, Sr = self._getStats(row)
                    if len(pk) > 0:

                        trims = self._getSampTrim(eves, starttimes, Nc, Sr, pk,
                                                  defaultDuration, function, sta,
                                                  ind, self.subspaces[sta], row)
                        if isinstance(trims, dict):
                            self.subspaces[sta].SampleTrims[ind] = trims
                self._updateOffsets()

    def _attach_travel_times(self, temkey, pk):
        """ get the travel times by subtracting picks from origin """
        # add origin times, calculate travel times
        pk['OT'] = [UTC(temkey[temkey.NAME == x].TIME.iloc[0]).timestamp
                    for x in pk.Event.values]
        pk['TT'] = pk.TimeStamp - pk.OT
        # pk.sort_values(['Event', 'Station', 'TimeStamp'], inplace=True)
        # pk.drop_duplicates(['Station', 'Event'], inplace=True)
        # tt = {row.Event: row.TT for ind, row in pk.iterrows()}
        # return tt

    def _getSampTrim(self, eves, starttimes, Nc, Sr, pk, defaultDuration,
                     fun, sta, num, DF, row):
        """
        Determine sample trims for each single or subspace
        """
        # stdict={}#intialize sample trim dict
        startsamps = []
        stopsamps = []
        secduration = []

        for ev in eves:  # loop through each event
            p = pk[pk.Event == ev]
            if len(p) < 1:  # if event is not recorded skip
                continue
            start = p.TimeStamp.min()
            startsampsEve = (start - starttimes[ev]) * (Nc * Sr)
            # see if any of the samples would be trimmed too much
            try:  # assume is single
                len_test = len(row.MPtd[ev]) < startsampsEve
            except AttributeError:  # this is really a subspace
                len_test = len(row.AlignedTD[ev]) < startsampsEve
            if len_test:
                utc_start = obspy.UTCDateTime(start)
                msg = (('Start samples for %s on %s exceeds avaliable data,'
                        'check waveform quality and ensure phase pick is for '
                        'the correct event. The origin time is %s and the '
                        'pick time is %s, Skipping attaching pick. '
                        ) % (ev, sta, ev, str(utc_start)))
                detex.log(__name__, msg, level='warn')
                return
            # make sure starting time is not less than 0 else set to zero
            if startsampsEve < 0:
                startsampsEve = 0
                start = starttimes[ev]
                msg = 'Start time in phase file < 0 for event %s' % ev
                detex.log(__name__, msg, level='warning', pri=False)
            if defaultDuration:
                stop = start + defaultDuration
                secduration.append(defaultDuration)
            else:
                stop = p.TimeStamp.max()
                secduration.append(stop - start)
            assert stop > start  # Make sure stop is greater than start
            assert stop > starttimes[ev]
            endsampsEve = (stop - starttimes[ev]) * (Nc * Sr)
            startsamps.append(startsampsEve)
            stopsamps.append(endsampsEve)
            # update stats attached to each event to reflect new start time
            otime = DF.Stats[num][ev]['origintime']  # origin time
            DF.Stats[num][ev]['Starttime'] = start
            DF.Stats[num][ev]['offset'] = start - otime
        if len(startsamps) > 0:
            sSamps = int(fun(startsamps))
            rSSamps = sSamps - sSamps % Nc
            eSamps = int(fun(stopsamps))
            rESamps = eSamps - eSamps % Nc
            dursec = int(fun(secduration))
            outdict = {'Starttime': rSSamps, 'Endtime': rESamps,
                       'DurationSeconds': dursec}
            return outdict
        else:
            return

    def _getStats(self, row):
        """
        Get the sampling rate, starttime, and number of channels for 
        each event group
        """
        eves = row.Events
        sr = [np.round(row.Stats[x]['sampling_rate']) for x in eves]
        if len(set(sr)) != 1:
            msg = ('Events %s on Station %s have different sampling rates or '
                   'no sampling rates' % (row.Station, row.events))
            detex.log(__name__, msg, level='error')
        Nc = [row.Stats[x]['Nc'] for x in eves]
        if len(set(Nc)) != 1:
            msg = (('Events %s on Station %s do not have the same channels or'
                    ' have no channels') % (row.Station, row.events))
            detex.log(__name__, msg, level='error')
        starttimes = {x: row.Stats[x]['starttime'] for x in eves}
        return eves, starttimes, list(set(Nc))[0], list(set(sr))[0]

    def _getOffsets(self, offsets, m=25.):
        """
        Get offsets, reject outliers bassed on median values (accounts 
        for possible mismatch in events and origin times)
        """
        if len(offsets) == 1:
            return offsets[0], offsets[0], offsets[0]
        d = np.abs(offsets - np.median(offsets))
        mdev = np.median(d)
        s = d / mdev if mdev else 0.
        if isinstance(s, float):
            offs = offsets
        else:
            offs = offsets[s < m]
        return [np.min(offs), np.median(offs), np.max(offs)]

    def getFAS(
            self,
            conDatNum,
            LTATime=5,
            STATime=0.5,
            staltalimit=8.0,
            useSubSpaces=True,
            useSingles=False,
            numBins=401,
            recalc=False,
            **kwargs):
        """
        Function to initialize a FAS (false alarm statistic) instance, used
        primarily for sampling and characterizing the null space of the 
        subspaces and singletons. Random samples of the continuous data are
        loaded, examined for high amplitude signals with a basic STA/LTA 
        method, and any traces with STA/LTA ratios higher than the 
        staltalimit parameter are rejected. The continuous DataFetcher
        already attached to the SubSpace instance will be used to get
        the continuous data. 
        Parameters
        -------------        
        ConDatNum : int
            The number of continuous data files (by default in hour chunks)
            to use.
        LTATime : float 
            The long term average time window in seconds used for 
            checking continuous data
        STATime : float
            The short term average time window in seconds for checking
            continuous data
        staltalimit : int or float
            The value at which continuous data gets rejected as too noisey
            (IE transient signals are present)
        useSubSpaces : bool
            If True calculate FAS for subspaces
        useSingles : bool
            If True calculate FAS for singles
        numBins : int
            Number of bins for binning distributions (so distribution can be 
            loaded and plotted later)
        
        Note
        ---------
        The results are stored in a DataFrame for each subspace/singleton
        under the "FAS" column of the main DataFrame
        """
        if useSubSpaces:
            self._updateOffsets()  # make sure offset times are up to date
            for sta in self.subspaces.keys():
                # check if FAS already calculated, only recalc if recalc
                fas1 = self.subspaces[sta]['FAS'][0]
                if isinstance(fas1, dict) and not recalc:
                    msg = ('FAS for station %s already calculated, to '
                           'recalculate pass True to the parameter recalc' %
                           sta)
                    detex.log(__name__, msg, pri=True)
                else:
                    self.subspaces[sta]['FAS'] = detex.fas._initFAS(
                        self.subspaces[sta],
                        conDatNum,
                        self.clusters,
                        self.cfetcher,
                        LTATime=LTATime,
                        STATime=STATime,
                        staltalimit=staltalimit,
                        numBins=numBins,
                        dtype=self.dtype)
        if useSingles:
            for sta in self.singles.keys():
                for a in range(len(self.singles[sta])):
                    fas1 = self.singles[sta]['FAS'][a]
                    if isinstance(fas1, dict) and not recalc:
                        msg = (('FAS for singleton %d already calculated on '
                                'station %s, to recalculate pass True to the '
                                'parameter recalc') % (a, sta))
                        detex.log(__name__, msg, pri=True)
                    # skip any events that have not been trimmed
                    elif len(self.singles[sta]['SampleTrims'][a].keys()) < 1:
                        continue
                    else:
                        self.singles[sta]['FAS'][a] = detex.fas._initFAS(
                            self.singles[sta][a:a + 1],
                            conDatNum,
                            self.clusters,
                            self.cfetcher,
                            LTATime=LTATime,
                            STATime=STATime,
                            staltalimit=staltalimit,
                            numBins=numBins,
                            dtype=self.dtype,
                            issubspace=False)

    def detex(self,
              utcStart=None,
              utcEnd=None,
              subspaceDB='SubSpace.db',
              trigCon=0,
              triggerLTATime=5,
              triggerSTATime=0,
              multiprocess=False,
              delOldCorrs=True,
              calcHist=True,
              useSubSpaces=True,
              useSingles=False,
              estimateMags=True,
              classifyEvents=None,
              eventCorFile='EventCors',
              utcSaves=None,
              fillZeros=False,
              **kwargs):
        """
        function to run subspace detection over continuous data and store 
        results in SQL database subspaceDB

        Parameters
        ------------
        utcStart : str or num
            An obspy.core.UTCDateTime readable object defining the start time
            of the correlations if not all avaliable data are to be used
        utcEnd : str num
            An obspy.core.UTCDateTime readable object defining the end time 
            of the correlations
        subspaceDB : str
            Path to the SQLite database to store detections in. If it already 
            exists delOldCorrs parameters governs if it will be deleted before
            running new detections, or appended to. 
        trigCon is the condition for which detections should trigger. 
            Once the condition is set the variable minCoef is used:
                0 is based on the detection statistic threshold
                1 is based on the STA/LTA of the detection statistic threshold
                (Only 0 is currently supported)
        triggerLTATime : number
            The long term average for the STA/LTA calculations in seconds.
        triggerSTATime : number
            The short term average for the STA/LTA calculations in seconds. 
            If ==0 then one sample is used.
        multiprocess : bool
            Determine if each station should be forked into its own process 
            for potential speed ups. Currently not implemented. 
        delOldCorrs : bool
            Determines if subspaceDB should be deleted before performing 
            detections. If False old database is appended to. 
        calcHist : boolean
            If True calculates the histagram for every point of the detection 
            statistic vectors (all hours, stations and subspaces) by keeping a
            a cumulative bin count. Only slows the detections down slightly 
            and can be useful for threshold sanity checks. The histograms are
            then returned to the main DataFrame in the SubSpace instance
            as the column histSubSpaces, and saved in the subspaceDB under the
            ss_hist and sg_hists tables for subspacs and singletons. 
        useSubspace : bool
            If True the subspaces will be used as detectors to scan 
            continuous data
        useSingles : bool
            If True the singles (events that did not cluster) will be used as 
            detectors to scan continuous data
        estimateMags : bool
            If True, magnitudes will be estimated for each detection by using
            two methods. The first is using standard deviation ratios, and the 
            second uses projected energy ratios (see chambers et al. 2015 for
            details).
        classifyEvents : None, str, or DataFrame
            If None subspace detectors will be run over continuous data.
            Else, detex will be run over event waveforms in order to classify
            events into groups bassed on which subspace they are most similar
            to. In the latter case the classifyEvents argument must be a 
            str (path to template key like csv) or DataFrame (loaded template
            key file). The same event DataFetcher attached to the cluster
            object will be used to get the data. This feature is Experimental. 
        eventCorFile : str
            A path to a new pickled DataFrame created when the eventDir option
            is used. Records the highest detection statistic in the file 
            for each event, station, and subspace. Useful when trying to
            characterize events.
        utcSaves : None or list of obspy DateTime readable objects
            Either none (not used) or an iterrable of objects readable by 
            obspy.UTCDateTime. When the detections are run if the continous 
            data cover a time indicated in UTCSaves then the continuous data 
            and detection statistic vectors,are saved to a pickled dataframe
            of the name "UTCsaves.pkl". This can be useful for debugging, or 
            extracting the DS vector for a time of interest. 
        fillZeros : bool
            If true fill the gaps in continuous data with 0s. If True 
            STA/LTA of detection statistic cannot be calculated in order to 
            avoid dividing by 0.
        Notes
        ----------
        The same filter and decimation parameters that were used in the
        ClusterStream instance will be applied.
        """
        # make sure no parameters that dont work yet are selected
        if multiprocess or trigCon != 0:
            msg = 'multiprocessing and trigcon other than 0 not supported'
            detex.log(__name__, msg, level='error')

        if os.path.exists(subspaceDB):
            if delOldCorrs:
                os.remove(subspaceDB)
                msg = 'Deleting old subspace database %s' % subspaceDB
                detex.log(__name__, msg, pri=True)
            else:
                msg = 'Not deleting old subspace database %s' % subspaceDB
                detex.log(__name__, msg, pri=True)

        if useSubSpaces:  # run subspaces
            TRDF = self.subspaces
            # determine if subspaces are defined (ie SVD has been called)
            stas = self.subspaces.keys()
            sv = [all(TRDF[sta].SVDdefined) for sta in stas]
            if not all(sv):
                msg = 'call SVD before running subspace detectors'
                detex.log(__name__, msg, level='error')

            Det = _SSDetex(TRDF, utcStart, utcEnd, self.cfetcher, self.clusters,
                           subspaceDB, trigCon, triggerLTATime, triggerSTATime,
                           multiprocess, calcHist, self.dtype, estimateMags,
                           classifyEvents, eventCorFile, utcSaves, fillZeros,
                           pick_times=self.pick_times)
            self.histSubSpaces = Det.hist

        if useSingles:  # run singletons
            # make sure thresholds are calcualted
            self.setSinglesThresholds()
            TRDF = self.singles
            Det = _SSDetex(TRDF, utcStart, utcEnd, self.cfetcher, self.clusters,
                           subspaceDB, trigCon, triggerLTATime, triggerSTATime,
                           multiprocess, calcHist, self.dtype, estimateMags,
                           classifyEvents, eventCorFile, utcSaves, fillZeros,
                           issubspace=False, pick_times=self.pick_times)
            self.histSingles = Det.hist

        # save addational info to sql database
        if useSubSpaces or useSingles:
            cols = ['FREQMIN', 'FREQMAX', 'CORNERS', 'ZEROPHASE']
            dffil = pd.DataFrame([self.clusters.filt], columns=cols, index=[0])
            detex.util.saveSQLite(dffil, subspaceDB, 'filt_params')

            # get general info on each singleton/subspace and save
            ssinfo, sginfo = self._getInfoDF()
            sshists, sghists = self._getHistograms(useSubSpaces, useSingles)
            if useSubSpaces and ssinfo is not None:
                # save subspace info
                detex.util.saveSQLite(ssinfo, subspaceDB, 'ss_info')
            if useSingles and sginfo is not None:
                # save singles info
                detex.util.saveSQLite(sginfo, subspaceDB, 'sg_info')
            if useSubSpaces and sshists is not None:
                # save subspace histograms
                detex.util.saveSQLite(sshists, subspaceDB, 'ss_hist')
            if useSingles and sghists is not None:
                # save singles histograms
                detex.util.saveSQLite(sghists, subspaceDB, 'sg_hist')

    def _getInfoDF(self):
        """
        get dataframes that have info about each subspace and single
        """
        sslist = []  # list in which to put DFs for each subspace/station pair
        sglist = []  # list in which to put DFs for each single/station pair
        for sta in self.Stations:
            if sta not in self.ssStations:
                msg = 'No subspaces on station %s' % sta
                detex.log(__name__, msg, pri=True)
                continue
            for num, ss in self.subspaces[sta].iterrows():  # write ss info
                name = ss.Name
                station = ss.Station
                events = ','.join(ss.Events)
                numbasis = ss.NumBasis
                thresh = ss.Threshold
                if isinstance(ss.FAS, dict) and len(ss.FAS.keys()) > 1:
                    b1, b2 = ss.FAS['betadist'][0], ss.FAS['betadist'][1]
                else:
                    b1, b2 = np.nan, np.nan
                cols = ['Name', 'Sta', 'Events', 'Threshold', 'NumBasisUsed',
                        'beta1', 'beta2']
                dat = [[name, station, events, thresh, numbasis, b1, b2]]
                sslist.append(pd.DataFrame(dat, columns=cols))
        for sta in self.Stations:
            if sta not in self.singStations:
                msg = 'No singletons on station %s' % sta
                detex.log(__name__, msg, pri=True)
                continue
            for num, ss in self.singles[sta].iterrows():  # write singles info
                name = ss.Name
                station = ss.Station
                events = ','.join(ss.Events)
                thresh = ss.Threshold
                if isinstance(ss.FAS, list) and len(ss.FAS[0].keys()) > 1:
                    b1, b2 = ss.FAS[0]['betadist'][0], ss.FAS[0]['betadist'][1]
                else:
                    b1, b2 = np.nan, np.nan
                cols = ['Name', 'Sta', 'Events', 'Threshold', 'beta1', 'beta2']
                dat = [[name, station, events, thresh, b1, b2]]
                sglist.append(pd.DataFrame(dat, columns=cols))
        if len(sslist) > 0:
            ssinfo = pd.concat(sslist, ignore_index=True)
        else:
            ssinfo = None
        if len(sglist) > 0:
            sginfo = pd.concat(sglist, ignore_index=True)
        else:
            sginfo = None
        return ssinfo, sginfo

    def _getHistograms(self, useSubSpaces, useSingles):
        """
        Pull out the histogram info for saving to database
        """
        cols = ['Name', 'Sta', 'Value']
        if useSubSpaces:
            bins = json.dumps(self.histSubSpaces['Bins'].tolist())
            dat = [['Bins', 'Bins', bins]]
            sshists = [pd.DataFrame(dat, columns=cols)]
            for sta in self.Stations:
                if sta in self.histSubSpaces.keys():
                    for skey in self.histSubSpaces[sta]:
                        try:
                            vl = json.dumps(self.histSubSpaces[sta][skey].tolist())
                        except AttributeError:
                            continue
                        dat = [[skey, sta, vl]]
                        sshists.append(pd.DataFrame(dat, columns=cols))
            sshist = pd.concat(sshists, ignore_index=True)
        else:
            sshist = None

        if useSingles:
            bins = json.dumps(self.histSingles['Bins'].tolist())
            dat = [['Bins', 'Bins', bins]]
            sghists = [pd.DataFrame(dat, columns=cols)]
            for sta in self.Stations:
                if sta in self.histSingles.keys():
                    for skey in self.histSingles[sta]:
                        try:
                            vl = json.dumps(self.histSingles[sta][skey].tolist())
                        except AttributeError:
                            pass
                        dat = [[skey, sta, vl]]
                        sghists.append(pd.DataFrame(dat, columns=cols))
            sghist = pd.concat(sghists, ignore_index=True)
        else:
            sghist = None

        return sshist, sghist

    ########################### Python Class Attributes

    def __getitem__(self, key):  # make object indexable
        if isinstance(key, int):
            return self.subspaces[self.ssStations[key]]
        elif isinstance(key, string_types):
            if len(key.split('.')) == 2:
                return self.subspaces[self._stakey2[key]]
            elif len(key.split('.')) == 1:
                return self.subspaces[self._stakey1[key]]
            else:
                msg = '%s is not a station in this cluster object' % key
                detex.log(__name__, msg, level='error')
        else:
            msg = '%s must either be a int or str of station name' % key
            detex.log(__name__, msg, level='error')

    def __len__(self):
        return len(self.subspaces)

    ############ MISC 
    def write(self, filename='subspace.pkl'):
        """
        pickle the subspace class
        Parameters
        -------------
        filename : str
            Path of the file to be created
        """
        cPickle.dump(self, open(filename, 'wb'))

    def printOffsets(self):
        """
        Function to print out the offset min max and ranges for each 
        station/subpace pair
        """
        for station in self.ssStations:
            for num, row in self.subspaces[station].iterrows():
                print('%s, %s, min=%3f, max=%3f, range=%3f' %
                      (row.Station, row.Name, row.Offsets[0], row.Offsets[2],
                       row.Offsets[2] - row.Offsets[0]))
