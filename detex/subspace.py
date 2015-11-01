# -*- coding: utf-8 -*-
"""
Created on Tue Jul 08 21:24:18 2014

@author: Derrick
Module to perform subspace detection and waveform similarity clustering
"""
import pandas as pd
import numpy as np
import obspy
import os
import sys
import numbers
import glob
import matplotlib.pyplot as plt
import json
import matplotlib as mpl
import detex
import scipy
import cPickle
import itertools
import collections
import copy
import colorsys
import multiprocessing
from struct import pack
import warnings

from scipy.cluster.hierarchy import dendrogram, fcluster
from detex.construct import fast_normcorr, multiplex

pd.options.mode.chained_assignment = None #mute setting copy warning 

#warnings.filterwarnings('error') #uncomment this to make all warnings errors
 
# lines for backward compat.
createSubSpace = detex.construct.createSubSpace
createCluster = detex.construct.createCluster


def loadClusters(filename='clust.pkl'): 
    """
    Function that uses pandas.read_pickle to load a pickled cluster
    (instance of detex.subspace.ClusterStream)
    Parameters
    ----------
    filename : str
        Path to the saved cluster isntance
    Returns
    ----------
    An instance of detex.subspace.ClusterStream 
    """
    cl = pd.read_pickle(filename)
    return cl
    
def loadSubSpace(filename='subspace.pkl'): 
    """
    Function that uses pandas.read_pickle to load a pickled subspace
    (instance of detex.subspace.SubSpaceStream)
    Parameters
    ----------
    filename : str
        Path to the saved subspace instance
    Returns
    ----------
    An instance of detex.subspace.SubSpaceStream 
    """
    ss = pd.read_pickle(filename)
    return ss
    



class ClusterStream(object):
    """
    A container for multiple cluster objects
    """
    def __init__(self, trdf,temkey, stakey, fetcher, eventList, CCreq, filt, 
                 decimate, trim, fileName, eventsOnAllStations, enforceOrigin):
       
        self.__dict__.update(locals()) #Instantiate all input variables
        self.CCreq=None # get rid of this attribute as it can vary between stations
        self.clusters=[0]*len(trdf)
        self.stalist=trdf.Station.values.tolist() #get station lists for indexing
        self.stalist2=[x.split('.')[1] for x in self.stalist]
        self.filename=fileName
        self.eventCodes=self._makeCodes()
        for num,row in trdf.iterrows():
            if not eventsOnAllStations:
                evlist=row.Events
            else:
                evlist=eventList
            self.clusters[num] = Cluster(self, row.Station, temkey, evlist,row.Link,CCreq,filt,decimate,trim,row.CCs)

    def writeSimpleHypoDDInput(self,fileName='dt.cc',coef=1,minCC=.35):
        """
        create a hypoDD cross correlation file (IE dt.cc), assuming the lag times are pure S times (true if S amplitude is larger than P)
       
        Parameters
        ----------
        fileName : str
            THe path to the new file to be created
        coef : float or int
            The exponential coeficient to apply to the correlation coeficient when creating file
       """
        if not self.enforceOrigin:
            detex.log(__name__,'Sample Lags are not meaningful unless origin times are enforced on each waveform. re-run detex.subspace.createCluster with enforceOrigins=True',level='warning')
            raise Exception('Sample Lags are not meaningful unless origin times are enforced on each waveform. re-run detex.subspace.createCluster with enforceOrigins=True')
        fil=open(fileName,'wb')
        reqZeros=int(np.ceil(np.log10(len(self.temkey)))) #required number of zeros for numbering all events
        for num1,everow1 in self.temkey.iterrows():
            for num2,everow2 in self.temkey.iterrows():
                if num1>=num2: #if autocorrelation or redundant pair just skip it
                    continue    
                ev1,ev2=everow1.NAME,everow2.NAME
                header=self._makeHeader(num1,num2,reqZeros)
                count=0
                for sta in self.stalist: #itterate through each station
                    Clu=self[sta]
                    try:    
                        ind1=np.where(np.array(Clu.key)==ev1)[0][0] #find station specific index for event1
                        ind2=np.where(np.array(Clu.key)==ev2)[0][0]
                    except IndexError: #if either event is nopt in index
                        detex.log(__name__,'%s or %s not found on station %s'%(ev1,ev2,sta))
                        #deb([Clu,ev1,ev2])
                        continue
                    
                    trdf=self.TRDF[self.TRDF.Station==sta].iloc[0]
                    #deb([trdf,ind1,ind2])
                    sr1,sr2=trdf.Stats[ev1]['sampling_rate'],trdf.Stats[ev2]['sampling_rate']
                    if sr1!=sr2:
                        detex.log(__name__,'Sampling rates not equal for %s and %s'%(ev1,ev2))
                        raise Exception('Sampling rates not equal for %s and %s'%(ev1,ev2))
                    else:
                        sr=sr1
                    Nc1,Nc2=trdf.Stats[ev1]['Nc'],trdf.Stats[ev2]['Nc']
                    if Nc1!=Nc2:
                        #deb([trdf,ev1,ev2,sta])
                        detex.log(__detex__,'Number of channels not equal for %s and %s on %s',level='warning')
                        continue
                    else:
                        Nc=Nc1
                    cc=trdf.CCs[ind2][ind1] #get cc value
                    if np.isnan(cc): # grab other part of symetric matrix if nan
                        try:
                            cc=trdf.CCs[ind1][ind2]
                        except KeyError:
                            continue
                        if np.isnan(cc):
                            continue
                    if cc<minCC or np.isnan(cc):
                        continue
                    lagsamps=trdf.Lags[ind2][ind1]
                    if np.isnan(lagsamps):
                        lagsamps=-trdf.Lags[ind2][ind1]
                    lags=lagsamps/(sr*Nc)
                    obsline=self._makeObsLine(sta,lags,cc**coef)
                    if np.isnan(cc):
                        deb(1)
                    if isinstance(obsline,str):
                        count+=1
                        if count==1:
                            fil.write(header+'\n')
                        fil.write(obsline+'\n')
        fil.close()
    
    def _makeObsLine(self,sta,dt,cc,pha='S',mincc=0,weightCoef=1):
        if cc<mincc: #if cc is lower than min, return nothing
            return
        line='%s %0.4f %0.4f %s' %(sta,dt,cc**weightCoef,pha)
        return line
    
    def _makeHeader(self,num1,num2,reqZeros):
        fomatstr='{:0'+"{:d}".format(reqZeros)+'d}'
        head='# '+fomatstr.format(num1)+' '+fomatstr.format(num2)+' '+'0.0' #assume cross corr and cat origins are identical
        return head
    
    def writeHypoDDStationInput(self,fileName='station.dat',useElevations=True,inFt=False):
        """
        Write the station input file for hypoDD (station.dat)
        
        Parameters
        ---------
        fileName : str
            Path to the output file
        useElevations : boolean
            If true also print elevations
        inFt : boolean
            If true elevations in station key are in ft, convert to meters
        """
        fil=open(fileName,'wb')
        conFact=0.3048 if inFt else 1 #conversion factor from ft to meters if needed
        for num,row in self.stakey.iterrows():
            line='%s %.6f %.6f'%(row.NETWORK+'.'+row.STATION,row.LAT,row.LON)
            if useElevations:
                line=line+' %.2f'%row.ELEVATION*conFact 
            fil.write(line+'\n')
        fil.close()
        
    def writeHypoDDEventInput(self,fileName='event.dat'):
        fil=open(fileName,'wb')
        reqZeros=int(np.ceil(np.log10(len(self.temkey))))
        fomatstr='{:0'+"{:d}".format(reqZeros)+'d}'
        for num,row in self.temkey.iterrows():
            utc=obspy.UTCDateTime(row.TIME)
            DATE='%04d%02d%02d'%(int(utc.year),int(utc.month),int(utc.day))
            TIME='%02d%02d%04d'%(int(utc.hour),int(utc.minute),int(utc.second*100))
            mag=row.MAG if row.MAG>-20 else 0.0
            ID=fomatstr.format(num)
            linea=DATE+', '+TIME+', '+'{:04f}, '.format(row.LAT)+'{:04f}, '.format(row.LON)+'{:02f}, '.format(row.DEPTH)
            lineb='{:02f}, '.format(mag)+'0.0, 0.0, 0.0, ' + ID
            fil.write(linea+lineb+'\n')
        fil.close()
            
        
        
    def _makeCodes(self):
        evcodes={}
        for num,row in self.temkey.iterrows():
            evcodes[num]=row.NAME
        return evcodes
    
    def updateReqCC(self,reqCC):
        """
        Updates the required correlation coefficient for clusters to form on all stations if reqCC is a float
        or on each station if reqCC is a dict whose keys reference individual station cluster objects
        
        Parameters:
        reqCC : float (between 0 and 1), or dict of reference keys and floats, or list of floats
        """
        if isinstance(reqCC,float):
            if reqCC<0 or reqCC>1:
                detex.log(__name__,'reqCC must be between 0 and 1',level='error')
                raise Exception ('reqCC must be between 0 and 1')
            for cl in self.clusters:
                cl.updateReqCC(reqCC)
        elif isinstance(reqCC,dict):
            for key in reqCC.keys():
                self[key].updateReqCC(reqCC[key])
        elif isinstance(reqCC,list):
            for num,cc in enumerate(reqCC):
                self[num].updateReqCC(cc)
        self.write()

    def printAtr(self): #print out basic attributes used to make cluster
        for cl in self.clusters:
            cl.printAtr()
            
    def dendro(self):
        """
        Create dendrograms for each station
        """
        for cl in self.clusters:
            cl.dendro()
    
        
    def simMatrix(self,groupClusts=False,savename=False,returnMat=False,**kwargs):
        """
        Function to create basic similarity matrix of the values in the cluster object
        
        Parameters
        -------
        groupClusts : boolean
            If True order by clusters on the simmatrix with the singletons coming last
        savename : str or False
            If not False, a path used by plt.savefig to save the current figure. The extension is necesary for specifying format. See plt.savefig for details
        returnMat : boolean 
            If true return the similarity matrix
        """
        """
        Create similarity matricies for each station
        """
        out=[]
        for cl in self.clusters:
            dout=cl.simMatrix(groupClusts,savename,returnMat,**kwargs)
            out.append(dout)
            
    def plotEvents(self,projection='merc',plotNonClusts=True):
        """
        Plot the event locations for each station
        
        Parameters
        ---------
        projection : str
            The pojection type to pass to basemap
        plotNonClusts : boolean
            If true also plot the events that don't cluster
        """
        for cl in self.clusters:
            cl.plotEvents(projection=projection,plotNonClusts=plotNonClusts)        
            
    def write(self): #uses pickle to write class to disk
        detex.log(__name__,'writing cluster object as %s' % self.filename,pri=True)
        cPickle.dump(self,open(self.filename,'wb'))
    
    def __getitem__(self,key):
        if isinstance(key,int):
            return self.clusters[key]
        elif isinstance(key,str):
            if len(key.split('.'))==2:
                return self.clusters[self.stalist.index(key)]
            elif len(key.split('.'))==1:
                return self.clusters[self.stalist2.index(key)]
            else:
                detex.log(__name__,'%s is not a station in this cluster object' % key,level='error')
                raise Exception('%s is not a station in this cluster object' % key)
        else:
            detex.log(__name__,'%s must either be a int or str of station name' %key,level='error')
            raise Exception ('%s must either be a int or str of station name' %key)
            
    def __len__(self): 
        return len(self.clusters) 
        
    def __repr__(self):
        outstr='SSClusterStream with %d stations '%(len(self.stalist))
        return outstr
        

class Cluster(object):
    def __init__(self,clustStream,station,temkey,eventList,link,CCreq,filt,decimate,trim,DFcc):
        #self.__dict__.update(locals()) #Instantiate all input variables
    
        self.link,self.DFcc,self.station,self.temkey=link,DFcc,station,temkey #instatiate the few needed varaibles
        self.key=eventList
        self.updateReqCC(CCreq)
        self.nonClustColor='0.6' #use a grey of 0.6 for non-clustering events
            
    def updateReqCC(self,newCCreq):
        """
        Function to update the required correlation coeficient for events to cluster without re-running all the correlations
        
        newCCreq is the new required correlation coeficient
        """
        if newCCreq<0 or newCCreq>1:
            detex.log(__name__,'Parameter CCreq must be between 0 and 1')
            raise Exception ('Parameter CCreq must be between 0 and 1')
        self.CCreq=newCCreq        
        self.dflink,serclus=self._makeDFLINK(truncate=False)
        
        dfcl=self.dflink[self.dflink.disSim<=1-self.CCreq] #events that actually cluster
        dfcl.sort(columns='disSim',inplace=True,ascending=False) #sort putting highest links in cluster on top
        dfcl.reset_index(inplace=True,drop=True)
        dftemp=dfcl.copy()
        clustlinks={}
        clustEvents={}
        clnum=0
        while len(dftemp)>0: 
            ser=dftemp.iloc[0]
            ndf=dftemp[[set(x).issubset(ser.II) for x in dftemp.II]]
            clustlinks[clnum]=ndf.clust
            clustEvents[clnum]=list(set([y for x in ndf.II.values for y in x]))
            dftemp=dftemp[~dftemp.index.isin(ndf.index)]
            clnum+=1
        self.clustlinks=clustlinks
        self.clusts=[[self.key[y] for y in clustEvents[x]] for x in clustEvents.keys()]
        self.singles=list(set(self.key).difference(set([y for x in self.clusts for y in x])))
        self.clustcount=np.sum([len(x) for x in self.clusts])
        self.clustColors=self._getColors(len(self.clusts))
        detex.log(__name__,'CCreq for station %s updated to CCreq=%1.3f'%(self.station,newCCreq),pri=True)

    def _getColors(self,numClusts):
        """
        See if there are enough defualt colors for the clusters, if not 
        Generate N unique colors (that probably dont look good together) 
        """        
        clustColorsDefault=['b','g','r','c','m','y','k']
        if numClusts<=len(clustColorsDefault): # if there are enough default python colors use them
            return clustColorsDefault[:numClusts]
        else: #if not generaete N uniwue colors
            colors=[]
            for i in np.arange(0., 360., 360. / numClusts):
                hue = i/360.
                lightness = (50 + np.random.rand() * 10)/100.
                saturation = (90 + np.random.rand() * 10)/100.
                cvect=colorsys.hls_to_rgb(hue, lightness, saturation)
                rgb=[int(x*255) for x in cvect]
                colors.append('#'+pack("BBB",*rgb).encode('hex')) #covnert to hex code
            return colors    
        
    def _makeColorDict(self,clustColors,nonClustColor):
        if len(self.clusts)<1:
            colorsequence=clustColors
        elif float(len(clustColors))/len(self.clusts)<1: #if not enough colors repeat color matrix
            colorsequence=clustColors*int(np.ceil((float(len(self.clusts))/len(clustColors))))
        else:  
            colorsequence=clustColors
        color_list=[nonClustColor]*3*len(self.dflink) #unitialize color list with default color        
        for a in range(len(self.clusts)):
            for b in self.clustlinks[a]:
                color_list[int(b)]=colorsequence[a]
        return color_list

    def _makeDFLINK(self,truncate=True): #make the link dataframe 
        N=len(self.link)
        link=np.append(self.link,np.arange(N+1,N+N+1).reshape(N,1),1) #append cluster numbers to link array
        if truncate:
            linkup=link[link[:,2]<=1-self.CCreq] #truncate after required coeficient
        else:
            linkup=link
        T=fcluster(link[:,0:4],1-self.CCreq,criterion='distance')
        serclus=pd.Series(T)
        
        clusdict=pd.Series([ np.array([x]) for x in np.arange(0,N+1)],index=np.arange(0,N+1)) 
        for a in range(len(linkup)):
            clusdict[int(linkup[a,4])]=np.append(clusdict[int(linkup[a,0])],clusdict[int(linkup[a,1])])
        dflink=pd.DataFrame(linkup,columns=['i1','i2','disSim','num','clust'])
        if len(dflink)>0:
            dflink['II']=list
        else:
            detex.log(__name__,'No events clustered bassed on the requirement that cor coef = %1.3f' % self.CCreq)
        for a in dflink.iterrows(): #enumerate cluster contents
            #dflink['II'][a[0]]=clusdict[a[1].i1].values.tolist()+clusdict[a[1].i2].values.tolist()
            dflink['II'][a[0]]=np.array(clusdict[int(a[1].i1)]).tolist()+np.array(clusdict[int(a[1].i2)]).tolist()
        return dflink,serclus    
    
        
            
    def dendro(self,hideEventLabels=True,show=True,saveName=False,legend=True,**kwargs): #creates a basic dendrogram plot
        """
        Function to plot the dendrogram clusters using scipy
        
        Parameters 
        -----
        hideEventLabels : boolean
            turns x axis labeling on/off. Better set to false if many events are in event pool
        show : boolean
            If true call the plt.show function
        saveName : str
            path to save figure. Extention denotes format. See plt.savefig for details
        legend : boolean
            If true plot a legend on the side of the dendrogram
            
        see scipy.cluster.hierarchy.dendrogram for acceptable kwargs and descriptions
        

        
        """
        # Get color schemes
        

        color_list=self._makeColorDict(self.clustColors,self.nonClustColor)
        #plt.subplot(111)
        for a in range(len(self.clusts)):
            plt.plot([],[],'-',color=self.clustColors[a])
        plt.plot([],[],'-',color=self.nonClustColor)
        #deb(color_list)
       # dendrogram(self.link,count_sort=True,link_color_func=lambda x: color_list[x],**kwargs)
        #deb([self.link,1-self.CCreq,c,color_list,self.clustColors])
        dendrogram(self.link,color_threshold=1-self.CCreq,count_sort=True,link_color_func=lambda x: color_list[x],**kwargs)
        ax=plt.gca()
        if legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend([str(x) for x in range(1,len(self.clusts)+1)]+['N/A'],loc='center left', bbox_to_anchor=(1, .5),title='Clusters') # I cheated and I lost, fix this when possible
        ax.set_ylim([0,1])
        if hideEventLabels:
            ax.set_xticks([])
        plt.xlabel('Events')
        plt.ylabel('Dissimilarity')
        plt.title(self.station)
        if saveName:
            plt.savefig(saveName,**kwargs)
        if show:
            plt.show()
    
    def plotEvents(self,projection='merc',plotNonClusts=True):
        """
        Function to use basemap to plot the physical locations of the events in the same color as their respective clusters
        
        """
        try:
            from  mpl_toolkits.basemap import Basemap
        except ImportError:
            detex.log(__name__,'mpl_toolskits does not have basemap, plotting cannot be perfromed')
            raise ImportError('mpl_toolskits does not have basemap, plotting cannot be perfromed')
        #TODO make dot size scale with magnitudes 
        self.dendro()
        plt.figure()
        #plt.subplot(1,3,1)
        
        latmin,latmax,lonmin,lonmax=self.temkey.LAT.min(),self.temkey.LAT.max(),self.temkey.LON.min(),self.temkey.LON.max()
        latbuff=abs((latmax-latmin)*0.1) #create buffers so there is a slight border with no events around map
        lonbuff=abs((lonmax-lonmin)*0.1)
        totalxdist=obspy.core.util.geodetics.gps2DistAzimuth(latmin,lonmin,latmin,lonmax)[0]/1000 #get the total x distance of plot in km
        emap=Basemap(projection='merc', lat_0 = np.mean([latmin,latmax]), lon_0 = np.mean([lonmin,lonmax]),
                     resolution = 'h', area_thresh = 0.1,
                     llcrnrlon=lonmin-lonbuff, llcrnrlat=latmin-latbuff,
                     urcrnrlon=lonmax+lonbuff, urcrnrlat=latmax+latbuff)
        emap.drawmapscale(lonmin, latmin, lonmin, latmin, totalxdist/4.5)
        
        temDFs=[self.temkey[self.temkey.NAME.isin(x)] for x in self.clusts]
        nocldf=self.temkey[self.temkey.NAME.isin([x for x in self.singles])]
        xmax,xmin,ymax,ymin=emap.xmax,emap.xmin,emap.ymax,emap.ymin
        horrange=max((xmax-xmin),(ymax-ymin)) #horizontal range
        zmin,zmax=self.temkey.DEPTH.min(),self.temkey.DEPTH.max()
        zscaleFactor=horrange/(zmax-zmin)
        
        x,y=emap(nocldf.LON.values,nocldf.LAT.values)
        emap.plot(x,y,'.',color=self.nonClustColor,ms=6.0)
        latdi,londi=[abs(latmax-latmin),abs(lonmax-lonmin)] #get maximum degree distance for setting scalable ticks   
        maxdeg=max(latdi,londi)
        parallels = np.arange(0.,80,maxdeg/4)
        emap.drawparallels(parallels,labels=[1,0,0,1])
        meridians = np.arange(10.,360.,maxdeg/4)
        emap.drawmeridians(meridians,labels=[1,0,0,1])        
        
        plt.figure()
        plt.plot(x,nocldf.DEPTH*zscaleFactor,'.',color=self.nonClustColor,ms=6.0)
        plt.yticks(np.linspace(zmin*zscaleFactor,zmax*zscaleFactor,10),['%0.1f'% x1 for x1 in np.linspace(zmin,zmax,10)])
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.ylabel('Depth (km)')
        plt.xlabel('Longitude')
        
        plt.figure()
        plt.plot(y,nocldf.DEPTH*zscaleFactor,'.',color=self.nonClustColor,ms=6.0)
        for a in range(len(self.clusts)):
                plt.figure(1)
                x,y=emap(temDFs[a].LON.values,temDFs[a].LAT.values)
                emap.plot(x,y,'.',color=self.clustColors[a])
                plt.figure(2)
                plt.plot(x,temDFs[a].DEPTH*zscaleFactor,'.',color=self.clustColors[a])
                plt.figure(3)
                plt.plot(y,temDFs[a].DEPTH*zscaleFactor,'.',color=self.clustColors[a])
        #make labels
        plt.yticks(np.linspace(zmin*zscaleFactor,zmax*zscaleFactor,10),['%0.1f'% x1 for x1 in np.linspace(zmin,zmax,10)])
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.ylabel('Depth (km)')
        plt.xlabel('Lattitude')
                
    def simMatrix(self,groupClusts=False,savename=False,returnMat=False,**kwargs):
        """
        Function to create basic similarity matrix of the values in the cluster object
        
        Parameters
        -------
        groupClusts : boolean
            If True order by clusters on the simmatrix with the singletons coming last
        savename : str or False
            If not False, a path used by plt.savefig to save the current figure. The extension is necesary for specifying format. See plt.savefig for details
        returnMat : boolean 
            If true return the similarity matrix
        """
        if groupClusts: #if grouping clusters together
            clusts=copy.deepcopy(self.clusts) # get cluster list 
            clusts.append(self.singles) #add singles list at end
            eveOrder=list(itertools.chain.from_iterable(clusts)) #Flatten cluster list
            indmask={num:self.key.index(eve) for num,eve in enumerate(eveOrder)} # create a mask forthe order
        else:
            indmask={x:x for x in range(len(self.key))} # blank index mask if not
        plt.figure()
        le=self.DFcc.columns.values.max()
        mat=np.zeros((le+1,le+1))
        #deb([le,indmask,self.DFcc])
        for a in range(le+1):
            for b in range(le+1):
                if a==b:
                    mat[a,b]=1
                else:
                    a1,b1=indmask[a],indmask[b] #new a and b coords based on mask
                    gi=max(a1,b1)
                    li=min(a1,b1)
                    mat[a,b]=self.DFcc.loc[li,gi]
                    mat[b,a]=self.DFcc.loc[li,gi]
                    
        cmap=mpl.colors.LinearSegmentedColormap.from_list('my_colormap',['blue','red'],256)
        img=plt.imshow(mat,interpolation='nearest',cmap = cmap,origin='upper',vmin=0,vmax=1)
        plt.clim(0,1)
        plt.grid(True,color='white')
        plt.colorbar(img,cmap=cmap)
        plt.title(self.station)
        if savename:
            plt.savefig(savename,**kwargs)
        if returnMat:
            return mat
    
        
    def write(self): #uses pickle to write class to disk
        cPickle.dump(self,open(self.filename,'wb'))
        
    def printAtr(self): #print out basic attributes used to make cluster
        print('%s Cluster'%self.station)
        print ('%d Events cluster out of %d'%(self.clustcount,len(self.singles)+self.clustcount))
        print('Total number of clusters = %d' % len(self.clusts))
        print ('Required Cross Correlation Coeficient = %.3f' % self.CCreq)
        
    def __getitem__(self,index): # allow indexing
        return self.clusts[index]
    def __iter__(self): # make class iterable
        return iter(self.clusts)
    def __len__(self): 
        return len(self.clusts)   
#    def __repr__(self):
#        self.printAtr()
#        return ''
        
        
class SubSpaceStream(object):
    """ Class used to hold subspaces for detector
    Holds both subspaces (as defined from the SScluster object) and single event clusters, or singles
    """
    
    def __init__(self, singlesDict, subSpaceDict, cl, dtype, Pf, cfetcher):
        self.cfetcher = cfetcher
        self.clusters=cl
        self.subspaces=subSpaceDict
        self.singles=singlesDict
        self.dtype=dtype
        self.Pf=Pf
        self.ssStations=self.subspaces.keys()
        self.singStations=self.singles.keys()     
        self.Stations=list(set(self.ssStations) | set(self.singStations))
        self.Stations.sort()
        self._stakey2={x:x for x in self.ssStations}
        self._stakey1={x.split('.')[1]:x for x in self.ssStations}
        
    def _checkSelection(self,selectCriteria,selectValue,Threshold): #make sure user defined values are acceptable
        if selectCriteria in [1,2,3]:
            if selectValue>1 or selectValue<0:
                raise Exception('When selectCriteria==%d selectValue must be a float between 0 and 1'%selectCriteria)
        elif selectCriteria ==4:
            if selectValue<0 or not isinstance(selectValue,int):
                raise Exception('When selectCriteria==3 selectValue must be an integer greater than 0')
        else:
            raise Exception('selectCriteria of %s is not supported' % str(selectCriteria))
        if not Threshold==None: 
            if not isinstance(Threshold,numbers.Number) or Threshold<0:
                raise Exception ('Unsupported type for Threshold, must be None or float between 0 and 1')
    
    
        
    def SVD(self,selectCriteria=2,selectValue=0.9,conDatNum=100,Threshold=None,normalize=False,useSingles=True,**kwargs): 
        """
        Function to perform SVD on the alligned waveforms and select which of the SVD basis are to be used in event detection
        Also assigns a detection threshold to each subspace-station pair 
        
        Parameters
        ----------------
        
        selctionCriteria : int, selectValue : number
            selectCriteria is the method for selecting which basis vectors will be used as detectors. selectValue depends
            on selectCriteria Options are:
        
            0 - using the given Pf, find number of dimensions to maximize detection probability !!! NOT YET IMPLIMENTED!!! 
                    selectValue - Not used 
                    (Need to find a way to estimate the doubly-non central F distribution in python to implement)
                    
            1 - Using a required dimension of representation and the given Pf estimate the effect dimension of representation 
            (the _estimateDimSpace method is called, it needs more testing) and use the central F distribution of the null space
            Harris 2006 EQ 20, find the required threshold
                selectValue - Average factional energy captured (ranges from o to 1)
            NOTE: THIS METHOD IS NOT YET STABLE
                    
            2 - select basis number based on an average fractional signal energy captured (see Figure 8 of Harris 2006)
            Then calculate an empirical distribution of the detection statistic (of the null space in order to get a 
            threshold for each subspace station pair using conDatNum of continuous data chunks without high amplitude signal 
            (see getFAS method)
                    selectValue - Average fractional energy captured can range from 0 (use no basis vectors) to 1 
                    (use all basis vectors). Harris uses 0.8. T
                
            3 - select basis number based on an average fractional signal energy captured (see Figure 8 of Harris 2006)
            Then set detection Threshold to a percentage of the minimum fractional energy captured. This method is a bit 
            quick and dirty but ensures all events in the waveform pool will be detected
                select value is a fraction representing the fraction of the minum fractional energy captured.
            
            4 - use a user defined number of basis vectors, beginning with most significant (Barrett and Beroza 2014 use first two basis 
            vectors as a "empirical" subspace detector). Then use same the value of Pf and the technique in method one to set threshold
                    selectValue - can range from 0 to number of events in subspace, if selectValue is greater than number of events use all events
            
        conDatNum : int
            The number of continuous data chunks to use to estimate the effective dimension of the signal space or to estimate the null distribution.
            Used if selectCriteria == 1,2,4
        Threshold : float or None
            Used to set each subspace at a user defined threshold. If not None overrides any of the previously defined method, avoids estimating 
            effective dimension of representation or distribution of the null space
        normalize : boolean
            If true normalize the amplitude of all the training events before preforming the SVD. Keeps higher amplitude events from dominating
            the SVD vectors but can over emphasize noise. Haris 2006 recomends the normalization but personal experience has found normalization
            can increase the detector's propensity to return false detections 
        useSingles : boolean
            If true also calculate the thresholds for singles
        """
        
        
        self._checkSelection(selectCriteria,selectValue,Threshold) # make sure user defined options are kosher
        for station in self.ssStations: #Iterate through all subspaces defined by cluster object
            for row in self.subspaces[station].iterrows(): #iterate through all stations in the subspace
                self.subspaces[station].UsedSVDKeys[row[0]]=[]
                svdDict={} #initialize empty dict to put all SVD vects in
                keys=row[1].Events
                keys.sort()                
                Arr,basisLength=self._trimGroups(row,keys,station) #get trimmed groups
                if normalize:
                    ArrNorm=np.array([x/np.linalg.norm(x) for x in Arr])
                    U, s, Vh = scipy.linalg.svd(np.transpose(ArrNorm), full_matrices=False) #perform SVD
                else:
                    try:
                        U, s, Vh = scipy.linalg.svd(np.transpose(Arr), full_matrices=False)     
                    except:
                        deb([station,row,Arr,keys])
                for eival in range(len(s)): #make dictionary of SVD with singular value as key and basis vector as value
                    svdDict[s[eival]]=U[:,eival]
                    
                #asign Parameters back to subspace dataframes
                self.subspaces[station].SVD[row[0]]=svdDict #assign SVD
                fracEnergy=self._getFracEnergy(row,svdDict,U)
                self.subspaces[station].FracEnergy[row[0]]=fracEnergy
                self.subspaces[station].UsedSVDKeys[row[0]]=self._getUsedBasis(row,svdDict,self.subspaces[station].FracEnergy[row[0]],selectCriteria,selectValue)
                self.subspaces[station].SVDdefined[row[0]]=True
                self.subspaces[station].NumBasis[row[0]]=len(self.subspaces[station].UsedSVDKeys[row[0]])
        if len(self.ssStations)>0:
            self._setThresholds(selectCriteria,selectValue,conDatNum,Threshold,basisLength)
        if len(self.singStations)>0:
            self.setSinglesThresholds(conDatNum=conDatNum,Threshold=Threshold)
                    
    def _setThresholds(self,selectCriteria,selectValue,conDatNum,Threshold,basisLength):
        if Threshold>0:
            for a,station in enumerate(self.ssStations): #Iterate through all subspaces defined by cluster object
                subspa=self.subspaces[station]
                for row in subspa.iterrows():
                    self.subspaces[station].Threshold[row[0]]=Threshold
        elif selectCriteria == 1:
            raise Exception('selectCriteria 1 currently not supported')
            for a,station in enumerate(self.ssStations):
                subspa=self.subspaces[station]
                basisdims= [len(x) for x in subspa.UsedSVDKeys.values]
                detex.log(__name__,'Estimating effective dimesion of signal space for subspace %d' %a,pri=1)
                sigdimreps= _estimateDimSpace(subspa,conDatNum,self.clusters,ConDir=self.condir,dtype=self.dtype) #estiamte dimension of signal space
                Thresholds=[]
                for badim,sigdim in zip(basisdims,sigdimreps):
                    Thresholds.append([1/((scipy.stats.f.isf(self.Pf,dfn=badim,dfd=sigdim-badim)*(badim)/(sigdim-badim))**(-1)-1)]) #reworking eqaution 20 of Harris 2006 should yeild this expression
                self.subspaces[station].Threshold=Thresholds
                self.subspaces[station].SigDimRep=sigdimreps
                
        elif selectCriteria in [2,4]:
            self.getFAS(conDatNum,staltalimit=8.0)
            for a,station in enumerate(self.ssStations):
                subspa=self.subspaces[station]
                for rownum,row in subspa.iterrows():
                    beta_a,beta_b=row.FAS['betadist'][0:2]
                    TH=scipy.stats.beta.isf(self.Pf,beta_a,beta_b,0,1) #get threshold
                    if TH>.9:
                        TH,Pftemp=self._approximateThreshold(beta_a,beta_b,self.Pf,1000,3)
                        
                        detex.log(__name__,'Scipy.stats.beta.isf failed with pf=%e, approximated Threshold to %f with a Pf of %e for station %s %s using forward grid search'%(self.Pf,TH,Pftemp,station,row.Name))

                    self.subspaces[station].Threshold[rownum]=TH
                
        elif selectCriteria == 3:
            for a,station in enumerate(self.ssStations):
                subspa=self.subspaces[station]
                for rownum,row in subspa.iterrows():
                    self.subspaces[station].Threshold[rownum]=row.FracEnergy['Minimum'][row.NumBasis]*selectValue
                    

    def _approximateThreshold(self,beta_a,beta_b,target,numintervals,numloops):
        """
        Because scipy.stats.beta.isf can break, if it returns a value near 1 when this is obvious wrong initialize grid search algo to get close
        to desired threshold using forward problem which seems to work where inverse fails
        See this bug report: https://github.com/scipy/scipy/issues/4677
        """
        startVal,stopVal=0,1     
        loops=0
        while loops<numloops:
            Xs=np.linspace(startVal,stopVal,numintervals)
            pfs=np.array([scipy.stats.beta.sf(x,beta_a,beta_b) for x in Xs])
            resids=abs(pfs-target)
            minind=resids.argmin()
            bestPf=pfs[minind]
            bestX=Xs[minind]
            startVal,stopVal=Xs[minind-1],Xs[minind+1]
            loops+=1
            if minind==0 or minind==numintervals-1:
                detex.log(__name__,'Grind search failing, set threshold manually',level='warning')
                raise Exception ('Grind search failing, set threshold manually')
        return bestX,bestPf
            
    def _trimGroups(self,row,keys,station): # basic function to get trimed subspaces
        if 'Starttime' in row[1].SampleTrims.keys() and 'Endtime' in row[1].SampleTrims.keys():
            stim=row[1].SampleTrims['Starttime']
            etim=row[1].SampleTrims['Endtime']
            if stim < 0: #make sure stim is not less than 0 #TODO impliment more robust fix
                stim = 0 
            Arr=np.vstack([row[1].AlignedTD[x][stim:etim]-np.mean(row[1].AlignedTD[x][stim:etim]) for x in keys])
            basisLength=Arr.shape[1]
        else:
            detex.log(__name__,'No trim times for %s and station %s, try running pickSubSpaceTimes'%(row[1].Name,station),pri=1)
            Arr=np.vstack([row[1].AlignedTD[x]-np.mean(row[1].AlignedTD[x]) for x in keys])
            basisLength=Arr.shape[1]
        return Arr,basisLength               
                    
    def _getSNRsByDim(self,aveFracEnergy,threshs,svdDict,sigDimRep):
        pass ##TODO figure out how to calculate the cumulative doubly non-central F distribution in python
                    
                
#    def _getThresholds(self,row,sigDimRep,svdDict): # calculated required thresholds as a function of dimensions
#        dims=range(0,len(svdDict.keys())+1)
#        threshs=[0]*len(dims)
#        for num,dim in enumerate(dims):
#            threshs[num]=1/((scipy.stats.f.isf(self.Pf,dim,sigDimRep-dim))**(-1)-1)
#        return threshs
            
           
    def _getUsedBasis(self,row,svdDict,cumFracEnergy,selectCriteria,selectValue):# function to populate  the keys of the selected SVD basis vectors
        keys=svdDict.keys()
        keys.sort(reverse=True)
        if selectCriteria in [1,2,3]:
            cumFracEnergy['Average'][-1]=1.00 #make sure last element is exactly 1
            ndim=np.argmax(cumFracEnergy['Average']>=selectValue)
            selKeys=keys[:ndim] #selected keys
        if selectCriteria==4:
            selKeys=keys[:selectValue+1]
        return selKeys
        
    def _getFracEnergy(self,row,svdDict,U): #calculate the percentage of energy that is repressented for each mp template in the SVD basis 
        fracDict={}
        keys=row[1].Events
        svales=svdDict.keys()
        svales.sort(reverse=True)
        for key in keys:
            if 'Starttime' in row[1].SampleTrims.keys() and 'Endtime' in row[1].SampleTrims.keys():
                aliwf=row[1].AlignedTD[key][row[1].SampleTrims['Starttime']:row[1].SampleTrims['Endtime']]
            else:
                aliwf=row[1].AlignedTD[key]
            try:
                repvect=np.insert(np.square(scipy.dot(np.transpose(U),aliwf)/scipy.linalg.norm(aliwf)),0,0)
            except:
                deb([U,aliwf,key,row,svdDict])
            cumrepvect=[np.sum(repvect[:x+1]) for x in range(len(repvect))]
            fracDict[key]=cumrepvect
        fracDict['Average']=np.average([fracDict[x] for x in keys],axis=0)
        fracDict['Minimum']=np.min([fracDict[x] for x in keys],axis=0)
        return(fracDict)      
       
    def write(self,filename='subspace.pkl'): #uses cPickle to write class to disk
        """
        pickle the subspace class
        Parameters
        -------------
        filename : str
            Path of the file to be created
        """
        cPickle.dump(self,open(filename,'wb'))

    def plotThresholds(self,conDatNum,xlim=[-.01,.5],**kwargs):
        """
        Function sample the continuous data and plot the calculated thresholds agaisnt a histogram of detection statistics
        created using random samples with no high amplitude signals
        
        Parameters
        ------
        
        conDatNum : int
            The number of continuous data chunks to use in the sampling
        xlim : list (number,number)
            The x limits on the plot
        
        **kwargs to pass to the getFAS call
        """

        self.getFAS(conDatNum,**kwargs)
        count=0
        #for subs in enumerate(self.subspaces):
        for station in self.ssStations:
            for rownum,row in self.subspaces[station].iterrows():
                beta_a,beta_b=row.FAS['betadist'][0:2]
                plt.figure(count)
                plt.subplot(2,1,1)
                bins=np.mean([row.FAS['bins'][1:],row.FAS['bins'][:-1]],axis=0)
                plt.plot(bins,row.FAS['hist'])
                plt.title('Station %s %s'%(station,row.Name))
                
                plt.axvline(row.Threshold,color='g')
                beta=scipy.stats.beta.pdf(bins,beta_a,beta_b)
                plt.plot(bins,beta*(max(row.FAS['hist'])/max(beta)),'k')
                plt.title('%s station %s'%(row.Name,row.Station))
                plt.xlim(xlim)
                plt.ylabel('Count')
                
                plt.subplot(2,1,2)
                bins=np.mean([row.FAS['bins'][1:],row.FAS['bins'][:-1]],axis=0)
                plt.plot(bins,row.FAS['hist'])
                plt.axvline(row.Threshold,color='g')
                plt.plot(bins,beta*(max(row.FAS['hist'])/max(beta)),'k')
                plt.xlabel('Detection Statistic')
                plt.ylabel('Count')
                plt.semilogy()
                plt.ylim(ymin=10**-1)
                plt.xlim(xlim)
                count +=1
        


    def plotFracEnergy(self):
        """
        Method to plot the fractional energy captured of by the subspace at various dimensions of rep.
        Each event is plotted as a grey dotted line, the average as a red solid line, and the chosen
        degree of rep is plotted as a solid green vertical line.
        Similar to Harris 2006 Fig 8
        """
        for a,station in enumerate(self.ssStations):
            f=plt.figure(a+1)
            f.set_figheight(1.85*len(self.subspaces[station]))
            for num,row in self.subspaces[station].iterrows():
                #CS=self.subspaces[station].iloc[b] #current series
                if not isinstance(row.FracEnergy,dict):
                    raise Exception('fractional energy not defiend, run SVD')
                plt.subplot(len(self.subspaces[station]),1,num+1)
                for event in row.Events:
                    plt.plot(row.FracEnergy[event],'--',color='0.6')
                plt.plot(row.FracEnergy['Average'],'r')
                plt.axvline(row.NumBasis,0,1,color='g')
                plt.ylim([0,1.1])
                plt.title('Station %s, %s'%(row.Station,row.Name))  
            f.subplots_adjust(hspace=.4)
            f.text(0.5, 0.06, 'Dimension of Representation', ha='center')
            f.text(0.04, 0.5, 'Fraction of Energy Captured', va='center', rotation='vertical')
        #f.tight_layout()
            plt.show()         
  
    def plotAlignedEvents(self): #plot aligned subspaces in SubSpaces object
        """ Plots the aligned events for each station in each cluster. Will trim if trim times (pick times) have been made
        """
        for a,station in enumerate(self.ssStations):        
            f=plt.figure(a)
            f.set_figheight(1.85*len(self.subspaces[station]))
            for row in self.subspaces[station].iterrows():
                plt.subplot(len(self.subspaces[station]),1,row[0]+1)
                events=row[1].Events
                for b in range(len(events)):
                    if 'Starttime' in row[1].SampleTrims.keys() and 'Endtime' in row[1].SampleTrims.keys():
                        aliwf=row[1].AlignedTD[events[b]][row[1].SampleTrims['Starttime']:row[1].SampleTrims['Endtime']]
                    else:
                        aliwf=row[1].AlignedTD[events[b]]
                    plt.plot(aliwf/(2*max(aliwf))+2*b)
                    plt.xlim([0,len(aliwf)])
                plt.title(row[1].Station)
                plt.xticks([])
                plt.yticks([])
                plt.title('Station %s, %d' % (station,row[1].Name))
            #f.suptitle() #plot does not look great, fix when possible
            plt.show()
                
    def plotBasisVectors(self,onlyused=True):
        """
        Plots the basis vectors selected after performing the SVD
        If SVD is not yet run will throw error
        
        Parameters
        ------------
        onlyUsed : boolean
            If true only the selected basis vectors will be plotted. If false all will be plotted (used in blue, unused in red)
        """
        if not self.subspaces.values()[0].iloc[0].SVDdefined:
            raise Exception('SVD not yet defined, call SVD before plotting basis vectors')
        for subnum,station in enumerate(self.ssStations):
            subsp=self.subspaces[station]
            plt.figure(subnum)
            for rownum,row in subsp.iterrows():
                plt.subplot(len(subsp),1,rownum)
                plt.title('%s station %s' % (row.Name,row.Station))
                if not onlyused:
                    keyz=row.SVD.keys()
                    keyz.sort(reverse=True)
                    for keynum,key in enumerate(keyz):
                        plt.plot(row.SVD[key]/(2*max(row.SVD[key]))-keynum,'r')
                
                for keynum,key in enumerate(row.UsedSVDKeys):
                    plt.plot(row.SVD[key]/(2*max(row.SVD[key]))-keynum,'b')
            plt.tight_layout()
            plt.yticks([])
                        
    def plotOffsetTimes(self):
        """
        Simple function to loop through each station/subspace pair and make histograms of offset times
        """
        count=0
        for station in self.ssStations:
            for num,row in self.subspaces[station].iterrows():
                if len(row.SampleTrims.keys())<1:
                    raise Exception('subspaces must be trimmed before calling this method')
                plt.figure(count)

                keys=row.Events
                offsets=[row.Stats[x]['offset'] for x in keys]
                plt.hist(offsets)
                plt.title('%s %s'%(row.Station,row.Name))
                plt.figure(count+1)
                numEvs=len(row.Events)
                ranmin=np.zeros(numEvs)
                ranmax=np.zeros(numEvs)
                orsamps=np.zeros(numEvs)
                for evenum,eve in enumerate(row.Events):
                    tem=self.clusters.temkey[self.clusters.temkey.NAME==eve].iloc[0]
                    condat=row.AlignedTD[eve]/max(2*abs(row.AlignedTD[eve]))+evenum+1
                    Nc,Sr=row.Stats[eve]['Nc'],row.Stats[eve]['sampling_rate']
                    starTime=row.Stats[eve]['starttime']
                    ortime=obspy.core.UTCDateTime(tem.TIME).timestamp
                    orsamps[evenum]=row.SampleTrims['Starttime']-(starTime-ortime)*Nc*Sr
                    plt.plot(condat,'k')
                    plt.axvline(row.SampleTrims['Starttime'],c='g')
                    plt.plot(orsamps[evenum],evenum+1,'r*')
                    ran=row.SampleTrims['Endtime']-orsamps[evenum]
                    ranmin[evenum]=orsamps[evenum]-ran*.1
                    ranmax[evenum]=row.SampleTrims['Endtime']+ran*.1
                plt.xlim(int(min(ranmin)),int(max(ranmax)))
                plt.axvline(min(orsamps),c='r')
                plt.axvline(max(orsamps),c='r')
                count+=2
                
                   
    def printOffsets(self):
        """
        Function to print out the offset min max and ranges for each station/subpace pair
        """
        for station in self.ssStations:
            for num,row in self.subspaces[station].iterrows():
                print('%s,%s, min=%3f,max=%3f, range=%3f' %(row.Station,row.Name,row.Offsets[0],row.Offsets[2],row.Offsets[2]-row.Offsets[0]))
    #return DFPick    
    def _makeOpStream(self,starow,traceLimit):
        st=obspy.core.Stream()
        if 'AlignedTD' in starow[1]: #if this is a subspace
            for num,key in enumerate(starow[1].Events):
                if num<traceLimit:
                    tr=obspy.core.Trace(data=starow[1].AlignedTD[key])
                    tr.stats.channel=key
                    tr.stats.network=starow[1].Name
                    tr.stats.station=starow[1].Station
                    st+=tr
            return st
        else: #if this is a single event
            for key in starow[1].Events:
                tr=obspy.core.Trace(data=starow[1].MPtd[key])
                tr.stats.channel=key
                tr.stats.station=starow[1].Station
                st+=tr
            return st
                    
    def pickTimes(self,duration=30,traceLimit=15,repick=False,subspace=True,singles=True):
        """
        Calls a modified version of obspyck (https://github.com/megies/obspyck), a GUI for picking phases,
        so user can manually select start times (trim) of unclustered and clustered events.
        triming down each waveform group will significantly decrease the runtime on subspaces, and is required
        for single events (or else they will not be used as detectors). 
        
        Parameters
        
        --------------
        
        duration : real number
            the time after the first pick (in seconds) to trim waveforms for SVD. The fact the stream is multiplexed is taken into account.
            If None is passed then the last pick will be used as the end time for truncating waveforms
        traceLimit : int
            Limits the number of traces that will show up to be manually picked to the first in traceLimit. Avoids killing the GUI with too
            many events. 
        repick : boolean
            If true repick times that already have sample trim times.
        subspace : boolean
            If true pick subspaces
        singles : boolean
            If true pick singletons
        """     
        if subspace:
            self._pickTimes(self.subspaces,duration,traceLimit,repick=repick)
        if singles:
            self._pickTimes(self.singles,duration,traceLimit,issubspace=False,repick=repick)
        
    def _pickTimes(self,trdfDict,duration,traceLimit,issubspace=True,repick=False):
        
        """
        method to call modified obpyck to pick start and optionally stop times for aligned templates.
        This is a good idea to clip out un-needed segments of data before preforming SVD to get subspace basis
        Will also greatly speed up the subspace as a detector if basis vectors are shorter
        
        
        duration is the duration from the first arrivial pick to define waveforms
        """
        for station in trdfDict.keys():
            for starow in trdfDict[station].iterrows():
                if not starow[1].SampleTrims or repick: # If the sample trim dictionary is empty
                    TR=self._makeOpStream(starow,traceLimit) #Make an obspy stream
                    Pks=None #This is needed or it crashes OS X versions
                    Pks=detex.streamPick.streamPick(TR)
                    d1={}
                    for b in Pks._picks:
                        if b: #if any picks made
                            d1[b.phase_hint]=b.time.timestamp
                    if d1: # if any picks made
                        sr=starow[1].Stats[starow[1].Events[0]]['sampling_rate'] #fetch sampling rate
                        Nc=starow[1].Stats[starow[1].Events[0]]['Nc'] #fetch number of channels
                        d1['Starttime']=int(min(d1.values()))-(int(min(d1.values())))%Nc #get sample divisible by NC to keep traces aligned
                        if duration: #if duration paramenter is defined (it is usually better to leave it defined)
                            d1['Endtime']=d1['Starttime']+int(duration*sr*Nc) 
                            d1['DurationSeconds']=duration
                        else:
                            d1['Endtime']=int(max(d1.values()))
                            d1['DurationSeconds']=(d1['Endtime']-d1['Starttime'])/(sr*Nc)
                        trdfDict[station].SampleTrims[starow[0]]=d1
                        for event in starow[1].Events: #update starttimes
                            stime=trdfDict[station].Stats[starow[0]][event]['starttime']
                            trdfDict[station].Stats[starow[0]][event]['starttime']=stime+d1['Starttime']/(Nc*sr) 
                            trdfDict[station].Stats[starow[0]][event]['offset']=trdfDict[station].Stats[starow[0]][event]['starttime']-trdfDict[station].Stats[starow[0]][event]['origintime']
                    if not Pks.KeepGoing:
                        detex.log(__name__, 'aborting picking, progress saved',pri=1)
                        return None
            self._updateOffsets()
    
    def attachPickTimes(self,pksFile='EventPicks.csv',ssMode='Average',defaultDuration=30):
        """
        Rather than picking times manually attach a file (either csv or pkl of pandas dataframe) with pick times.
        Pick time file must have the following fields:  TimeStamp,Station,Event,Phase (as created by detex.util.pickPhases)
        Parameters. For each event finds the start time and endtime 
        ----------
        pksFile : str
            Path to the input file (either csv or pickle)
        ssMode : str; 'Average','Max', or 'Min'
            Describes how to handle selecting a common pick time for subspace group (each event in a subspace cannot be treated 
            independently as the entire group is aligned to maximize similarity)
            Average-Trims the stack to the sample corresponding to the average of the first arriving phase
            Median- Trims the stack to the sample corresponding to the median of the first arriving phase
        defaultDuration : int or None
            if Int, the default duration (in seconds) to trim the signal to starting from the first arrival in pksFile for each event
            or subspace stack. If None, then durations are defined by first arriving phase and last arriving phase for each event
        """
        try: #read pksFile
            pks=pd.read_csv(pksFile)
        except:
            try:
                pks=pd.read_pickle(pksFile)
            except:
                detex.log(__name__,'%s does not exist, or it is not a pkl or csv file' % pksFile,level='warning')
                raise Exception('%s does not exist, or it is not a pkl or csv file' % pksFile )
        
        #get appropriate function according to ssmode
        if ssMode=='Average':
            fun=np.average
        elif ssMode=='Max':
            fun=np.max
        elif ssMode=='Min':
            fun=np.min
        elif ssMode=='Median':
            fun=np.median
        else:
            detex.log(__name__,'ssMode %s not supported, options are: Average,Max,Min,Median'%ssMode,level='warning')
            raise Exception('ssMode %s not supported, options are: Average,Max,Min,Median'%ssMode)                
                
        for cl in self.clusters: #loop through each station in cluster object, get singles and subspaces
            sta=cl.station #current station
            
            ### Attach singles
            if sta in self.singles.keys():
                for num,row in self.singles[sta].iterrows():
                    if len(row.SampleTrims.keys())>0: #skip if sampletrims already defined
                        continue
                    pk=pks[(pks.Event.isin(row.Events))&(pks.Station==sta)] #phases that apply to current event and station
                    eves,starttimes,Nc,Sr=self._getStats(row)
                    if len(pk)>0:
                        trims=self._getSampleTrims(eves,starttimes,Nc,Sr,pk,ssMode,defaultDuration,fun,num,self.singles[sta])
                        if isinstance(trims,dict):
                            self.singles[sta].SampleTrims[num]=trims
                self._updateOffsets()
            ### Attach Subspaces
            if sta in self.subspaces.keys():
                for num,row in self.subspaces[sta].iterrows():
                    if len(row.SampleTrims.keys())>0: #skip if sampletrims already defined
                        continue
                    pk=pks[(pks.Event.isin(row.Events))&(pks.Station==sta)] #phases that apply to current event and station
                    eves,starttimes,Nc,Sr=self._getStats(row)
                    if len(pk)>0:
                        trims=self._getSampleTrims(eves,starttimes,Nc,Sr,pk,ssMode,defaultDuration,fun,num,self.subspaces[sta])
                        if isinstance(trims,dict):
                            self.subspaces[sta].SampleTrims[num]=trims
                self._updateOffsets()
                
    def _getSampleTrims(self,eves,starttimes,Nc,Sr,pk,ssMode,defaultDuration,fun,num,DF):
        """
        Determine sample trims for each single or subspace
        """
        #stdict={}#intialize sample trim dict
        startsamps=[]
        stopsamps=[]
        secduration=[]
            
        for ev in eves: #loop through each event
            p=pk[pk.Event==ev]
            if len(p)<1: #if event is not recorded skip
                continue
            start=p.TimeStamp.min()
            startsampsEve = (start-starttimes[ev])*(Nc*Sr) 
            if startsampsEve < 0: #make sure starting time is not less than 0 else set to zero
                startsampsEve = 0
                start = starttimes[ev]
                msg = 'Start time (from phase file) is less than 0 for event %s' % ev
                detex.log(__name__, msg, level='warning', pri=False)
            if defaultDuration:
                stop=start+defaultDuration
                secduration.append(defaultDuration)
            else:
                stop=p.TimeStamp.max()
                secduration.append(stop-start)
            assert stop>start #Make sure stop is greater than start
            assert stop>starttimes[ev]
            if stop < starttimes[ev]:
                deb([start, duration])
            endsampsEve = (stop-starttimes[ev])*(Nc*Sr)
            startsamps.append(startsampsEve)
            stopsamps.append(endsampsEve)
            #update stats attached to each event to reflect new start time
        
            DF.Stats[num][ev]['Starttime']=start
            DF.Stats[num][ev]['offset']=start-DF.Stats[num][ev]['origintime']   
        if len(startsamps)>0:
            outdict={'Starttime':int(fun(startsamps))-int(fun(startsamps))%Nc,'Endtime':int(fun(stopsamps))-int(fun(stopsamps))%Nc,'DurationSeconds':int(fun(secduration))}
            return outdict
        else:
            return
    def _getStats(self,row):
        """
        Get the sampling rate, starttime, and number of channels for each event group
        """
        eves=row.Events
        sr=[np.round(row.Stats[x]['sampling_rate']) for x in eves]
        if len(set(sr))!=1:
            detex.log(__name__,'Events %s on Staion %s have different sampling rates or no sampling rates'%(row.Station,row.events),level='warning')
            raise Exception('Events %s on Staion %s have different sampling rates or no sampling rates'%(row.Station,row.events))
        
        Nc=[row.Stats[x]['Nc'] for x in eves]
        if len(set(Nc))!=1:
            detex.log(__name__,'Events %s on Staion %s have different numbers of channels or no channels'%(row.Station,row.events),level='warning')
            raise Exception('Events %s on Staion %s have different numbers of channels or no channels'%(row.Station,row.events))
        starttimes={x:row.Stats[x]['starttime'] for x in eves}
        return eves,starttimes,list(set(Nc))[0],list(set(sr))[0]

        
    def _updateOffsets(self):
        """
        Calculate offset (predicted origin times), throw out extreme outliers using median and median scaling
        """
        for sta in self.subspaces.keys():
            for num,row in self.subspaces[sta].iterrows():
                keys=row.Stats.keys()
                offsets=[row.Stats[x]['offset'] for x in keys]
                self.subspaces[sta].Offsets[num]=self._getOffsets(np.array(offsets))
        for sta in self.singles.keys():
            for num,row in self.singles[sta].iterrows():
                keys=row.Stats.keys()
                offsets=[row.Stats[x]['offset'] for x in keys]
                self.singles[sta].Offsets[num]=self._getOffsets(np.array(offsets))
    
    def _getOffsets(self, offsets, m = 25.):
        """
        Get offsets, reject outliers bassed on median values (accounts for possible missmatch in events and origin times)
        """
        if len(offsets) == 1:
            return offsets[0],offsets[0],offsets[0]
        d = np.abs(offsets - np.median(offsets))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        try:
            offs=offsets[s<m]
        except: 
            deb([offsets])
        return [np.min(offs),np.median(offs),np.max(offs)]

        
    def setSinglesThresholds(self,conDatNum=50,recalc=False,Threshold=None):
        """
        Set thresholds for the singletons (unclustered events) by fitting a beta distribution to noise
        
        Parameters
        ----------
        condatNum : int
            The number of continuous data chunks to use to fit PDF
        recalc : boolean
            If true recalculate the the False Alarm Statistics
        Threshold : None or float between 0 and 1
            If not None, do not calculate empircal values simply use given threshold
        """
        for sta in self.singStations:
            self.singles[sta]=self.singles[sta][[len(x.keys())>0 for x in self.singles[sta].SampleTrims]] #delete any singles that do not have pick times
            self.singles[sta].reset_index(inplace=True,drop=True)
            self.singles[sta].Name=['SG%d' %x for x in range(len(self.singles[sta]))]
        if not Threshold: self.getFAS(conDatNum,useSingles=True,useSubSpaces=False) #get empirical dist unless manual threshold is passed
        for sta in self.singStations:
            for num,row in self.singles[sta].iterrows():
                if len(row.SampleTrims)<1: #skip singles with no pick times
                    continue
                if Threshold:
                    TH=Threshold
                else:
                    beta_a,beta_b=row.FAS[0]['betadist'][0:2]
                    TH=scipy.stats.beta.isf(self.Pf,beta_a,beta_b,0,1) #get threshold
                    if TH>.9:
                        TH,Pftemp=self._approximateThreshold(beta_a,beta_b,self.Pf,1000,3)
                        #print 'Scipy.stats.beta.isf failed with pf=%e, approximated Threshold to %f with a Pf of %e for station %s %s using forward grid search'%(self.Pf,TH,Pftemp,row.Station,row.Name)
                self.singles[sta]['Threshold'][num]=TH
                #print TH
        
    def getFAS(self,conDatNum,ConDir='ContinuousWaveForms',LTATime=5,STATime=0.5,staltalimit=8.0,useSubSpaces=True,useSingles=False,numBins=401,recalc=False):
        
        """ 
        Function to initialize a FAS instance, used primarily for smapling the detection distribution of the different subspaces and singles
        
        conDatNum is the number of continuous data files (by default in hour chunks) to use
        
        ConDir is the directory in the detex.getdata format where contious waveforms are located
        
        LTATime is the long term average time window in seconds used for checking continuous data
        
        STATime is the short term average time window in seconds for checking continuous data
        
        staltalimit is the value at which continuous data gets rejected as too noisey (IE transient signals are present)
        
        continuousDataLength is the length, in seconds, of the longest continuous data chunk, by defualt 1 hour 2 minutes is used to avoid skipping any data
        
        useSubspace and useSingles are both booleans to indicate if the action should be preformed for subspaces, singles, or both
        
        numBins is for storing distribution data, the number of bins between -1 and 1 that will be used in the histogram
        
        """
        
        if useSubSpaces:
            self._updateOffsets() #make sure offset times are up to date
            for station in self.subspaces.keys():
                if isinstance(self.subspaces[station]['FAS'][0],dict) and not recalc:
                    print('FAS for station %s already calculated, to recalculate pass True to the parameter recalc' % station)
                else:
                    self.subspaces[station]['FAS']=detex.fas._initFAS(self.subspaces[station],conDatNum,self.clusters,self.cfetcher,
                    LTATime=LTATime,STATime=STATime,staltalimit=staltalimit,numBins=numBins,dtype=self.dtype)
        if useSingles:
            #raise Exception('Singles not yet implemented')
            for station in self.singles.keys():
                for a in range(len(self.singles[station])):
                    if isinstance(self.singles[station]['FAS'][a],dict) and not recalc:
                        print('FAS for single event %d already calculated, to recalculate pass True to the parameter recalc' % a)
                    elif len(self.singles[station]['SampleTrims'][a].keys())<1: #skip any events that have not been trimmed
                        continue
                    else:
                        self.singles[station]['FAS'][a]=detex.fas._initFAS(self.singles[station][a:a+1],conDatNum,self.clusters,self.cfetcher,
                        LTATime=LTATime,STATime=STATime,staltalimit=staltalimit,numBins=numBins,dtype=self.dtype,issubspace=False)
                        
    def detex(self,UTCstart=None,UTCend=None,subspaceDB='SubSpace.db',trigCon=0,triggerLTATime=5,triggerSTATime=0,
        multiprocess=False,delOldCorrs=True,extrapolateTimes=True, calcHist=True,useSubSpaces=True,useSingles=False,
        estimateMags=True,pks=None,eventDir=None,eventCorFile='EventCors.pkl',UTCSaves=None,fillZeros=False):
        """
        function to run subspace detection over continuous data and store results in SQL database subspaceDB
        
        Parameters
        ------------
        UTCstart : str or num
            An obspy.core.UTCDateTime readable object defining the start time of the correlations if not all avaliable data
            are to be used
        UTCend : str num
            An obspy.core.UTCDateTime readable object defining the end time of the correlations
        subspaceDB : str
            path to the SQLite database to store detections in. If it already exists delOldCorrs parameters
            governs if it will be deleted
        trigCon is the condition for which detections should trigger. Once the condition is set the variable minCoef is used
            0 is based on the detection statistic threshold
            1 is based on the STA/LTA of the detection statistic threshold (does not currently support FAS to set this parameter)    
        triggerLTATime : number
            The longterm average for the STA/LTA calculations in seconds
        triggerSTATime : number
            The short term average for the STA/LTA calculations in seconds. If ==0 then one sample is used
        multiprocess : boolean 
            Determine if each station should be forked into its own process for huge potential speed savings. Currently doesn't work        
        delOldCorrs : boolean 
            Determines if subspaceDB should be deleted silently (without user input) before current detections are processed
        extrapolateTimes : boolean
            Enable cosine subsample exptrapolation for each detection
        calcHist : boolean
            If True calculates the histagram for every point of the detection statistic vectors (all hours, stations and subspaces). Only slows the 
            detections down slightly. The histograms are then returned to the subspace stream object as the attribute histSubSpaces
        useSubspace : boolean
            If True the subspaces will be used as detectors to scan continuous data
        useSingles : boolean
            If True the singles (events that did not cluster) will be used as detectors to scan continuous data (DOESNT WORK YET)
        estimateMags : boolean
            If True magnitudes will be estimated for each detection by projecting the continuous data where the detection occured into the subspace,
            as well as all the template waveforms that went into the subspace's creation, then applying the iterative magnitude scaling method 
            in Gibbons and Ringdal 2006. Because a representation of the singal is used (IE the projection of the continuous data into the subspace)
            this method is less vulnerable to noise distortions when estimating lower magnitudes.
        pks : None or str
            A path to the pks file of the same form as the once created by detex.util.trimTemplates that will be used for estimating P and S arrival times
            for each detection. If False, or the path is not readable with the pandas.read_pickle method no estimations will not be made
        eventDir : None or str
            If a path to an event directory is passed (IE EventWaveForms) the subspace detector will be run on all the events in the eventwaveform
            file rather than on the continuous data directory
        eventCorFile : str
            A path to a new pickled data frame created when the eventDir option is used. Records the highest detection statistic in the file for each 
            event, station, and subspace. Useful when trying to characterize events. 
        UTCSaves : None or list of obspy.core.DateTime readable objects
            Either none (no effect) or an iterrable  of obspy.core.UTCDateTime readable objects. For continuous data chunk being scanned if a time in the 
            UTCdate falls within the start time and end time of the continuous data the vector of DS, along with cotninous data, thresholds, etc. is save to
            a pickled dataframe of the name "UTCsaves.pkl"
        fillZeros : boolean
            If true fill the gaps in continuous data with 0s. If True STA/LTA of detection statistic cannot be calculated in order to avoid dividing by 0            
        """
    
        if multiprocess or trigCon!=0: #make sure no parameters that dont work yet are selected
            detex.log(__name__,'multiprocessing and trigcon other than 0 not yet supported',level='warning')
            raise Exception ('multiprocessing and trigcon other than 0 not yet supported')
        
        if os.path.exists(subspaceDB) and not delOldCorrs: #If old database is around delete it
            user_input=raw_input('%s already exists, delete it? ("y" or "yes" else do not delete)\n'%subspaceDB)
            if user_input=='yes' or user_input=='y':
                detex.util.DoldDB(subspaceDB)    
                
        elif delOldCorrs and os.path.exists(subspaceDB):
            detex.util.DoldDB(subspaceDB)
            
        if useSubSpaces:
            TRDF=self.subspaces
            if not all([y['SVDdefined'] for x in TRDF.keys() for num,y in self.subspaces[x].iterrows()]): #make sure SVD has been performed
                raise Exception('subspace not yet defined, call SVD before attempting to run subspace detectors')
            Det=SSDetex(TRDF,UTCstart,UTCend,self.cfetcher,self.clusters,subspaceDB,trigCon,triggerLTATime,triggerSTATime,
            multiprocess,self.clusters.filt,self.clusters.decimate,extrapolateTimes,calcHist,self.dtype,estimateMags,
            pks,eventDir,eventCorFile,UTCSaves,fillZeros)
            self.histSubSpaces=Det.hist
            
        if useSingles:
            TRDF=self.singles
            Det=SSDetex(TRDF,UTCstart,UTCend,self.cfetcher,self.clusters,subspaceDB,trigCon,triggerLTATime,triggerSTATime,
            multiprocess,self.clusters.filt,self.clusters.decimate,extrapolateTimes,calcHist,self.dtype,estimateMags,
            pks,eventDir,eventCorFile,UTCSaves,fillZeros,issubspace=False)
            self.histSingles=Det.hist
            
        if useSubSpaces or useSingles: # save addational info to sql database
        
            dffilt=pd.DataFrame([self.clusters.filt],columns=['FREQMIN','FREQMAX','CORNERS','ZEROPHASE'],index=[0]) #save filter info
            detex.util.saveSQLite(dffilt,subspaceDB,'filt_params')
            
            ssinfo,sginfo=self._getInfoDF() #get general info on each single/subspace
            sshists,sghists=self._getHistograms(useSubSpaces,useSingles)
            
            if useSubSpaces and len(ssinfo)>0: detex.util.saveSQLite(ssinfo,subspaceDB,'ss_info') #save subspace info
            if useSingles and len(sginfo>0): detex.util.saveSQLite(sginfo,subspaceDB,'sg_info') #save singles info
            if useSubSpaces and len(sshists>0): detex.util.saveSQLite(sshists,subspaceDB,'ss_hist') #save subspace histograms
            if useSingles and len(sghists>0): detex.util.saveSQLite(sghists,subspaceDB,'sg_hist') #save singles histograms
            
    def _getInfoDF(self):
        """
        get dataframes that have info about each subspace and single
        """
        sslist=[] #empty list in which to put DFs for each subspace/station pair
        sglist=[] #empty listin which  to put DFs for each single/station pair
        for sta in self.Stations:
            if not sta in self.ssStations:
                continue
            for num, ss in self.subspaces[sta].iterrows(): #write the subspace info
                name=ss.Name
                station=ss.Station
                events=','.join(ss.Events)
                numbasis=ss.NumBasis
                thresh=ss.Threshold
                if len(ss.FAS.keys())>1:
                    beta1,beta2=ss.FAS['betadist'][0],ss.FAS['betadist'][1]
                else:
                    beta1,beta2=np.Nan,np.Nan
                sslist.append(pd.DataFrame([[name,station,events,thresh,numbasis,beta1,beta2]],columns=['Name','Sta','Events','Threshold','NumBasisUsed','beta1','beta2']))
            if not sta in self.singStations:
                continue
            for num, ss in self.singles[sta].iterrows(): #write the singles info
                name=ss.Name
                station=ss.Station
                events=','.join(ss.Events)
                thresh=ss.Threshold
                if len(ss.FAS[0].keys())>1:
                    beta1,beta2=ss.FAS[0]['betadist'][0],ss.FAS[0]['betadist'][1]
                else:
                    beta1,beta2=np.Nan,np.Nan
                sglist.append(pd.DataFrame([[name,station,events,thresh,beta1,beta2]],columns=['Name','Sta','Events','Threshold','beta1','beta2']))
        ssinfo=pd.concat(sslist,ignore_index=True)
        sginfo=pd.concat(sglist,ignore_index=True)
        return ssinfo,sginfo
        
    def _getHistograms(self,useSubSpaces,useSingles):
        """
        Pull out the histogram info for each station-subspace or single pair
        """
        if useSubSpaces:
            sshists=[pd.DataFrame([['Bins','Bins',json.dumps(self.histSubSpaces['Bins'].tolist())]],columns=['Name','Sta','Value'])]
            for sta in self.Stations:
                if sta in self.histSubSpaces.keys():
                    for skey in self.histSubSpaces[sta]:
                        sshists.append(pd.DataFrame([[skey,sta,json.dumps(self.histSubSpaces[sta][skey].tolist())]],columns=['Name','Sta','Value']))

            sshist=pd.concat(sshists,ignore_index=True)
        else:
            sshist=None
        if useSingles:
            sghists=[pd.DataFrame([['Bins','Bins',json.dumps(self.histSingles['Bins'].tolist())]],columns=['Name','Sta','Value'])]
            for sta in self.Stations:
                if sta in self.histSingles.keys():
                    for skey in self.histSingles[sta]:
                        sghists.append(pd.DataFrame([[skey,sta,json.dumps(self.histSingles[sta][skey].tolist())]],columns=['Name','Sta','Value']))
            sghist=pd.concat(sghists,ignore_index=True)
        else:
            sghist=None
        return sshist,sghist
      
    def __getitem__(self,key): #make object indexable
        if isinstance(key,int):
            return self.subspaces[self.ssStations[key]]
        elif isinstance(key,str):
            if len(key.split('.'))==2:
                return self.subspaces[self._stakey2[key]]
            elif len(key.split('.'))==1:
                return self.subspaces[self._stakey1[key]]
            else:
                detex.log(__name__,'%s is not a station in this cluster object' % key,level='warning')
                raise Exception('%s is not a station in this cluster object' % key)
        else:
            detex.log(__name__,'%s must either be a int or str of station name' % key,level='warning')
            raise Exception ('%s must either be a int or str of station name'%key)
            
    def __len__(self): 
        return len(self.subspaces) 
                    

class SSDetex(object):
    """
    dummy class to run subspace detections for all events in EveDir with corresponding waveforms in 
    ConDir
    """
    def __init__(self,TRDF,UTCstart,UTCend,cfetcher,clust,subspaceDB,trigCon,triggerLTATime,triggerSTATime,multiprocess,filt,
                 decimate,extrapolateTimes,calcHist,dtype,estimateMags,pks,eventDir,eventCorFile,UTCSaves,fillZeros,issubspace=True):
        self.__dict__.update(locals()) # Instantiate all input variables
        if eventDir:
            self.continuousDataLength=cfetcher.timeBeforeOrigin + cfetcher.timeAfterOrigin
        else:
            #self.continuousDataLength=detex.util.getEveDataLength(EveDir=eventDir)
            self.continuousDataLength=cfetcher.conDatDuration + cfetcher.conBuff
        if pks: #Try to read in picks
            try:
                self.pks_df=pd.read_pickle(pks)
            except:
                detex.log(__name__,'reading %s failed' % pks,level='warning')
                self.pks=None
        if self.eventDir: #if this is an event directory correlation 
            self.eventCorList=[] #initilialize blank list to store all dataframes in
        if isinstance(UTCSaves,collections.Iterable): # if using UTCsave list initialize empty list
            self.UTCSaveList=[]        
            self.UTCSaves=np.array([obspy.core.UTCDateTime(x).timestamp for x in UTCSaves])
        jobs = []
        stations= TRDF.keys() # get list of all stations repressented in current TRDF
        self.hist={} 
        self.hist['Bins']=np.linspace(0,1,num=401)
        for sta in stations: #loop through each station in all subspaces/singles
            DFsta=TRDF[sta] #make data frame from all subspaces/singles that share a station. This is done to reduce IO costs in reading continuous data
            DFsta.reset_index(inplace=True,drop=True)
            if len(DFsta)>0 and not self.eventDir:
                self.hist[sta]=[None]*len(TRDF)
                if multiprocess==True: #TODO fix this, multiprocessing is broken
                    p = multiprocessing.Process(target=self._CorStations(DFsta,sta))
                    jobs.append(p)
                    p.start()
                else:
                    self.hist[sta]=self._CorStations(DFsta,sta)
            elif self.eventDir:
                try:
                    DFeve=pd.concat(self.eventCorList,ignore_index=True)
                    DFeve.to_pickle(self.eventCorFile)
                except ValueError:
                    detex.log(__name__,'No events in Df concatenated',level='warn')
        if isinstance(UTCSaves,collections.Iterable):
            try:
                DFutc=pd.concat(self.UTCSaveList,ignore_index=True)
                DFutc.to_pickle('UTCsaves.pkl')
            except ValueError:
                detex.log(__name__,'DFutc empty, not saving',level='warning')
                
    def _CorStations(self,DFsta,sta): 
        skey = self.clust.stakey
        stakey = skey[skey.STATION == sta.split('.')[1]]
        channels=list(set([z for x in DFsta.Channels for y in x.values() for z in y]))
        channels.sort()
        samplingRate=list(set([y['sampling_rate'] for x in DFsta.Stats for y in x.values()])) #get sampling rates
        Names=DFsta.Name.values
        threshold={x.Name:x.Threshold for num,x in DFsta.iterrows()} #dictionary of required coeficients 
        Names.sort()
        histdic={na:[0.0]*(len(self.hist['Bins'])-1) for na in Names}
        if len(samplingRate)>1:  # Make sure there is only one sampling rate for all subspaces using current station station
            detex.log(__name__,'More than one sampling rate found for station %s in subspace detection, aborting' % DFsta.Station[0],level='warning')
            raise Exception('More than one sampling rate found for station %s in subspace detection, aborting' % DFsta.Station[0])
        else:
            try:
                samplingRate=samplingRate[0]
            except:
                return
                #deb([DFsta,sta])
        # get the ammount that each continuous data sample needs to be trimmed before performing ss detections
        try:
            contrim={x[1].Name:-(self.continuousDataLength % 3600)+(x[1].SampleTrims['Endtime']-x[1].SampleTrims['Starttime'])/(samplingRate*len(channels)) for x in DFsta.iterrows()}
        except:
            deb([DFsta,sta,self])
        staPksDf=None        
        if self.pks:
            staPksDf=self.pks_df[self.pks_df.Station==sta]
        if not self.eventDir:
            histdict=self._CorConDat(threshold,histdic,sta,channels,contrim,Names,DFsta,samplingRate,staPksDf, stakey)
        else:
            if os.path.exists(self.eventDir):
                histdict=self._CorEventDat(threshold,histdic,sta,channels,contrim,Names,DFsta,samplingRate,staPksDf)
            else:
                detex.log(__name__,'%s does not exist' % self.eventDir)
                raise Exception('%s does not exist' % self.eventDir)
        return histdict
    
    def _CorEventDat(self,threshold,histdic,sta,channels,contrim,Names,DFsta,samplingRate,staPksDf):
        """ 
        Function for when subspace is to be run on event waveforms
        """
        tempFiles=[]
        contrim=0
        DF=pd.DataFrame()
        numdets=0
        for event in glob.glob(os.path.join(self.eventDir,'*')):
            for staEvent in glob.glob(os.path.join(event,sta+'*')): 
                tempFiles.append(staEvent)
        if len(tempFiles)<1:
            detex.log(__name__, 'No events found for %s'%sta,level='warning')
        else:
            ssArrayTD,ssArrayFD,reqlen,offsets,mags,eventWaveForms,events,WFU,UtU=self._loadMPSubSpace(DFsta,self.continuousDataLength,sta,channels,samplingRate,returnFull=True,PKS=staPksDf)
            for fileToCorr in tempFiles: #loop through each chunk of continuous data
                CorDF,MPcon,ConDat=self._getRA(ssArrayTD,ssArrayFD,fileToCorr,len(channels),reqlen,contrim,Names)
                if not isinstance(CorDF,pd.DataFrame): #if something is broken skip hour
                    detex.log(__name__,'%s failed' % fileToCorr ,level='warning')
                    
                    continue
                for name,row in CorDF.iterrows(): # iterate through each subspace/single
                    self.eventCorList.append(pd.DataFrame([[sta,name,row.File,row.MaxDS,row.MaxSTALTA]],columns=['Station','Subspace','File','DS','DS_STALTA']))
                    if self.calcHist and len(CorDF)>0: #If calculating histogram of Statistic of detection 
                        try:
                            histdic[name]=histdic[name]+np.histogram(row.SSdetect,bins=self.hist['Bins'])[0]
                        except:
                            detex.log(__name__,'binning failed for %s'%row.Name,level='warning')
                    if isinstance(self.UTCSaves,collections.Iterable):
                       self._makeUTCSaveDF(row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD)
                    if self._evalTriggerCondition(row,name,threshold): # Trigger Condition
                        Sar=self._CreateCoeffArray(row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD,WFU,UtU,staPksDf)
                        if len(Sar)>300:
                            detex.log(__name__,'over 300 events found in single continuous data chunk, perphaps minCoef is too low?',level='warning')
                        if len(Sar)>0:
                            DF=DF.append(Sar,ignore_index=True)
                        if len(DF)>500:
                            numdets+=500
                            detex.util.saveSQLite(DF,self.subspaceDB,'ss_df')
                            DF=pd.DataFrame()
            if len(DF)>0:
                detex.util.saveSQLite(DF,self.subspaceDB,'ss_df')
            detType='Subspaces' if self.issubspace else 'Singletons'
            detex.log(__name__,'%s detections on %s completed, %d potential detection(s) recorded' %(detType,sta,len(DF)+numdets),pri=True)
        return histdic

        
            
    def _CorConDat(self,threshold,histdic,sta,channels,contrim,Names,DFsta,samplingRate,staPksDf, stakey):
        """
        Function to use when subspace is to be run over continuous data
        """
        numdets=0
        tableName='ss_df' if self.issubspace else 'sg_df'
        DF=pd.DataFrame() # Initialize emptry data frame, will later be dumped to SQL database
        ssArrayTD,ssArrayFD,reqlen,offsets,mags,eventWaveForms,events,WFU,UtU=self._loadMPSubSpace(DFsta,self.continuousDataLength,sta,channels,samplingRate,returnFull=True,PKS=staPksDf) 
        conDatGen = self.cfetcher.getConData(stakey, utcstart=self.UTCstart, utcend=self.UTCend)
        #glob.glob(os.path.join(self.ConDir,sta,str(conrangeyear[a]),str(b),sta+'*'))
        for st in conDatGen: #loop through each chunk of continuous data
        
            CorDF,MPcon,ConDat=self._getRA(ssArrayTD,ssArrayFD,st,len(channels),reqlen,contrim,Names)
            if not isinstance(CorDF,pd.DataFrame): #if something is broken skip hour                        
                detex.log(__name__,'%s failed' % fileToCorr,level='warning') 
                continue
            for name,row in CorDF.iterrows(): # iterate through each subspace/single
                if self.calcHist and len(CorDF)>0: 
                    try:
                        histdic[name]=histdic[name]+np.histogram(row.SSdetect,bins=self.hist['Bins'])[0]
                    except:
                        detex.log(__name__,'binning failed',level='warning')
                if isinstance(self.UTCSaves,collections.Iterable):
                    self._makeUTCSaveDF(row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD)
                if self._evalTriggerCondition(row,name,threshold): # Trigger Condition
                    #deb(row)
                    Sar=self._CreateCoeffArray(row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD,WFU,UtU,staPksDf)
                    if len(Sar)>300:
                        detex.log(__name__,'over 300 events found in single continuous data chunk, perphaps minCoef is too low?',level='warning')
                    if len(Sar)>0:
                        DF=DF.append(Sar,ignore_index=True)
                    if len(DF)>500:
                        detex.util.saveSQLite(DF,self.subspaceDB,tableName)
                        DF=pd.DataFrame()
                        numdets+=500
        if len(DF)>0:
            detex.util.saveSQLite(DF,self.subspaceDB,tableName)
        detType='Subspaces' if self.issubspace else 'Singletons'
        detex.log(__name__,'%s on %s completed, %d potential detection(s) recorded' %(detType,sta,len(DF)+numdets),pri=1)
        if self.calcHist:
            return histdic
            
            
        
    def _makeUTCSaveDF(self,row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD):
        TS1=row.TimeStamp
        TS2=row.TimeStamp+len(MPcon)/(row.SampRate*float(row.Nc))
        inUTCs=(self.UTCSaves>TS1)&(self.UTCSaves<TS2)
        if any(inUTCs):
            Th=threshold[name]
            of=offsets[name]
            seri=pd.Series([sta,name,Th,of,TS1,TS2,self.UTCSaves[inUTCs],MPcon],index=['Station','Name','Threshold','offset','TS1','TS2','UTCSaves','MPcon'])
            df=pd.DataFrame(pd.concat([seri,row])).T
            self.UTCSaveList.append(df)
        return
        
    def _loadMPSubSpace(self,DFsta,continuousDataLength,sta,channels,samplingRate,returnFull=False,PKS=None): # function to load subspace representations
        if 'UsedSVDKeys' in DFsta.columns and self.issubspace: #test if input TRDF row is subspace
            ssArrayTD,rele,ssArrayFD,offsets,mags,eventWaveforms,eves,WFU,UtUdict,pwave,swave={},{},{},{},{},{},{},{},{},{},{} # intitialize empty dicts
            if not all ([isinstance(x.UsedSVDKeys,list) for num,x in DFsta.iterrows()]):
                raise Exception ('SVD not defined, run SVD on subspace stream class before calling detex')
            if not all([set(y)==set(DFsta.Channels[0].values()[0]) for x in DFsta.Channels for y in x.values()]): #make sure are channels are the same for all events
                detex.log(__name__,'all stations in subspace do not have the same channels',level='warning')
                raise Exception ('all stations in subspace do not have the same channels')
            Nc=len(channels) #num of channels
            for a in DFsta.iterrows():
                events=a[1].Events
                U=np.array([a[1].SVD[x] for x in a[1].UsedSVDKeys]) #used SVD basis
                ssArrayTD[a[1].Name]=U 
                UtU=np.dot(np.transpose(U),U)
                rele[a[1].Name]=int(continuousDataLength*samplingRate*Nc+np.max(np.shape(ssArrayTD[a[1].Name])))
                ssArrayFD[a[1].Name]=np.array([scipy.fftpack.fft(x[::-1],n=2**rele[a[1].Name].bit_length()) for x in ssArrayTD[a[1].Name]])
                offsets[a[1].Name]=a[1].Offsets
                mag=np.array([a[1].Stats[x]['magnitude'] for x in events])
                if 'Starttime' in a[1].SampleTrims.keys(): #if picks were made
                    WFs=np.array([a[1].AlignedTD[x][a[1].SampleTrims['Starttime']:a[1].SampleTrims['Endtime']]for x in events])
                else:
                    WFs=np.array([a[1].AlignedTD[x] for x in events])
#                delvals=np.where(mag<-100)[0] # find magnitudes that are not defined, IE -999
#                if len(delvals)>0:
#                    mag=np.delete(mag,delvals)
#                    WFs=np.delete(WFs,delvals,axis=0)
#                    events=np.delete(events,delvals)
                #pwave[a[1].Name],swave[a[1].Name]=self._getPandSWaves(PKS,a,events)
                mags[a[1].Name]=mag
                eves[a[1].Name]=events
                eventWaveforms[a[1].Name]=WFs
                WFU[a[1].Name]=np.dot(WFs,UtU) # events projected into subspace
                UtUdict[a[1].Name]=UtU
                
        elif not self.issubspace: #if singles not subspaces
            ssArrayTD,rele,ssArrayFD,offsets,mags,eventWaveforms,eves,WFU,UtUdict,pwave,swave={},{},{},{},{},{},{},{},{},{},{} # intitialize empty dicts
            Nc=len(channels)
            for a in DFsta.iterrows():
                if len(a[1].SampleTrims.keys())<1: #skip hours where no trim times are defined
                    continue
                events=a[1].Events
                Upre=[np.array(a[1].MPtd.values()[0][a[1].SampleTrims['Starttime']:a[1].SampleTrims['Endtime']])]
                U=np.array([x/np.linalg.norm(x) for x in Upre])
                ssArrayTD[a[1].Name]=U 
                UtU=np.dot(np.transpose(U),U)
                rele[a[1].Name]=int(continuousDataLength*samplingRate*Nc+np.max(np.shape(ssArrayTD[a[1].Name])))
                ssArrayFD[a[1].Name]=np.array([scipy.fftpack.fft(x[::-1],n=2**rele[a[1].Name].bit_length()) for x in ssArrayTD[a[1].Name]])
                offsets[a[1].Name]=a[1].Offsets
                mag=np.array([a[1].Stats[x]['magnitude'] for x in events])
                WFs=Upre
#                delvals=np.where(mag<-100)[0] # find magnitudes that are not defined, IE -999
#                if len(delvals)>0:
#                    mag=np.delete(mag,delvals)
#                    WFs=np.delete(WFs,delvals,axis=0)
#                    events=np.delete(events,delvals)
                #pwave[a[1].Name],swave[a[1].Name]=self._getPandSWaves(PKS,a,events)
                mags[a[1].Name]=mag
                eves[a[1].Name]=events
                eventWaveforms[a[1].Name]=WFs
                WFU[a[1].Name]=np.dot(WFs,UtU) # events projected into subspace
                UtUdict[a[1].Name]=UtU         
        
        if returnFull:
            return ssArrayTD,ssArrayFD,rele,offsets,mags,eventWaveforms,eves,WFU,UtUdict
        else:
            return ssArrayTD,ssArrayFD,rele
            
    def _getPandSWaves(self,PKS,a,events):
        """
        Function to get P and S wave segments
        """ 
        Pwaves=[]
        Swaves=[]
        if isinstance(PKS,pd.DataFrame):
            for eve in events:
                stat=a[1].Stats[eve]
                pkeve=PKS[PKS.Name==eve]
                if len(pkeve)<1 or pkeve.iloc[0].S==0: #if current event not found in picks files
                    Pwaves.append(-999)
                    Swaves.append(-999)
                else:
                    pk=pkeve.iloc[0]
                    Poffset=pk.P-stat['starttime']
                    Pend=pk.S-pk.P
                    Psamps=[int(a[1].SampleTrims['Starttime']+round(Poffset*stat['sampling_rate'])*stat['Nc']),int(round(Pend*stat['sampling_rate'])*stat['Nc'])]
                    Pwaves.append(a[1].AlignedTD[eve][Psamps[0]:Psamps[0]+Psamps[1]])
#                    plt.figure()
#                    plt.plot(a[1].AlignedTD[eve])
#                    plt.axvline(Psamps[0],c='r')
#                    plt.axvline(Psamps[0]+Psamps[1],c='r')
#                    plt.xlim(a[1].SampleTrims['Starttime']-400,a[1].SampleTrims['Endtime'])
#                    plt.show()
            #deb([a,Pwaves])
            return Pwaves,Swaves
        else:
            return None,None
        #deb([PKS,a])                             
    def _CreateCoeffArray(self,CorSeries,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD,WFU,UtU,staPKsDf):  
        #WFlen=np.shape(WFU[name])[1] #get length of window in samples
        dpv=0
        if self.trigCon==0:
            Ceval=CorSeries.SSdetect.copy()
        elif self.trigCon==1:
            Ceval=CorSeries.STALTA.copy()
        Sar=pd.DataFrame(columns=['DS','DS_STALTA','STMP','Name','Sta','MSTAMPmin','MSTAMPmax','Mag','SNR','ProEnMag'])
        count=0
        while Ceval.max()>=threshold[name]: 
            trigIndex=Ceval.argmax()
            coef=CorSeries.SSdetect[trigIndex]
            #times=times+[float(trigIndex)/sr+starttime]
            if self.extrapolateTimes:  #extrapolate times
                times=self._subSamp(Ceval,trigIndex,CorSeries.SampRate,CorSeries.TimeStamp)
                times1=float(trigIndex)/CorSeries.SampRate+CorSeries.TimeStamp
                if abs(times1-(times))>1.0/CorSeries.SampRate:
                    detex.log(__name__,'subsample extrapolation shifts time more than one sample',level='warning')
                    raise Exception('subsample extrapolation shifts time more than one sample') #make sure subsample extrapolation doesnt shift more than 1  sample
            else:
                times=[float(trigIndex)/CorSeries.SampRate+CorSeries.TimeStamp]
                
            if self.fillZeros: #if zeros are being filled dont even try STA/LTA
                SLValue=0.0
            else:
                SLValue=CorSeries.STALTA[trigIndex]
            #alpha[len(alpha)]=self._calcAlpha(trigIndex,Y,X,chans)
            Ceval=self._downPlayArrayAroundMax(Ceval,CorSeries.SampRate,dpv)
            
            
            if self.estimateMags: #estimate magnitudes
                ProEnmag,stdMag,SNR=self._estimateMagnitude(trigIndex,CorSeries,MPcon,mags[name],events[name],WFU[name],UtU[name],eventWaveForms[name],coef)
            else:
                ProEnmag,stdMag,SNR=np.NaN,np.NaN,np.NaN
            
            if count>4000: #kill switch to prevent infinite loop (just in case)
                detex.log(__name__,' _CreatCoeffArray loop exceeds limit of 4000 events in one continuous data chunk',level='warning')
                raise Exception (' _CreatCoeffArray loop exceeds limit of 4000 events in one continuous data chunk')
            MSTAMPmax,MSTAMPmin=times-np.min(offsets[name]),times-np.max(offsets[name])
            Sar.loc[count]=[coef,SLValue,times,name,sta,MSTAMPmin,MSTAMPmax,stdMag,SNR,ProEnmag]
            count+=1
        return Sar
    
    def _estimateArrivals(self,trigIndex,CorSeries,events,MPcon,pwave,swave,WFU):
        WFlen=np.shape(WFU)[1]
        ConDat=MPcon[trigIndex*CorSeries.Nc-WFlen:trigIndex*CorSeries.Nc+WFlen]
        Pccs=np.zeros((len(pwave),2))
        CCses=[]
        for evenum,eve in enumerate(pwave):
            if isinstance(eve,np.ndarray):
                CCs=fast_normcorr(eve,ConDat)
                CCses.append(CCs)
                ind=CCs.argmax()
                stmp=self._getArrivalStartTimes(ind,CorSeries,trigIndex,WFlen)
                Pccs[evenum,0]=CCs.max()
                Pccs[evenum,1]=stmp
        deb(CCses)
        
    def _getArrivalStartTimes(self,ind,CorSeries,trigIndex,WFlen):
        return (trigIndex-WFlen+ind)*(CorSeries.SampRate/CorSeries.Nc)+CorSeries.TimeStamp
        
        """ 
        basic function to get the corresponding time stamp for the max arrival times
        """
        
    def _estimateMagnitude(self,trigIndex,CorSeries,MPcon,mags,events,WFU,UtU,eventWaveForms,coef):
        """
        Estimate magnitudes by applying iterative waveform scaling to the detection for each event used in the subspace
        creation
        Currently uses the slower "fast_normcorr" method 
        """
        #TODO use the frequency domain templates to speed up
        #TODO Clean this up
        #deb([trigIndex,CorSeries,MPcon,mags,events,WFU,UtU])

        WFlen=np.shape(WFU)[1] # event waveform length
        ConDat=MPcon[trigIndex*CorSeries.Nc:trigIndex*CorSeries.Nc+WFlen] #continuous data chunk that triggered  subspace
        if self.issubspace:
            ssCon=np.dot(UtU,ConDat) # continuous data chunk projected into subspace
            proEn=np.var(ssCon)/np.var(WFU,axis=1)
    
        
        #calculate noise level before event (std) in order to estimate SNR
        if trigIndex*CorSeries.Nc>5*WFlen:
            rollingstd=pd.rolling_std(MPcon[trigIndex*CorSeries.Nc-5*WFlen:trigIndex*CorSeries.Nc],WFlen)[WFlen-1:]
        else:
            rollingstd=pd.rolling_std(MPcon[trigIndex*CorSeries.Nc:trigIndex*CorSeries.Nc+WFlen+6*WFlen],WFlen)[WFlen-1:]
        baseNoise=np.median(rollingstd) #take median of std for noise level
        SNR=np.std(ConDat)/baseNoise #estiamte SNR
        
        # Calc magnitudes, use projected energy estimates (similar to waveform scaling) and 
        touse=mags>-15 #only use magnitude if it is greater than -15 (-999 is often default if magnitude was not calculated)
        if self.issubspace:
            if not any(touse): #if no defined magnitudes avaliable
                projectedEnergyMags,stdMags=np.NaN,np.Nan
            else:
                eventCors=np.array([fast_normcorr(x,ConDat)[0] for num,x in enumerate(eventWaveForms)]) #correlation coefs between each event and continuous data
                projectedEnergyMags=np.sum([(mags[x]+np.log10(np.sqrt(proEn[x])))*np.square(eventCors[x]) for x in range(len(proEn)) if mags[x]>-10])/np.sum(np.square(eventCors[touse]))    
                stdMags=np.sum([(mags[x]+np.log10(np.std(ConDat)/np.std(eventWaveForms[x])))*np.square(eventCors[x]) for x in range(len(proEn)) if mags[x]>-10])/np.sum(np.square(eventCors[touse]))    
        else:
            assert len(mags)==1 #make sure if single is being used only one magnitude is recorded
            if np.isnan(mags[0]) or mags[0]<-15:
                projectedEnergyMags=np.NaN
                stdMags=np.NaN
            else:
                projectedEnergyMags=mags[0]+np.log10(np.dot(ConDat,WFU[0])/np.dot(WFU[0],WFU[0])) #use simple waveform scaling if single
                stdMags=mags[0]+np.log10(np.std(ConDat)/np.std(WFU[0]))
    

        return projectedEnergyMags,stdMags,SNR
        
    def _subSamp(self,Ceval,trigIndex,sr,starttime) :
        """ Method to estimate subsample time delays using cosine-fit interpolation
        Cespedes, I., Huang, Y., Ophir, J. & Spratt, S. 
        Methods for estimation of sub-sample time delays of digitized echo signals. 
        Ultrason. Imaging 17, 142–171 (1995)"""
        ind=Ceval.argmax()
        if trigIndex != ind:
            detex.log(__name__,'something is messed up, trigIndex and CC.argmax no equal',level='warning')
            raise Exception('something is messed up, trigIndex and CC.argmax no equal')
        if ind==0 or ind==len(Ceval)-1: # If max occurs at beg or end of CC set as beg or end, no extrapolation
            tau=float(ind)/sr + starttime
        else:
            alpha=np.arccos((Ceval[ind-1]+Ceval[ind+1])/(2*Ceval[ind]))
            tau=-(np.arctan((Ceval[ind-1]-Ceval[ind+1])/(2*Ceval[ind]*np.sin(alpha)))/alpha)*1.0/sr+ind*1.0/sr+starttime
            if -np.arctan((Ceval[ind-1]-Ceval[ind+1])/(2*Ceval[ind]*np.sin(alpha)))/alpha >1:
                detex.log(__name__,'Something wrong with extrapolation, more than 1 sample shift predicted ',level='warning')
                raise Exception('Something wrong with extrapolation, more than 1 sample shift predicted ')
        return tau
        
    def _replaceNanWithMean(self,arg): # Replace where Nans occur with closet non-Nan value
        ind = np.where(~np.isnan(arg))[0]
        first, last = ind[0], ind[-1]
        arg[:first] = arg[first+1]
        arg[last + 1:] = arg[last]
        return arg
    
    def _getStaLtaArray(self,C,LTA,STA): # Get STA/LTA 
        if STA==0:
            STA=1
            STArray=np.abs(C)
        else:
            STArray=pd.rolling_mean(np.abs(C),STA,center=True)
            STArray=self._replaceNanWithMean(STArray)
        LTArray=pd.rolling_mean(np.abs(C),LTA,center=True)
        LTArray=self._replaceNanWithMean(LTArray)
        out=np.divide(STArray,LTArray)
        return out
        
    def _evalTriggerCondition(self,Corrow,name,threshold,returnValue=False): 
        """ Evaluate if Trigger condition is met and return true or false or correlation value if returnValue=True
        """
        Out=False
        if self.trigCon==0:
            trig= Corrow.MaxDS
            if trig>threshold[name]:
                Out=True
        elif self.trigCon==1:
            trig=Corrow.maxSTALTA
            if trig>threshold[name]:
                Out=True
            #trig=Cors[maxIn]
        if returnValue==True:
            return trig
        if returnValue==False:
            return Out
        
    def _downPlayArrayAroundMax(self,C,sr,dpv,buff=20): #function to zero out correlation arrays around where max occurs
        index=C.argmax()
        if index<buff*sr+1:
            C[0:int(index+buff*sr)]=dpv
        elif index>len(C)-buff*sr:
            C[int(index-sr*buff):]=dpv
        else:
            C[int(index-sr*buff):int(sr*buff+index)]=dpv
        return C
    
    def UTCfitin(self,yearrange,jdayrange,UTCstart,UTCend):
        """Function called by DeTex to make sure years and times fit into times and dates"""
        if not UTCstart:
            UTCstart=-np.inf
        if not UTCend:
            UTCend=np.inf
        jdrangeOut=[[]]*len(yearrange)
        for a in range(len(yearrange)):
            for b in range(len(jdayrange[a])):
                UTCrep=obspy.core.UTCDateTime(year=int(yearrange[a]),julday=int(jdayrange[a][b]))
                if UTCrep.timestamp <= UTCend.timestamp and UTCrep.timestamp >= UTCstart:
                    jdrangeOut[a].append(jdayrange[a][b])
        return jdrangeOut
    
            
    def _standardizeCorrelation(self,R): #Function to take CC coefficients and rolling std matricies and calculate combined coeficients
        X=np.zeros(len(R[0][0]))
        sigt=0
        sigs=np.zeros(len(R[0][1]))
        for a in range(len(R)):
            X=X+np.multiply(R[a][0],R[a][1])*R[a][2]*R[a][3]
            sigt=sigt+np.square(R[a][2]) #Equiv template sigma
            sigs=sigs+np.square(R[a][1]) #Equiv Search space sigma
        sigs=np.sqrt(sigs/len(R))
        sigt=np.sqrt(sigt/len(R))
        CC=np.divide(X,sigs*sigt*len(R)*R[0][3])
        
        return CC,X
        
    def _getTemplate(self,EveDir,eve,sta): #Function to load templates, also returns channels found in template
        eventSubDir=glob.glob(os.path.join(EveDir,eve,sta,'*'))[0] # Length will only be 1
        chans=[os.path.basename(x).split('.')[2].split('_')[0] for x in glob.glob(os.path.join(eventSubDir,'*'))]
        tem=[0]*len(chans)
        for a in range(len(chans)):
            tem[a]=obspy.core.read(os.path.join(eventSubDir,'*'+chans[a]+'*.sac'))
        global debug
        return chans, tem
    
    def _getcontinuousRanges(self,ConDir,sta,UTCstart,UTCend):
        # get lists of all hours and days for which continuous data is avaliable
        conrangeyear=[os.path.basename(x) for x in glob.glob(os.path.join(ConDir,sta,'*'))]
        conrangejulday=[0]*len(conrangeyear)
        for a in range(len(conrangeyear)):
            conrangejulday[a]=[os.path.basename(x) for x in glob.glob(os.path.join(ConDir,sta,conrangeyear[a],'*'))]
        if UTCstart or UTCend: #Trim conrangeyear and conrangejulday to fit in UTCstart and UTCend if defined
            conrangejulday=self.UTCfitin(conrangeyear,conrangejulday,UTCstart,UTCend)
        return conrangejulday,conrangeyear
            
    def _getAdjustedTimes(self,templatePath):
        """Using template time stampes the minimum starttime. The differnce
        between the min and each other startime is then subtracted from time stamps in RES so 
        non-coincidence events should occur at nearly the same time"""
        Templates=templatePath
        timeStarts={} #initialized dictionary where time offsets will by stored with the key of station name
        for waveform in glob.glob(os.path.join(Templates,'*')):
            trimSer=self.pksDF[self.pksDF.Path==waveform].iloc[0]
            trim=[trimSer[0],trimSer[1]]
            TR=obspy.core.read(waveform)
            self._checkTraceStartTimes(TR)
            if len(trim)==0: #Look for trim file (tms) if not found then simply use trace
                timeStarts[TR[0].stats.network+'.'+TR[0].stats.station]=TR[0].stats.starttime.timestamp
            else:
                timeStarts[TR[0].stats.network+'.'+TR[0].stats.station]=trim[0]
        minStart=min(timeStarts.values())
        offSetTimes={}
        for a in timeStarts:
            offSetTimes[a]=timeStarts[a]-minStart
        return offSetTimes

    
    def _checkTraceStartTimes(self,TR): #Make sure all channels of each station have equal starttimes
        T0=abs(TR[0].stats.starttime.timestamp)
        sr=TR[0].stats.sampling_rate
        for a in range(len(TR)):
            Tn=abs(TR[a].stats.starttime.timestamp)
            if Tn>2*sr+T0 or Tn<-sr*2+T0:
                detex.log(__name__,'Time stamps not equal for all channels of ' +TR[0].stats.station,level='warning')
                raise Exception('Time stamps not equal for all channels of ' +TR[0].stats.station)

    def _getFilesToCorr(self,sta,jday,year,chans,ConDir):
                                #Get unique hours and corresponding files for all components
        FilesToCorr=[] 
        for a2 in range(len(chans)):
            FilesToCorr=FilesToCorr+glob.glob(os.path.join(ConDir,sta,str(year),str(jday),chans[a2],'*'))
        sharedHours=list(set([os.path.basename(x).split('-')[1].split('.')[0].split('T')[1] for x in FilesToCorr]))
        FilesToCorr=[0]*len(chans)
        for a2 in range(len(chans)):
            FilesToCorr[a2]=[0]*len(sharedHours)
            for a3 in range(len(sharedHours)):
                #FilesToCorr is now a len(chans) by len(sharedHours) matrix populated with paths to continuous waveforms
                FilesToCorr[a2][a3]=glob.glob(os.path.join(ConDir,sta,year,jday,chans[a2],'*'+'T'+sharedHours[a3]+'.sac'))
        return FilesToCorr, sharedHours
    
    def _getRA(self,ssArrayTD,ssArrayFD,st,Nc,reqlen,contrim,Names):     
        CorDF=pd.DataFrame(index=Names,columns=['SSdetect','STALTA','TimeStamp','SampRate','MaxDS','MaxSTALTA','Nc','File'])
        #CorDF['File']=os.path.basename(fileToCorr)
        try:
            conStream=detex.construct._applyFilter(st,self.filt,self.decimate,self.dtype,fillZeros=self.fillZeros)
        except:
            detex.deb([st,self.filt,self.decimate,self.dtype,self.refillZeros])
            detex.log(__name__,'failed to filter %s, skipping' % st,level='warning')
            return None,None,None
        
        CorDF.SampRate=conStream[0].stats.sampling_rate
        MPcon,ConDat,TR=multiplex(conStream,Nc,returnlist=True,retst=True)
        CorDF.TimeStamp=TR[0].stats.starttime.timestamp
        if isinstance(contrim,dict):
            ctrim=np.median(contrim.values())
        else:
            ctrim=contrim
        if ctrim<0: #Trim continuous data to avoid overlap
            MPconcur=MPcon[:int(ctrim*CorDF.SampRate[0]*Nc)]
        else:
            MPconcur=MPcon
        MPconFD=scipy.fftpack.fft(MPcon,n=2**int(np.median(reqlen.values())).bit_length())
 
        
        for name,row in CorDF.iterrows(): #loop through each hour in this Jday and calculate statistic
            if len(MPcon)<=np.max(np.shape(ssArrayTD[name])): # make sure the template is shorter than continuous data else skip
                MPcon=None  
            if isinstance(MPcon,np.ndarray): #If channels not equal in length and multiplexing fails delete hour, else continue with CCs                  
                ssd=self._MPXDS(MPconcur,reqlen[name],ssArrayTD[name],ssArrayFD[name],Nc,MPconFD)
#                try:                
#                    maxind=ssd.argmax()
#                except ValueError:
#                    return None,None,None
                CorDF.SSdetect[name]=ssd
                if len(ssd)<10:
                    return None,None,None
                CorDF.MaxDS[name]=ssd.max()
                if CorDF.MaxDS[name]>1.1: # If an infinity value occurs, zero it. 
                    ssd[np.isinf(ssd)] = 0
                    CorDF.SSdetect[name]=ssd
                    CorDF.MaxDS[name]=ssd.max()
                if not self.fillZeros: #dont calculate sta/lta if zerofill is used
                    try:
                        CorDF.STALTA[name]=self._getStaLtaArray(CorDF.SSdetect[name],self.triggerLTATime*CorDF.SampRate[0],self.triggerSTATime*CorDF.SampRate[0])
                    except:
                        return None,None,None
                    CorDF.MaxSTALTA[name]=CorDF.STALTA[name].max()
                CorDF.Nc[name]=Nc
                
            else:
                return None,None,None
        return CorDF,MPcon,ConDat
     
    def FFTemplate(self,MPtem,reqlen):# apply the fft to the template, pad appropriately
        n = len(MPtem)
        nt = (MPtem-np.mean(MPtem))/(np.std(MPtem)*n)
        sum_nt=nt.sum()
        MPtemFD=scipy.fftpack.fft(nt,n=2**reqlen.bit_length())
        return MPtemFD,sum_nt   

    def _MPXDS(self,MPcon,reqlen,ssArrayTD,ssArrayFD,Nc,MPconFD): # multiplex subspace detection statistic function
#        MPconFD=scipy.fftpack.fft(MPcon,n=2**reqlen.bit_length())
        n = np.int32(np.shape(ssArrayTD)[1]) #length of each basis vector
        a = pd.rolling_mean(MPcon, n)[n-1:] #rolling mean of continuous data
        b = pd.rolling_var(MPcon, n)[n-1:]  # rolling var of continuous data
        b *= n #rolling power in vector
        sum_ss=np.sum(ssArrayTD,axis=1) #the sume of all the subspace basis vectors
        av_norm=np.multiply(a.reshape(1,len(a)),sum_ss.reshape(len(sum_ss),1)) #term to account for non-averaged vectors
        m1=np.multiply(ssArrayFD,MPconFD)    
        if1=scipy.real(scipy.fftpack.ifft(m1))[:,n-1:len(MPcon)]-av_norm
        result1=np.sum(np.square(if1),axis=0)/b
        return result1[::Nc]    

def deb(varlist):
    global de
    de=varlist
    sys.exit(1)     

           