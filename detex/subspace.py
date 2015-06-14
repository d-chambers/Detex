# -*- coding: utf-8 -*-
"""
Created on Tue Jul 08 21:24:18 2014

@author: Derrick
Module to perform subspace detections, as well as return relative offsets of templates for other modules
"""
import pandas as pd, numpy as np, obspy, os, glob, matplotlib.pyplot as plt, json
import matplotlib as mpl
import detex, scipy
import cPickle, itertools
import collections, copy
import colorsys

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import multiprocessing,sys, warnings, numbers
pd.options.mode.chained_assignment = None #mute setting copy warning
from struct import pack

#warnings.filterwarnings('error') #uncomment this to make all warnings be thrown as errors in order to find where warnings are coming from
 
#np.seterr(all='raise')

######################################## CLUSTERING FUNCTIONS AND CLASSES  ###############################


def createCluster(CCreq=0.5,indir='EventWaveForms',templateDir=True,filt=[1,10,2,True],StationKey='StationKey.csv',
                  TemplateKey='TemplateKey.csv',trim=[100,200],filelist=None,allram=True,masterStation=None,saveclust=True,
                  clustname='clust.pkl',decimate=None, dtype='double',consistentLength=True,eventsOnAllStations=False):
                      
                      
    """ Function to create the cluster class which contains the linkage matrix, event names, and a few visualization methods
    
    Parameters
    -------

    CCreq : float, between 0 and 1
        The required cc between vectors in order for a cluster group to form. 0 means all vectors used
    indir : str
        Path to the directory containing the event waveforms
    templateDir : boolean
        If true indicates indir is formated in the way detex.getdata orginizes the directories
    filt : list
        A list of the required input parameters for the obspy bandpass filter [freqmin,freqmax,corners,zerophase]
    StationKey : str
        Path to the station key used by the events (#TODO get ride of this parameter and use station key path already in template key )
    TemplateKey : boolean
        Path to the template key 
    trim : list 
        A list with seconds to trim from start of each stream in [0] and the total duration in seconds to keep from trim point in [1], the second
        parameter greatly influences the runtime and if the first parameter is incorrectly selected the waveform may be missed entirely
    filelist : list of str or None
        A list of paths to obspy readable seismic records. If none use indir
    allram : boolean 
        If true then all the taces are read into the ram before correlations, saves time but could potentially fill up ram (Only True is currently supported)
    masterStation : str
        Allows user to set which station in StationKey should be used for clustering, if the string of a single
        station name is passed cluster analysis is only performed on that station. The event groups will then be forced for all stations
        If none is passed then all stations are clustered independantly (IE no master station to force event groups in subspace class)
    saveClust : boolean
        If true save the cluster object in the current working directory as clustname
    clustname : str
        path (or name) to save the clustering instance, only used if saveClust
    decimate : int or None
        A decimation factor to apply to all data (parameter is simply passed to the obspy trace/stream method decimate). Can greatly increase 
        speed and is desirable if the data are oversampled
    dytpe : str
        The data type to use for recasting both event waveforms and continous data arrays. If none the defualt of float 64 is kept. Options include:
            double- numpy float 64
            single- numpy float 32, much faster and amenable with cuda GPU processing, sacrifices precision
    consistentLength : boolean
        If true the data in the events files are more or less the same length. Switch to false if the data are not, but can greatly increase 
        run times. 
    eventsOnAllStations : boolean
        If true only use the events that occur on all stations, if false dont let each station have an independant event list
    Returns
    ---------
    An instance of the detex SSSlustering class
    """
    # Read in stationkey and set master station, if no master station selected us first station
    # 
    stakey=pd.read_csv(StationKey)
    stakey=stakey[[isinstance(x,str) or abs(x) >= 0 for x in stakey.NETWORK]]
    stakey['STATION']=[str(x) for x in stakey.STATION] #make sure station and network are strs
    stakey['NETWORK']=[str(x) for x in stakey.NETWORK]
    if masterStation == None:
        pass
        #stakey=stakey.iloc[0:1]
    elif isinstance(masterStation,str):
        masterStation=[masterStation]
    if isinstance(masterStation,list):
        if len(masterStation[0].split('.'))>1: #if NETWORK.STATION format is used convert to just station
            #stakey=[x.split('.')[1] for x in stakey]
            masterStation=[x.split('.')[1] for x in masterStation]
        stakey=stakey[stakey.STATION.isin(masterStation)]
    if len(stakey)==0:
        raise Exception('Master station is not in the station key, aborting clustering')
    temkey=pd.read_csv(TemplateKey)
    # Intialize parts of DF that will be used to store cluster info

    stakey.reset_index(drop=True,inplace=True)
    #stakey['Delay']=list
    if not templateDir:
        filelist=glob.glob(os.path.join(indir,'*'))
    TRDF=_loadEvents(filelist,indir,filt,trim,stakey,templateDir,decimate,temkey,dtype)  
    TRDF.sort(columns='Station',inplace=True)
    TRDF.reset_index(drop=True,inplace=True)
    if consistentLength:
        TRDF=_testStreamLengths(TRDF)
    #deb([TRDF,ddf])
    TRDF['Link']=None

    # Loop through stationkey performing cluster analsis only on stationkey    
    if eventsOnAllStations: #If only useing events common to all stations
        eventList=list(set.intersection(*[set(x) for x in TRDF.Events])) #get list of events that occur on all required stations
        eventList.sort()        
        if len(eventList)<2:
            raise Exception('less than 2 events in population have required stations')
    for a in TRDF.iterrows(): #loop over master station(s)
        print 'getting CCs and lags on ' + a[1].Station   
        if not eventsOnAllStations:
            eventList=a[1].Events
        DFcc,DFlag=_makeDFcclags(eventList,a,consistentLength=consistentLength)
        TRDF.Lags[a[0]]=DFlag
        TRDF.CCs[a[0]]=DFcc
        #deb([DFlag,DFcc])
        cx=np.array([])
        lags=np.array([])
        cxdf=1.0000001-DFcc
        cxx=[x[xnum:] for xnum,x in enumerate(cxdf.values)] #flatten cxdf and index out nans 
        cx=np.fromiter(itertools.chain.from_iterable(cxx), dtype=np.float64)     
        try:
            cx,cxdf=_ensureUnique(cx,cxdf) # ensure x is unique in order to link correlation coeficients to lag time in dictionary
        except:
            deb([cx,cxdf])
        for b in DFlag.iterrows(): #this is not efficient, consider rewriting withoutloops
            lags=np.append(lags,b[1].dropna().tolist())
        try:    
            link = linkage(cx) #get cluster linkage
        except:
            deb([a,cx,cxdf,lags])
        TRDF.loc[a[0],'Link']=link
    #DFcc=pd.DataFrame(cxw,index=range(len(cxw)),columns=range(1,len(cxw)+1))
    trdf=TRDF[['Station','Link','CCs','Lags','Events']] # a truncated TRDF only passign what is needed for clustering
    clust=SSClusterStream(trdf,temkey,eventList,CCreq,filt,decimate,trim,indir,saveclust,clustname,templateDir,filelist,StationKey,saveclust,clustname,eventsOnAllStations)
    
    if saveclust:
        clust.write()
    return clust
    
def loadClusters(filename='clust.pkl'): #load a cluster with filename as input arg, default to default name
    """
    Simple function to utilize cPickle to load a saved instance of SSclustering
    
    filename is the name of the saved SSclustering instance
    """
    from detex.subspace import SSClusterStream
    with open(filename,'rb') as fp:
        outob=cPickle.load(fp)
    return outob
    
def loadSubSpace(filename='subspace.pkl'): 
    """
    Simple functon ot use cPickle to load previously saved SubSpaceStream instances
    
    filename is the name of the pickeled instance
    """
    from detex.subspace import SubSpaceStream
    outob=cPickle.load(open(filename,'rb'))
    return outob


class SSClusterStream(object):
    """
    A container for multiple cluster objects
    """
    def __init__(self,TRDF,temkey,eventList,CCreq,filt,decimate,trim,indir,saveclust,clustname,templateDir,filelist,StationKey,save,filename,eventsOnAllStations): #save,filename
        self.__dict__.update(locals()) #Instantiate all input variables
        self.TRDF=None # clear this variable as it takes up too much space and is not used later
        self.StationKey=StationKey
        self.temkey=temkey
        self.clusters=[0]*len(TRDF)
        self.stalist=TRDF.Station.values.tolist() #get station lists for indexing
        self.stalist2=[x.split('.')[1] for x in self.stalist]
        self.filename=filename
        for num,row in TRDF.iterrows():
            if not eventsOnAllStations:
                evlist=row.Events
            else:
                evlist=eventList
            self.clusters[num]=SSCluster(self,row.Station,temkey,evlist,row.Link,CCreq,filt,decimate,trim,row.CCs,indir,templateDir,filelist,StationKey)
        if save:
            self.write()
    
    def updateReqCC(self,reqCC):
        """
        Updates the required correlation coefficient for clusters to form on all stations if reqCC is a float
        or on each station if reqCC is a dict whose keys reference individual station cluster objects
        
        Parameters:
        reqCC : float (between 0 and 1), or dict of reference keys and floats, or list of floats
        """
        if isinstance(reqCC,float):
            if reqCC<0 or reqCC>1:
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
        print 'writing cluster object as %s' % self.filename
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
                raise Exception('%s is not a station in this cluster object' % key)
        else:
            raise Exception ('%s must either be a int or str of station name')
            
    def __len__(self): 
        return len(self.clusters) 
        

class SSCluster(object):
    def __init__(self,clustStream,station,temkey,eventList,link,CCreq,filt,decimate,trim,DFcc,indir,templateDir,filelist,StationKey):
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
        print('CCreq for station %s updated to CCreq=%1.3f'%(self.station,newCCreq))

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
            print 'WARNING : No events clustered bassed on the requirement that cor coef = %1.3f' % self.CCreq
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
        

######################### SUBSPACE FUNCTIONS AND CLASSES #####################


def createSubSpace(Pf=10**-10,clust=None,clustFile='clust.pkl',minEvents=3,dtype='double',condir='ContinousWaveForms' ):
    """
    Function to create subspaces on all avaliable stations bassed on the clusters in cSSClustering object which will either
    be passed directly as the keyword clust or will be loaded from the path in clustFile
    
    Note
    ----------
    Most of the parameters that define where seismic data are located, which events to use, and which stations to use
    are already defined in the cluster (SSClustering) object. Therefore, it is extremely important to not move any files
    around and to call the detex.subspace.createSubSpace function in the same directory where the clustering object was 
    created.  
    
    
    Parameters
    -----------
    
    Pf : float
        The probability of false detection as modeled by the statistical framework in Harris 2006 Theory (eq 20)
        Or by fitting a PDF to an empirical estimation of the null space (similar to Wiechecki-Vergara 2001)
        Thresholds are not set until calling the SVD function of the subspace stream object
    clust: detex.subspace.SSClustering object
        A clusting object to pass directly rather than reading the file in clustFile
    clustFile : str
        Path to the saved SSClustering object if None is passed via the clust parameter 
    minEvents : int
        The Min number of events that must be in a cluster in order for a subspace to be created from that cluster
    dtype : str ('single' or 'double')
        The data type of the numpy arrays used in the calculation options are:
        single- a np.float32 single precision repressentation. Slightly faster (~30%) than double but at a cost of sig figs
        double- a np.float64 (default)
        
    Returns
    -----------
    
    A detex.subspace.SubSpaceStream object
    
    """
    if clust==None: #if no cluster object passed read a pickled one
        cl=cPickle.load(open(clustFile,'rb'))
    else:
        cl=clust
    print ('Starting Subspace Construction')
    #cl.printAtr()
    #CCreq=cl.CCreq
    temkey=cl.temkey
    # Read in stationkey
    stakey=pd.read_csv(cl.StationKey)
    # Intialize parts of DF that will be used to store cluster info
    #allChans=np.array([y.split('-') for y in list(set([x for x in stakey.CHANNELS.values]))]).flatten().tolist()
    
    TRDF  =_loadEvents(cl.filelist,cl.indir,cl.filt,cl.trim,stakey,cl.templateDir,cl.decimate,temkey,dtype)# TRDF is DF main container for processed data    
    for num,row in TRDF.iterrows(): #Fill in cluster info from cluster object
        if row.Station in cl.stalist: # if station repressented in cluster object
            TRDF.loc[num,'Link'],TRDF.loc[num,'Clust']=cl[row.Station].link,cl[row.Station].clusts
        else: # If not use first station, which should be the masterstation
            TRDF.loc[num,'Link'],TRDF.loc[num,'Clust']=cl[0].link,cl[0].clusts
    
   
   # Loop through stationkey performing cluster analsis only on stationkey, force all other stations to follow station key clustering       
    
    ## NOTE!!! If data for an event is not found in the master station it will not be used in the clustering of other station where it might be found   
    subSpaceDict={}
    for num,row in TRDF.iterrows(): #Loop through each cluster
        #eventListCluster=row.Events #all events in current cluster
        staSS=_makeSSDF(row)
        staSS=staSS[[len(x)>=minEvents for x in staSS.Events]] #only keep subspaces that meet min req, dont renumber
        staSS.reset_index(drop=True,inplace=True)
        for sa in staSS.iterrows(): 
            eventList= sa[1].Events#all events of current cluster also in station
            eventList.sort()
            DFcc,DFlag=_makeDFcclags(eventList,sa)
            staSS.Lags[sa[0]]=DFlag
            staSS.CCs[sa[0]]=DFcc
            cx=np.array([])
            lags=np.array([])
            cxdf=1.0000001-DFcc #subtract from one plus a small number to avoid roundoff error letting cc >1 
            for b in cxdf.iterrows():
                cx=np.append(cx,b[1].dropna().tolist()) #use 1 - coeficient because clustering uses distances
            cx,cxdf=_ensureUnique(cx,cxdf) # ensure x is unique in order to link correlation coeficients to lag time in dictionary
            for b in DFlag.iterrows(): #this is not efficient, consider rewriting withoutloops
                lags=np.append(lags,b[1].dropna().tolist())
            link = linkage(cx) #get cluster linkage
            staSS.Link[sa[0]]=link
            CCtoLag=_makeLagSeries(cx,lags) # a map from cross correlation coeficinets to lag times
            delays=_getDelays(link,CCtoLag,cx,lags,sa,cxdf)
            delayNP=-1*np.min(delays)
            delayDF=pd.DataFrame(delays+delayNP,columns=['SampleDelays'])
            delayDF['Events']=[eventList[x] for x in delayDF.index]
            staSS.AlignedTD[sa[0]]=_alignTD(delayDF,sa)
            staSS['Stats'][sa[0]]=_updateStartTimes(sa,delayDF,temkey)
            offsets=[staSS['Stats'][sa[0]][x]['offset'] for x in sa[1].Stats.keys()] #offset times
            staSS['Offsets'][sa[0]]=[np.min(offsets),np.median(offsets),np.max(offsets)]
        subSpaceDict[row.Station]=staSS.drop(['MPfd','MPtd','Link','Lags','CCs'],axis=1)
    singlesDict=_makeSingleEventDict(cl,TRDF) #make a list of sngles (events that dont cluseter) to pass to subspace object
    substream= SubSpaceStream(singlesDict,subSpaceDict,cl,dtype,Pf,condir)
    return substream

def _updateStartTimes(sa,delayDF,temkey): #update the starttimes to reflect the values trimed in alignement process
    statsdict=sa[1].Stats
    for key in statsdict.keys():
        temtemkey=temkey[temkey.NAME==key].iloc[0]
        delaysamps=delayDF[delayDF.Events==key].iloc[0].SampleDelays
        Nc=statsdict[key]['Nc']
        sr=statsdict[key]['sampling_rate']
        statsdict[key]['starttime']=statsdict[key]['starttime']+delaysamps/(sr*Nc)
        statsdict[key]['origintime']=obspy.core.UTCDateTime(temtemkey.TIME).timestamp
        statsdict[key]['magnitude']=temtemkey.MAG
        statsdict[key]['offset']=statsdict[key]['starttime']-statsdict[key]['origintime'] #predict offset time
    return statsdict

def _testLags(eventList,sa):# test lag times
    DFcc=pd.DataFrame(columns=np.arange(1,len(eventList)),index=np.arange(0,len(eventList)-1))
    DFlag=pd.DataFrame(columns=np.arange(1,len(eventList)),index=np.arange(0,len(eventList)-1))
    for b in DFcc.index.values:
        for c in range(b+1,len(DFcc)+1):
            mptd1=sa[1].AlignedTD[eventList[b]]
            mptd2=sa[1].AlignedTD[eventList[c]]
            rele=2*len(mptd1)
            reqlen=rele.bit_length()
            mpfd1=np.fft.fft(mptd1,n=2**reqlen)
            mpfd2=np.fft.fft(mptd2,n=2**reqlen)
            Nc1=sa[1].Channels[eventList[b]]
            Nc2=sa[1].Channels[eventList[b]]
            maxcc,sampleLag=_CCX2(mpfd1,mpfd2,mptd1,mptd2,Nc1,Nc2)
            DFcc[c][b]=maxcc
            DFlag[c][b]=sampleLag
    return DFcc,DFlag

def _makeDFcclags(eventList,sa,consistentLength=True):
    DFcc=pd.DataFrame(columns=np.arange(1,len(eventList)),index=np.arange(0,len(eventList)-1))
    DFlag=pd.DataFrame(columns=np.arange(1,len(eventList)),index=np.arange(0,len(eventList)-1))
    for b in DFcc.index.values:
        for c in range(b+1,len(DFcc)+1):
            mptd1=sa[1].MPtd[eventList[b]]
            mptd2=sa[1].MPtd[eventList[c]]
            if consistentLength:
                mpfd1=sa[1].MPfd[eventList[b]]
                mpfd2=sa[1].MPfd[eventList[c]]
                Nc1=sa[1].Channels[eventList[b]]
                Nc2=sa[1].Channels[eventList[b]]
                maxcc,sampleLag=_CCX2(mpfd1,mpfd2,mptd1,mptd2,Nc1,Nc2)
            else:
                maxlen=np.max([len(mptd1),len(mptd2)])
                if not len(mptd1)<maxlen:
                    mptd1,mptd2=mptd2,mptd1
                mptd2=np.pad(mptd2,(len(mptd1)/2,len(mptd1)/2),'constant',constant_values=(0,0))
                cc=fast_normcorr(mptd1, mptd2)
                
                maxcc=cc.max() #TODO fix this sloppy fix
                sampleLag=0
            DFcc[c][b]=maxcc
            DFlag[c][b]=sampleLag
    return DFcc,DFlag

def _CCX2(mpfd1,mpfd2,mptd1,mptd2,Nc1,Nc2): #Function find max correlation coeficient and corresponding lag times

    if len(Nc1)!=len(Nc2): #make sure there are the same number of channels
        raise Exception('Number of Channels not equal') #maybe make channels be the same?
    Nc=len(Nc1) #Number of channels
    if len(mptd1)!=len(mptd2) or len(mpfd2) !=len(mpfd1): #if TD or FD lengths not equal raise exception
        raise Exception('lengths not equal on multiplexed data streams')
    n=len(mptd1)
    #mptd2=np.lib.pad(mptd2, (n/2,n/2), 'constant', constant_values=(0,0))
    mptd2Temp=mptd2.copy()
    mptd2Temp=np.lib.pad(mptd2Temp, (n-1,n-1), 'constant', constant_values=(0,0))
    a = pd.rolling_mean(mptd2Temp, n)[n-1:]
    #deb([mptd2Temp,n])
    b = pd.rolling_std(mptd2Temp, n)[n-1:]
    b *= np.sqrt((n-1.0) / n)
    c=np.real(scipy.fftpack.ifft(np.multiply(np.conj(mpfd1),mpfd2))) #[:n+1]    
    c1=np.concatenate([c[-n+1:],c[:n]])
    result = ((c1 - mptd1.sum() * a) / (n*b*np.std(mptd1)))[Nc-1::Nc]
    #result = ((c1 - mptd1.sum() * a) / (n*b*np.std(mptd1)))
    try:
        maxcc=np.nanmax(result)
        maxind=np.nanargmax(result)
        if maxcc>1.1: #if a inf is found in array
            result[result== np.inf]=0
            maxcc=np.nanmax(result)
            maxind=np.nanargmax(result)
    except:
        return 0.0,0.0
        #deb([mpfd1,mpfd2,mptd1,mptd2,Nc1,Nc2])
    
    #return maxcc,maxind-n +1
    return maxcc,(maxind+1)*Nc-n
    

    
def _alignTD(delayDF,sa): #loop through delay Df and apply offsets to create alligned arrays dictionary
    aligned={}    
    TDlengths=len(sa[1].MPtd[delayDF.Events[0]])-max(delayDF.SampleDelays) #find the required length for each aligned stream 
    for b in delayDF.iterrows(): 
        orig=sa[1].MPtd[b[1].Events] #TODO this is not efficient or pythonic, revise when possible 
        orig=orig[b[1].SampleDelays:]
        orig=orig[:TDlengths]
        aligned[b[1].Events]=orig 
        if len(orig)==0:
            raise Exception('Alignment of multiplexed stream failing, try raising ccreq of imput cluster object')
    return aligned

def _makeSingleEventDict(cl,TRDF): 
    singlesdict={}
    for num,row in TRDF.iterrows():
        singleslist=[0]*len(cl[row.Station].singles)
        DF=pd.DataFrame(index=range(len(singleslist)),columns=[x for x in TRDF.columns if not x in ['Clust','Link','Lags','CCs']])
        #evelist=cl[row.Station].singles
        for a1 in range(len(singleslist)):
            #DF.Events[a[0]]=evlist
            evelist=[cl[row.Station].singles[a1]]
            DF.Station[a1]=row.Station
            DF.MPtd[a1]=_returnDictWithKeys(row,'MPtd',evelist)
            DF.MPfd[a1]=_returnDictWithKeys(row,'MPfd',evelist)
            DF.Stats[a1]=_returnDictWithKeys(row,'Stats',evelist)
            DF.Channels[a1]=_returnDictWithKeys(row,'Channels',evelist)
            DF.Events[a1]=DF.MPtd[a1].keys()
        DF['SampleTrims']=[{} for x in range(len(DF))]
        singlesdict[row.Station]=DF
    return singlesdict

def _makeSSDF(row): #Recast row of TRDF into a dataframe for subpace creation
    DF=pd.DataFrame(index=range(len(row.Clust)),columns=[x for x in row.index if x!='Clust'])
    DF['Name']=['SS%d' % x for x in range(len(DF))]
    DF['Events']=object
    DF['AlignedTD']=object
    DF['SVD']=object
    DF['UsedSVDKeys']=object
    DF['FracEnergy']=object
    DF['SVDdefined']=False
    DF['SampleTrims']=[{} for x in range(len(DF))]  
    DF['Threshold']=np.float
    DF['SigDimRep']=object
    DF['FAS']=object
    DF['NumBasis']=int
    DF['Offsets']=list
    DF['Station']=row.Station
    for a in DF.iterrows():
        evelist=row.Clust[a[0]]
        DF['Events'][a[0]]=evelist
        #DF.Events[a[0]]=evlist
        DF.MPtd[a[0]]=_returnDictWithKeys(row,'MPtd',evelist)
        DF.MPfd[a[0]]=_returnDictWithKeys(row,'MPfd',evelist)
        DF.Stats[a[0]]=_returnDictWithKeys(row,'Stats',evelist)
        DF.Channels[a[0]]=_returnDictWithKeys(row,'Channels',evelist)
    DF=DF[[len(x)>1 for x in DF.Events]]
    DF.reset_index(drop=True,inplace=True)
    return DF

def _returnDictWithKeys(row,column,evelist): #function used to get only desired values form dictionary 
    temdict={k: row[column].get(k,None) for k in evelist}
    dictout={k: v for k, v in temdict.items() if not v is None}
    return dictout
        
def _getStartOffset(delaysNP,stakey,clunum,TRDF,a,cltrim):
    zeroEvent=TRDF.EventFiles[delaysNP.argmax()]
    filler=glob.glob(os.path.join(zeroEvent,a[1].NETWORK+'.'+a[1].STATION,'*'))[0]
    trim=np.load(glob.glob(os.path.join(filler,'*.npy'))[0])
    TR=obspy.core.read(os.path.join(filler,'*.sac'))
    TR=TR.slice(endtime=obspy.core.UTCDateTime(trim[0])) #trim template to start time
    sstrim=len(TR[0].data)-cltrim*TR[0].stats.sampling_rate # Get subspace offset, subtracted by the cluster trim
    stakey['StartOffset'][a[0]][clunum]=sstrim


class SubSpaceStream(object):
    """ Class used to hold subspaces for detector
    Holds both subspaces (as defined from the SScluster object) and single event clusters, or singles
    """
    
    def __init__(self,singlesDict,subSpaceDict,cl,dtype,Pf,condir):
        self.clusters=cl
        self.condir=condir
        self.subspaces=subSpaceDict
        self.singles=singlesDict
        self.dtype=dtype
        self.Pf=Pf
        self.ssStations=self.subspaces.keys()
        self.singStations=self.singles.keys()     
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
        
    def SVD(self,selectCriteria=2,selectValue=0.8,conDatNum=100,Threshold=None,normalize=True,**kwargs): 
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
            the SVD vectors but also gives noisy low-amplitude events (if used) equal weight, which may not be desirable 
        kwargs are arguments passed to the getFas call, see it's doc string for info
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
                    U, s, Vh = scipy.linalg.svd(np.transpose(Arr), full_matrices=False)                 
                for eival in range(len(s)): #make dictionary of SVD with singular value as key and basis vector as value
                    svdDict[s[eival]]=U[:,eival]
                    
                #asign Parameters back to subspace dataframes
                self.subspaces[station].SVD[row[0]]=svdDict #assign SVD
                fracEnergy=self._getFracEnergy(row,svdDict,U)
                self.subspaces[station].FracEnergy[row[0]]=fracEnergy
                self.subspaces[station].UsedSVDKeys[row[0]]=self._getUsedBasis(row,svdDict,self.subspaces[station].FracEnergy[row[0]],selectCriteria,selectValue)
                self.subspaces[station].SVDdefined[row[0]]=True
                self.subspaces[station].NumBasis[row[0]]=len(self.subspaces[station].UsedSVDKeys[row[0]])

        self._setThresholds(selectCriteria,selectValue,conDatNum,Threshold,basisLength)
        

                    
    def _setThresholds(self,selectCriteria,selectValue,conDatNum,Threshold,basisLength):
        if Threshold>0:
            for a,station in enumerate(self.ssStations): #Iterate through all subspaces defined by cluster object
                subspa=self.subspaces[station]
                for row in subspa.iterrows():
                    self.subspaces[station].Threshold[row[0]]=Threshold
        elif selectCriteria == 1:
            for a,station in enumerate(self.ssStations):
                subspa=self.subspaces[station]
                basisdims= [len(x) for x in subspa.UsedSVDKeys.values]
                print 'Estimating effective dimesion of signal space for subspace %d' %a
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

                    TH=scipy.stats.beta.isf(self.Pf,beta_a,beta_b) #get threshold
                    if TH>.9:
                        TH,Pftemp=self._approximateThreshold(beta_a,beta_b,self.Pf,1000,3)
                        print 'Scipy.stats.beta.isf failed with pf=%e, approximated Threshold to %f with a Pf of %e for station %s %s using forward grid search'%(self.Pf,TH,Pftemp,station,row.Name)

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
                raise Exception ('Grind search failing, set threshold manually')
        return bestX,bestPf
            
    def _trimGroups(self,row,keys,station): # basic function to get trimed subspaces
        if 'Starttime' in row[1].SampleTrims.keys() and 'Endtime' in row[1].SampleTrims.keys():
            stim=row[1].SampleTrims['Starttime']
            etim=row[1].SampleTrims['Endtime']
            Arr=np.vstack([row[1].AlignedTD[x][stim:etim]-np.mean(row[1].AlignedTD[x][stim:etim]) for x in keys])
            #basisLength=row[1].SampleTrims['Endtime']-row[1].SampleTrims['Starttime']
            basisLength=Arr.shape[1]
        else:
            print ('No trim times for %s and station %s, try running pickSubSpaceTimes'%(row[1].Name,station))
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
            repvect=np.insert(np.square(scipy.dot(np.transpose(U),aliwf)/scipy.linalg.norm(aliwf)),0,0)
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
        Function sample the continous data and plot the calculated thresholds agaisnt a histogram of detection statistics
        created using random samples with no high amplitude signals
        
        Parameters
        ------
        
        conDatNum : int
            The number of continous data chunks to use in the sampling
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
                tr.stats.station='Sta'
                st+=tr
            return st
            
    def pickSubSpaceTimes(self,duration=30,traceLimit=15):
        """
        Calls a modified version of obspyck to generate GUI so user can manually select start times of events groups,
        triming down each waveform group will significantly decreate the runtime.
        
        Parameters
        
        --------------
        
        duration : real number
            the time after the first pick (in seconds) to trim waveforms for SVD. The fact the stream is multiplexed is taken into account.
            If None is passed then the last pick will be used as the end time for truncating waveforms
        traceLimit : int
            Limits the number of traces that will show up to be manually picked to the first in traceLimit. Avoids killing the GUI with too
            many events. 
            
        Notes
        -------------
        If the waveforms looked to be aligned poorly it will result in a higher order of repressentation required to capture all the energy in
        the event vectors and significantly increased runtime. The required correlation coeficient may have been set too high in the 
        initial clustering.
        """
    
        self._pickTimes(self.subspaces,duration,traceLimit)
        

    def pickSinglesTimes(self,duration=30,traceLimit=15):
        """
        Calls a modified version of obspyck to generate GUI so user can manually select start times of events,triming down
        each single to only waveforms  will significantly decreate the runtime.
        
        kwargs include:
        
        duration- the time after the first pick (in seconds) to trim waveforms for SVD. The fact the stream is multiplexed is taken into account.
        If None is passed then the last pick will be used as the end time for truncating waveforms
        """
        self._pickTimes(self.singles,duration)
        
    def _pickTimes(self,trdfDict,duration,traceLimit):
        
        """
        method to call modified obpyck to pick start and optionally stop times for aligned templates.
        This is a good idea to clip out un-needed segments of data before preforming SVD to get subspace basis
        Will also greatly speed up the subspace as a detector if basis vectors are shorter
        
        
        duration is the duration from the first arrivial pick to define waveforms
        """
        for station in trdfDict.keys():
            for starow in trdfDict[station].iterrows():
                if not starow[1].SampleTrims: # If the sample trim dictionary is empty
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
                        print 'aborting picking, progress saved'
                        return None
        
        self._updateOffsets(trdfDict)
        
    def _updateOffsets(self,trdfDict):
        """
        Calculate offset (predicted origin times), throw out extreme outliers using median and median scaling
        """
        for station in trdfDict.keys():
            for num,row in self.subspaces[station].iterrows():
                keys=row.Stats.keys()
                offsets=[row.Stats[x]['offset'] for x in keys]
                trdfDict[station].Offsets[num]=self._getOffsets(np.array(offsets))
    
    def _getOffsets(self,offsets, m = 25.):
        """
        Get offsets, reject outliers bassed on median values (accounts for possible missmatch in events and origin times)
        """
        d = np.abs(offsets - np.median(offsets))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        offs=offsets[s<m]
        return [np.min(offs),np.median(offs),np.max(offs)]
        
    def getFAS(self,conDatNum,ConDir='ContinousWaveForms',LTATime=5,STATime=0.5,staltalimit=8.0,useSubSpaces=True,useSingles=False,numBins=401,recalc=False):
        
        """ 
        Function to initialize a FAS class object, used primarily for smapling the detection distribution of the different subspaces and singles
        
        
        conDatNum is the number of continous data files (by default in hour chunks) to use
        
        ConDir is the directory in the detex.getdata format where contious waveforms are located
        
        LTATime is the long term average time window in seconds used for checking continous data
        
        STATime is the short term average time window in seconds for checking continous data
        
        staltalimit is the value at which continous data gets rejected as too noisey (IE transient signals are present)
        
        ContinousDataLength is the length, in seconds, of the longest continous data chunk, by defualt 1 hour 2 minutes is used to avoid skipping any data
        
        useSubspace and useSingles are both booleans to indicate if the action should be preformed for subspaces, singles, or both
        
        numBins is for storing distribution data, the number of bins between -1 and 1 that will be used in the histogram
        
        """
        
        if useSubSpaces:
            self._updateOffsets(self.subspaces) #make sure offset times are up to date
            for station in self.subspaces.keys():
                if isinstance(self.subspaces[station]['FAS'][0],dict) and not recalc:
                    print('FAS for station %s already calculated, to recalculate pass True to the parameter recalc' % station)
                else:
                    self.subspaces[station]['FAS']=_initFAS(self.subspaces[station],conDatNum,self.clusters,ConDir=ConDir,
                    LTATime=LTATime,STATime=STATime,staltalimit=staltalimit,numBins=numBins,dtype=self.dtype)
        if useSingles:
            raise Exception('Singles not yet implemented')
            for a in range(len(self.singles)):
                if isinstance(self.singles[a]['FAS'][0],dict) and not recalc:
                    print('FAS for single event %d already calculated, to recalculate pass True to the parameter recalc' % a)
                else:
                    self.singles[a]['FAS']=_initFAS(self.subspaces[a],conDatNum,self.clusters,ConDir=ConDir,
                    LTATime=LTATime,STATime=STATime,staltalimit=staltalimit,numBins=numBins,dtype=self.dtype)
    

    def detex(self,UTCstart=None,UTCend=None,subspaceDB='SubSpace.db',trigCon=0,triggerLTATime=5,triggerSTATime=0,
        multiprocess=False,delOldCorrs=True,extrapolateTimes=True, calcHist=True,useSubSpaces=True,useSingles=False,
        estimateMags=True,pks=None,eventDir=None,eventCorFile='EventCors.pkl',UTCSaves=None,fillZeros=False):
        """
        function to run subspace detection over continous data and store results in SQL database subspaceDB
        
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
            If True the subspaces will be used as detectors to scan continous data
        useSingles : boolean
            If True the singles (events that did not cluster) will be used as detectors to scan continous data (DOESNT WORK YET)
        estimateMags : boolean
            If True magnitudes will be estimated for each detection by projecting the continous data where the detection occured into the subspace,
            as well as all the template waveforms that went into the subspace's creation, then applying the iterative magnitude scaling method 
            in Gibbons and Ringdal 2006. Because a representation of the singal is used (IE the projection of the continous data into the subspace)
            this method is less vulnerable to noise distortions when estimating lower magnitudes.
        pks : None or str
            A path to the pks file of the same form as the once created by detex.util.trimTemplates that will be used for estimating P and S arrival times
            for each detection. If False, or the path is not readable with the pandas.read_pickle method no arrivial time estiamtes will be used
        eventDir : None or str
            If a path to an event directory is passed (IE EventWaveForms) the subspace detector will be run on all the events in the eventwaveform
            file rather than on the continous data directory
        eventCorFile : str
            A path to a new pickled data frame created when the eventDir option is used. Records the highest detection statistic in the file for each 
            event, station, and subspace. Useful when trying to characterize events. 
        UTCSaves : None or list of obspy.core.DateTime readable objects
            Either none (no effect) or an iterrable  of obspy.core.UTCDateTime readable objects. For continous data chunk being scanned if a time in the 
            UTCdate falls within the start time and end time of the continous data the vector of SD, along with cotninous data, thresholds, etc. is save to
            a pickled dataframe of the name "UTCsaves.pkl"
        fillZeros : boolean
            If true fill the gaps in continous data with 0s. If True STA/LTA of detection statistic cannot be calculated in order to avoid dividing by 0            
        """
    
        if multiprocess or trigCon!=0 or useSingles: #make sure no parameters that dont work yet are selected
            raise Exception ('multiprocessing, trigcon other than 0, and useSingles not yet supported')
        if useSubSpaces:
            TRDF=self.subspaces
            if not all([y['SVDdefined'] for x in TRDF.keys() for num,y in self.subspaces[x].iterrows()]): #make sure SVD has been performed
                raise Exception('subspace not yet defined, call SVD before attempting to run subspace detectors')
            Det=SSDetex(TRDF,UTCstart,UTCend,self.condir,self.clusters.indir,subspaceDB,trigCon,triggerLTATime,triggerSTATime,
            multiprocess,self.clusters.filt,self.clusters.decimate,delOldCorrs,extrapolateTimes,calcHist,self.dtype,estimateMags,
            pks,eventDir,eventCorFile,UTCSaves,fillZeros)
            
            self.histSubSpaces=Det.hist
#            if any([x is not None for x in Det.hists]): #see if histograms were returned
#                self.subspaces[a]['Hist']=Det.hists
        if useSingles:
            TRDF=self.singles
            Det=SSDetex(TRDF,self.Pf,UTCstart,UTCend,self.ConDir,self.clusters.indir,subspaceDB,trigCon,triggerLTATime,triggerSTATime,
            multiprocess,self.clusters.filt,self.clusters.decimate,delOldCorrs,extrapolateTimes,calcHist,self.dtype)
            self.histSingles=Det.hist
#            if any([x is not None for x in Det.hists]): #see if histograms were returned
#                self.singles[a]['Hist']=Det.hists
        if useSubSpaces or useSingles: # save addational info to sql database
            dffilt=pd.DataFrame([self.clusters.filt],columns=['FREQMIN','FREQMAX','CORNERS','ZEROPHASE'],index=[0])
            detex.util.saveSQLite(dffilt,subspaceDB,'filt_params')
            sslist=[] #empty list to put DFs for each subspace/station pair
            sshists=[pd.DataFrame([['Bins','Bins',json.dumps(Det.hist['Bins'].tolist())]],columns=['Name','Sta','Value'])]
            for sta in self.ssStations:
                for num, ss in self.subspaces[sta].iterrows():
                    name=ss.Name
                    station=ss.Station
                    events=','.join(ss.Events)
                    numbasis=ss.NumBasis
                    thresh=ss.Threshold
                    sslist.append(pd.DataFrame([[name,station,events,thresh,numbasis]],columns=['Name','Sta','Events','Treshold','NumBasisUsed']))
                for skey in Det.hist[sta]:
                    sshists.append(pd.DataFrame([[skey,sta,json.dumps(Det.hist[sta][skey].tolist())]],columns=['Name','Sta','Value']))
            detex.util.saveSQLite(pd.concat(sslist,ignore_index=True),subspaceDB,'ss_info')
            detex.util.saveSQLite(pd.concat(sshists,ignore_index=True),subspaceDB,'ss_hist')
            #deb(Det)
            #dfsingkey=pd.DataFrame() #create singles info to save to sql database
            #dfsingkey['Single']=','.join([x.Events.values[0][0] for x in self.singles]) # make list of subspaces
            #dfsskey['Stations']=','.join([x.Station.values[0] for x in self.singles]) # get all stations in each subspace
            #detex.util.saveSQLite(dfsskey,subspaceDB,'single_info')
      
    def __getitem__(self,key): #make object indexable
        if isinstance(key,int):
            return self.subspaces[self.ssStations[key]]
        elif isinstance(key,str):
            if len(key.split('.'))==2:
                return self.subspaces[self._stakey2[key]]
            elif len(key.split('.'))==1:
                return self.subspaces[self._stakey1[key]]
            else:
                raise Exception('%s is not a station in this cluster object' % key)
        else:
            raise Exception ('%s must either be a int or str of station name')
            
    def __len__(self): 
        return len(self.subspaces) 
                
        
########################## Shared Subspace and Cluster Functions ###########################################
        
def _loadEvents(filelist,indir,filt,trim,stakey,templateDir,decimate,temkey,dtype):# Load file list and apply filters, trims
    #Initialize TRDF, a container for a great many things including event templates, multiplexed data, obspy traces etc.     
    TRDF=pd.DataFrame()
    if not isinstance(filelist,list) and not filelist==None:
        raise Exception ('filelist must be set to either a list or None')
    stakey=stakey[[isinstance(x,str) for x in stakey.STATION]]
    TRDF['Station']=[str(x)+'.'+str(y) for x,y in zip(stakey.NETWORK.values,stakey.STATION.values)]
    if not isinstance(filelist,list):
        eventFiles=np.array([os.path.join(indir,x) for x in temkey.NAME])
        tempsExist=np.array([os.path.exists(x) for x in eventFiles]) # see if each tempalte is found where expected
        print('%s does not exist, check templatekey'%x for x in eventFiles[~tempsExist])
        existingEventFiles=eventFiles[tempsExist]
    else:
        eventFiles=np.array([os.path.join(indir,x) for x in filelist])
        tempsExist=np.array([os.path.exists(x) for x in eventFiles])
        print('%s does not exist, check templatekey'%x for x in eventFiles[~tempsExist])
        existingEventFiles=eventFiles[tempsExist]
    if len(existingEventFiles)<1: #make sure there are some files to work with
        raise Exception('No file paths in eveFiles, %s is probably empty'%indir)

    TRDF['Events']=list
    TRDF['MPtd']=object
    TRDF['MPfd']=object
    TRDF['Channels']=object
    TRDF['Stats']=object
    TRDF['Link']=list
    TRDF['Clust']=list
    TRDF['Lags']=object
    TRDF['CCs']=object
    TRDF=TRDF[[isinstance(x,str) for x in TRDF.Station]] #get rid of any possible NaN stations
    #Make list in data frame that shows which stations are in each event
    # Load streams into dataframe to call later
    for a in TRDF.iterrows():
        TRDF.MPtd[a[0]]={}
        TRDF.MPfd[a[0]]={}  
        Streams,TRDF['Events'][a[0]],TRDF['Channels'][a[0]],TRDF['Stats'][a[0]]=_loadStream(eventFiles,templateDir,filt,trim,decimate,a[1].Station,dtype)
        for key in TRDF['Events'][a[0]]:      
            Nc=len(TRDF['Channels'][a[0]][key])
            TRDF.MPtd[a[0]][key],TR=multiplex(Streams[key],Nc,retTR=True) # multiplex channs
            TRDF['Stats'][a[0]]['starttime']=TR[0].stats.starttime.timestamp #update starttime
            reqlen=2*len(TRDF.MPtd[a[0]][key]) #required length with 0 padding
            TRDF.MPfd[a[0]][key]=scipy.fftpack.fft(TRDF.MPtd[a[0]][key],n=2**reqlen.bit_length())
    TRDF.reset_index(inplace=True,drop=True) 
    return TRDF
    
def _testStreamLengths(TRDF):
    for a in TRDF.iterrows():
        medlen=np.median([len(x) for x in a[1].MPtd.values()])
        keysToKill=[x for x in a[1].Events if len(a[1].MPtd[x])!= medlen]
        tmar=np.array(TRDF.Events[a[0]])
        tk=[[not x in keysToKill for x in TRDF.Events[a[0]]]]
        #deb([tmar,tk])
        TRDF.Events[a[0]]=tmar[np.array(tk)[0]]
        for key in keysToKill:
            TRDF.MPtd[a[0]].pop(key,None)
            TRDF.MPfd[a[0]].pop(key,None)
    return TRDF
    
    
def _alignTraces(traces,delayNP): #function that takes obspy streams with 1 length and delay values in samples and allignes traces, then return np arrays
    if any([len(x)!=1 for x in traces]): #Make sure each trace has a length less than 2
        raise Exception('Only one trace should be in each stream, debug code')
    nptraces=[x[0].data for x in traces]
     #make all shifts negative so begining of traces can be truncated for allignment
    for a in range(len(nptraces)):
        nptraces[a]=nptraces[a][np.abs(delayNP[a]):]
    minlen=np.min([len(x) for x in nptraces])
    for a in range(len(nptraces)): #trim to make all traces same size
        nptraces[a]=nptraces[a][:minlen]
        
    return nptraces


    
def _getDelays(link,CCtoLag,cx,lags,sa,cxdf):
    N=len(link)
    linkup=np.append(link,np.arange(N+1,2*N+1).reshape(N,1),1) #append cluster numbers to link array
    clustDict=_getClustDict(linkup,len(linkup))
    dflink=pd.DataFrame(linkup,columns=['i1','i2','cc','num','clust'])
    if len(dflink)>0:
        dflink['II']=list
    dflink['ev1']=0
    dflink['ev2']=0
    for a in dflink.iterrows():
        #dflink['II'][a[0]]=clustDict[a[1].i1].values.tolist()+clustDict[a[1].i2].values.tolist()
        dflink['II'][a[0]]=clustDict[int(a[1].i1)].tolist()+clustDict[int(a[1].i2)].tolist()
        tempdf=cxdf[cxdf==a[1].cc].dropna(how='all').dropna(axis=1) #get a dataframe with only index as ev1 and column as ev2, not efficient consider revising
        dflink.ev1[a[0]],dflink.ev2[a[0]]=tempdf.index[0],tempdf.columns[0]   
    lags=_downTheLagHole(dflink,cx,lags,CCtoLag,clustDict,clustDict.iloc[-1])
    return lags
    #return clusts,lagByClust   
        
def _downTheLagHole(dflink,x,lags,CCtoLag,clustDict,clus): #Function to follow ind1 through clustering linkage in linkup and return total offset time
    if len(dflink)<1: #if event falls in its own cluster return simply the event with a zero
        lagSeries=pd.Series([0],index=clus)
    else:       
        #nummap=pd.Series(range(len(dflink)+1),index=dflink['II'][dflink.index.values.max()]) #map iteration of event group to event
        allevents=dflink['II'][dflink.index.values.max()] #return integer of all events
        allevents.sort()
        lagSeries=pd.Series([0]*(len(dflink)+1),index=allevents) #total lags for each station
        #deb([dflink,lagSeries,clustDict])
         #map int series to actual event number as defined in DFT
        for a in dflink.iterrows():
            if a[1].ev1 in clustDict[int(a[1].i1)]:
                cl22=clustDict[int(a[1].i2)]
            else:
                cl22=clustDict[int(a[1].i1)]
#            if np.min(clustDict[a[1].i2]) < np.min(clustDict[a[1].i1]):
#                cl22=clustDict[a[1].i1]
#            else:
#                cl22=clustDict[a[1].i2]
            #print cl22
            currentLag=CCtoLag[a[1].cc]
            for b in cl22: #record and update lags (or offsets) for second cluster                
                lagSeries[b]+=currentLag # 
                #lags=_updateLags(nummap[b],lags,len(df),currentLag)
                lags=_updateLags(b,lags,len(dflink),currentLag)
            CCtoLag=_makeLagSeries(x,lags)
    return lagSeries
        
        
def _updateLags(evenum,lags,N,currentLag):#function to add current lag shifts to effected lag times (see Haris 2006 appendix B)
    #too many python loops, make cython code when possible
    dow=_getDow(N,evenum) #get the index to add to lags for columns
    acr=_getAcr(N,evenum)
    for a in acr:
        lags[a]+=currentLag
    for a in dow:
        lags[a]-=currentLag
    return lags

def _getDow(N,evenum):
    dow=[0]*evenum
    if len(dow)>0:
        for a in range(len(dow)):
            dow[a]=_triangular(N-1)-1+evenum-_triangular(N-1-a)
            #dow[a]=_triangular(N)+evenum-_triangular(N+1-a)-1
    return dow
    
def _getAcr(N,evenum):
    acr=[0]*(N-evenum)
    if len(acr)>0:
        acr[0]=_triangular(N)-_triangular(N-(evenum))
        for a in range(1,len(acr)):
            acr[a]=acr[a-1]+1
    return acr

def _triangular(n): #calculate sum of triangle with base N, see http://en.wikipedia.org/wiki/Triangular_number
    return sum(range(n+1))

def _getClustDict(linkup,N): #get pd series that will define the base events in each cluster (including intermediate clusters)
    clusdict=pd.Series([ np.array([x]) for x in np.arange(0,N+1)],index=np.arange(0,N+1)) 
    for a in range(len(linkup)):
        clusdict[int(linkup[a,4])]=np.append(clusdict[int(linkup[a,0])],clusdict[int(linkup[a,1])])
    return clusdict          
def _ensureUnique(cx,cxdf): #make sure each coeficient is unique so it can be used as a key to reference time lags, if not unique perturb slightly
    se=pd.Series(cx)
    dups=se[se.duplicated()]
    count=0
    while len(dups)>0:
        print 'Duplicates found in correlation coefficients, perturbing slightly to get unique values'
        for a in dups.iteritems():
            se[a[0]]=a[1]-abs(.00001*np.random.rand())
        count+=1
        dups=se[se.duplicated()]
        if count>10:
            raise Exception('cannot make Coeficients unique, killing program')
    if count>1: # if the cx has been perturbed update cxdf
        for a in range(len(cxdf)):
            sindex=sum(pd.isnull(a[1]))
            cxdf.values[a,sindex:]=cx[_triangular(len(cxdf))-_triangular(len(cxdf)-a),_triangular(len(cxdf))-_triangular(len(cxdf)-(a+1))]       
    return se.values,cxdf
    
def _makeLagSeries(x,lags):
    LS=pd.Series(lags,index=x)
    return LS    
    
def _loadStream (eventFiles,templateDir,filt,trim,decimate,station,dtype): #loads all traces into stream object and applies filters and trims
    StreamDict={} # Initialize dictionary for stream objects
    channelDict={}
    stats={}
    STlens={}
    trLen=[]#trace length
    allzeros=[] # empty list to stuff all zero keys into in order to delet later
    for eve in eventFiles:
        if templateDir==False:
            try:
                ST=_applyFilter(obspy.core.read(eve),filt,decimate,dtype)
            except:
                continue
        else:
            try:
                ST=_applyFilter(obspy.core.read(os.path.join(eve,station+'*')),filt,decimate,dtype)
            except:
               # print('could not read %s or preprocessing failed' %os.path.join(eve,station+'*') )
                continue
        #ST=_applyFilter(ST,filt,decimate,dtype)
        Nc=len(list(set([x.stats.channel for x in ST]))) #get number of channels
        if isinstance(trim,list) or isinstance(trim,tuple):
            try: #if trim doesnt work (meaning event waveform incomplete) then skip
                ST.trim(starttime=ST[0].stats.starttime+trim[0],endtime=ST[0].stats.starttime+trim[0]+trim[1])
            except ValueError:
                continue
        if Nc != len(ST): #if length is not at least num of channels (IE if not all channels present) skip trace
            continue
        evename=os.path.basename(eve)
        StreamDict[evename]=ST
        channelDict[evename]=[x.stats.channel for x in ST]
        stats[evename]={'processing':ST[0].stats['processing'],'sampling_rate':ST[0].stats.sampling_rate,'starttime':ST[0].stats.starttime.timestamp,'Nc':Nc}
        totalLength=np.sum([len(x) for x in ST])
        if any([not np.any(x.data) for x in ST]):
            allzeros.append(evename)
        trLen.append(totalLength)
        STlens[evename]=totalLength
    mlen=np.median(trLen)
    keysToRemove=[x for x in StreamDict.keys() if STlens[x] < mlen*.2]
    for key in keysToRemove: # delete keys of events who have less than 80% of the median data points 
        print '%s is fractured or much shorter than the other events, deleting' %key
        StreamDict.pop(key,None)
        channelDict.pop(key,None)
        stats.pop(key,None)
    for key in set(allzeros):
        #deb([StreamDict,allzeros])
        print '%s has at least one channel that is all zeros, deleting' % key
        StreamDict.pop(key,None)
        channelDict.pop(key,None)
        stats.pop(key,None)
    if len(StreamDict.keys())<2:
        raise Exception('Less than 2 events survived preprocessing for station %s. Check input parameters, especially trim' % station)
    evlist= StreamDict.keys()
    evlist.sort()
    return StreamDict, evlist, channelDict, stats
            
def multiplex(TR,Nc,trimTolerance=15,Template=False,returnlist=False,retTR=False):
    if Nc==1:
        C1=TR[0].data
        C=TR[0].data
    else:
        chans=[x.data for x in TR]
        minlen=np.array([len(x) for x in chans])  
        if max(minlen)-min(minlen) > 15:
            if Template:
                raise Exception('timTolerance exceeded, examin Template: \n%s' % TR[0])
            else:
                print('timTolerance exceeded, examin data, Trimming anyway: \n%s' % TR[0])
                trimDim=min(minlen)
                chansTrimed=[x[:trimDim] for x in chans]
        elif max(minlen)-min(minlen)>0 : #trim a few samples off the end if necesary
            trimDim=min(minlen)
            chansTrimed=[x[:trimDim] for x in chans]
        elif max(minlen)-min(minlen)==0:
            chansTrimed=chans
        C=np.vstack((chansTrimed))
        C1=np.ndarray.flatten(C,order='F')
    out=[C1]
    if returnlist:
        out.append(C)
    if retTR:
        out.append(TR)
    if len(out)==1:
        return out[0]
    else:
        return out
 
        
def _mergeChannels(TR): #function to find longest continous data chucnk and discard the rest
    channels=list(set([x.stats.channel for x in TR]))
    temTR=TR.select(channel=channels[0])
    lengths=np.array([len(x.data) for x in temTR])
    lemax=lengths.argmax()
    TR.trim(starttime=TR[lemax].stats.starttime,endtime=TR[lemax].stats.endtime)
    return TR

def _mergeChannelsFill(TR):
    TR.merge(fill_value=0.0)
    return TR

############################## Subspace Detex and FAS ###################################################

def _initFAS(TRDF,conDatNum,cluster,ConDir='ContinousWaveForms',LTATime=5,STATime=0.5,numBins=401,dtype='double',staltalimit=6.5):
    """ Function to randomly scan through continous data and fit statistical distributions to histograms in order to get a suggested value
    for various trigger conditions"""
    #TO DO- write function to test length of continous data rather than making user define it
    results=[0]*len(TRDF)
    histBins=np.linspace(-.01,1,num=numBins) #create bins for histograms
    ContinousDataLength=detex.util.getContinousDataLength(Condir=ConDir)
    for a in TRDF.iterrows(): #Loop through each station on the subspace or singles data frame   
        try:     
            results[a[0]]={'bins':histBins}
        except:
            deb([a,TRDF,results])
        chans=list(set(np.concatenate([x for x in a[1].Channels.values()])))
        chans.sort()
        Nc=len(chans)
        
        ssArrayTD,ssArrayFD,reqlen,Nc=_loadMPSubSpace(a,ContinousDataLength) #load subspace rep for given station/event
        if len(ssArrayTD)<1:
            deb([a,ContinousDataLength,ssArrayTD,ssArrayFD,reqlen,Nc])
        #hist=[0.0]*(len(histBins)-1)
        #histRev=[0.0]*(len(histBins)-1)
        jdays=_mapjdays(a,ConDir)
        if conDatNum>len(jdays)*(3600/ContinousDataLength)*4: #make sure there are at least conDatNum samples avaliable
            print 'Not enough continous data for conDatNum=%d, decreasing to %d' %(conDatNum,len(jdays)*(3600/ContinousDataLength)*4)
            conDatNum=int(len(jdays)*(3600/ContinousDataLength)*4)
        
        CCmat=[0]*conDatNum
        #ccmatRev=[0]*conDatNum #reversed ccmat
        #MPtemFD,sum_nt=self.FFTemplate(MPtem,reqlen) #multplexed template in frequency domain
        usedJdayHours=[] #initialize blank list that will be used to make sure repeated chunks of data are not used
        for c in range(conDatNum):
            condat,usedJdayHours=_loadRandomContinousData(a,jdays,cluster,usedJdayHours,dtype,LTATime,STATime,staltalimit)
            if c<10 and condat[0].stats.endtime-condat[0].stats.starttime > ContinousDataLength: #test first 10 random samples for length requirement
                deb([c,condat,ContinousDataLength,a,jdays,cluster,usedJdayHours,dtype,LTATime,STATime,staltalimit])
                raise Exception('Continous data read in is %d, which is longer than %d seconds, redefine ContinousDataLength when calling getFAS' 
                % (int(condat[0].stats.endtime-condat[0].stats.starttime),ContinousDataLength))
            MPcon=multiplex(condat,Nc)
            CCmat[c]=_MPXSSCorr(MPcon,reqlen,ssArrayTD,ssArrayFD,Nc)
            #ccmatRev[c]=_MPXSSCorrTR(MPcon,reqlen,ssArrayTD,ssArrayFD,Nc)
        if dtype=='double':
            CCs=np.fromiter(itertools.chain.from_iterable(CCmat), dtype=np.float64)
            #CCsrev=np.fromiter(itertools.chain.from_iterable(ccmatRev), dtype=np.float64)
        elif dtype=='single':
            CCs=np.fromiter(itertools.chain.from_iterable(CCmat), dtype=np.float32)
       # deb(CCs)
            #CCsrev=np.fromiter(itertools.chain.from_iterable(ccmatRev), dtype=np.float32)        
        #CCs=np.array(CCmat).flatten()
        #deb(CCs)
        results[a[0]]['bins']=histBins
        results[a[0]]['hist']=np.histogram(CCs,bins=histBins)[0]
        #results[a[0]]['normdist']=scipy.stats.norm.fit(CCs)
        betaparams=scipy.stats.beta.fit(CCs,floc=0,fscale=1)
        results[a[0]]['betadist']=betaparams # enforce hard limits on detection statistic
        results[a[0]]['nnlf']=scipy.stats.beta.nnlf(betaparams,CCs) #calculate negative log likelihood for a "goodness of fit" measure
        #results[a[0]]['histRev']=histRev+np.histogram(CCsrev,bins=histBins)[0]
        
        #deb([CCs,template]) #start here, figure out distribution
    return results
    
    
def _estimateDimSpace(TRDF,conDatNum,cluster,ConDir='ContinousWaveForms',LTATime=5,STATime=0.5,numBins=401,dtype='double',staltalimit=6.5):#estimate the dimesnion of the signal space, see Harris 2006 Theory Equation 17-19
    """ Function to randomly scan through continous data and fit statistical distributions to histograms in order to get a suggested value
    for various trigger conditions"""
    #TO DO- write function to test length of continous data rather than making user define it
    results=[0]*len(TRDF)
    ContinousDataLength=detex.util.getContinousDataLength(Condir=ConDir)
    for a in TRDF.iterrows(): #Loop through each station on the subspace or singles data frame
        chans=list(set(np.concatenate([x for x in a[1].Channels.values()])))
        chans.sort()
        Nc=len(chans)
        
        ssArrayTD,ssArrayFD,reqlen,Nc=_loadMPSubSpace(a,ContinousDataLength) #load suspace rep for given station/event
        temlen=ssArrayTD.shape[1] if len(ssArrayTD.shape) >1 else len(ssArrayTD) #get template length
        jdays=_mapjdays(a,ConDir)
        if conDatNum>len(jdays)*(3600/ContinousDataLength)*4: #make sure there are at least conDatNum samples avaliable
            print 'Not enough continous data for conDatNum=%d, decreasing to %d' %(conDatNum,len(jdays)*(3600/ContinousDataLength)*4)
            conDatNum=int(len(jdays)*(3600/ContinousDataLength)*4)
        
        CCmat=[0]*conDatNum
        #MPtemFD,sum_nt=self.FFTemplate(MPtem,reqlen) #multplexed template in frequency domain
        usedJdayHours=[] #initialize blank list that will be used to make sure repeated chunks of data are not used
        for c in range(conDatNum):
            condat1,usedJdayHours=_loadRandomContinousData(a,jdays,cluster,usedJdayHours,dtype,LTATime,STATime,staltalimit)
            condat2,usedJdayHours=_loadRandomContinousData(a,jdays,cluster,usedJdayHours,dtype,LTATime,STATime,staltalimit)
            MPcon1=multiplex(condat1,Nc)
            MPcon2=multiplex(condat2,Nc)
            temind1,temind2=int(np.random.rand()*(len(MPcon2)-temlen)),int(np.random.rand()*(len(MPcon2)-temlen))
            MPtem1=MPcon2[temind1:temind1+temlen]
            MPtem2=MPcon1[temind2:temind2+temlen]
            CCmat[c]=np.concatenate([_MPDimCorr(MPtem1,MPcon1),_MPDimCorr(MPtem2,MPcon2)])
        if dtype=='double':
            CCs=np.fromiter(itertools.chain.from_iterable(CCmat), dtype=np.float64)
        elif dtype=='single':
            CCs=np.fromiter(itertools.chain.from_iterable(CCmat), dtype=np.float32)
        #CCs=np.array(CCmat).flatten()
        dimrep=1+1/np.var(CCs)
        if dimrep>temlen:
            raise Exception ('calculated effective dimesion greater than length of template, this is bad')
        results[a[0]]=dimrep
        #deb([CCs,template]) #start here, figure out distribution
    return results
    
        
def _MPXSSCorr(MPcon,reqlen,ssArrayTD,ssArrayFD,Nc): # multiplex subspace detection statistic function
    MPconFD=scipy.fftpack.fft(MPcon,n=2**reqlen.bit_length())
    n = np.int32(np.shape(ssArrayTD)[1]) #length of each basis vector
    a = pd.rolling_mean(MPcon, n)[n-1:] #rolling mean of continous data
    b1 = pd.rolling_var(MPcon, n)[n-1:]
    b1 *= np.sqrt((n-1.0) / n)
    b = pd.rolling_var(MPcon, n)[n-1:]  # rolling var of continous data
    b *= n #rolling power in vector
    sum_ss=np.sum(ssArrayTD,axis=1) #the sume of all the subspace basis vectors
    av_norm=np.multiply(a.reshape(1,len(a)),sum_ss.reshape(len(sum_ss),1)) #term to account for non-averaged vectors
    m1=np.multiply(ssArrayFD,MPconFD)    
    if1=scipy.real(scipy.fftpack.ifft(m1))[:,n-1:len(MPcon)]-av_norm
    result1=np.sum(np.square(if1),axis=0)/b
    return result1[::Nc]
    
def _MPDimCorr(t,s): #simple correlation for estimation of embeded dimension 
    n = len(t)
    nt = (t-np.mean(t))/(np.std(t)*n)
    sum_nt = nt.sum()
    a = pd.rolling_mean(s, n)[n-1:]
    b = pd.rolling_std(s, n)[n-1:]
    b *= np.sqrt((n-1.0) / n)
    c = np.convolve(nt[::-1], s, mode="valid")
    result = (c - sum_nt * a) / b    
    return result
    
def _MPXSSCorrTR(MPcon,reqlen,ssArrayTD,ssArrayFD,Nc): #time reversed multiplex cross correlation
    MPcon=MPcon[::-1]
    MPconFD=scipy.fftpack.fft(MPcon,n=2**reqlen.bit_length())
    n = int(np.shape(ssArrayTD)[1]) #length of each basis vector
    ns= int(np.shape(ssArrayTD)[0]) #number of subspaces
    a = pd.rolling_mean(MPcon, n)[n-1:]
    b = pd.rolling_std(MPcon, n)[n-1:]
    b *= np.sqrt((n-1.0) / n)
    nt=[scipy.sum(ssArrayTD[x,:]-scipy.mean(ssArrayTD[x,:])) for x in range(ns)] 
    #TODO explore other ways of calculating this, could probably be more efficient
    result=np.sum(np.square(np.array([(np.real(scipy.fftpack.ifft(np.multiply(ssArrayFD[x,:],MPconFD)))[n-1:len(MPcon)] - nt[x] * a) for x in range(ns)])),axis=0)/np.square(b*np.sqrt(n))
    
    #result = np.linalg.norm(np.array([(np.real(scipy.fftpack.ifft(np.multiply(ssArrayFD[x,:],MPconFD)))[n-1:len(MPcon)] - nt[x] * a) for x in range(ns)]),axis=0)/(b*np.sqrt(n))
    
    return result[Nc-1::Nc]

def _loadMPSubSpace(a,ContinousDataLength): # function to load subspace representations
    if 'UsedSVDKeys' in a[1].index: #test if input TRDF row is subspace
        if not isinstance(a[1].UsedSVDKeys,list):
            raise Exception ('SVD not defined, run SVD on subspace stream class before calling false alarm statistic class')
        if not all(x==a[1].Channels.values()[0] for x in a[1].Channels.values()):
            raise Exception ('all stations in subspace do not have the same channels')
        Nc=len(a[1].Channels.values()[0]) #num of channels
        #ssArrayTD=np.array([a[1].SVD[x][trimdex[0]:trimdex[1]] for x in a[1].UsedSVDKeys])
        ssArrayTD=np.array([a[1].SVD[x] for x in a[1].UsedSVDKeys])
        rele=int(ContinousDataLength*a[1].Stats.values()[0]['sampling_rate']*Nc+np.max(np.shape(ssArrayTD)))
        ssArrayFD=np.array([scipy.fftpack.fft(x[::-1],n=2**rele.bit_length()) for x in ssArrayTD])
    #=np.array([a[1].SVD[x] for x in a[1].UsedSVDKeys]) #time domain subspace array
    #ssArrayFD=np.array([scipy.fftpack.fft(a[1].SVD[x],n=reqlen) for x in a[1].UsedSVDKeys]) #fequency domain subspace array
    return ssArrayTD,ssArrayFD,rele,Nc
        
def _getStaLtaArray(self,C,LTA,STA): # Get STA/LTA 
    if STA==0:
        STA=1
        STArray=np.multiply(C,C)
    else:
        STArray=pd.rolling_mean(np.multiply(C,C),STA,center=True)
        STArray=self._replaceNanWithMean(STArray)
    LTArray=pd.rolling_mean(np.multiply(C,C),LTA,center=True)
    LTArray=self._replaceNanWithMean(LTArray)
    out=np.divide(STArray,LTArray)
    return out

            
def _loadRandomContinousData(station,jdays,cluster,usedJdayHours,dtype,LTATime,STATime,staltalimit): #loads random chunks of data from total availible data
    filt=cluster.filt
    failcount=0
    while failcount<50:
        rand1=np.round(np.random.rand()*(len(jdays)-1))
        try:
            WFs=glob.glob(os.path.join(jdays[int(rand1)],'*'))
            rand2=np.round(np.random.rand()*len(WFs))
            if [rand1,rand2] in usedJdayHours:
                failcount+=1
                continue
            ST=obspy.core.read(WFs[int(rand2)])
            TR=_applyFilter(ST,filt)
            TRz=TR.select(component='Z').copy()
            cft = obspy.signal.trigger.classicSTALTA(TRz[0].data, STATime*TRz[0].stats.sampling_rate , LTATime*TRz[0].stats.sampling_rate) #make sure no high amplitude signal is here
            if dtype=='single':
                for num,tr in enumerate(TR):
                    TR[num].data=tr.data.astype(np.float32)
            if cft.max()>staltalimit:
                failcount+=1
                #print 'rejecting continous data'
            else:
                usedJdayHours.append([rand1,rand2])
                break
        except: #allow a certain number of failures to account for various possible data problems
            failcount+=1
        if failcount>49: 
            print 'Not enough traces found with no high amplitude signals, try adjusting the number of continous data files used or changing stalta parameters of\
            the getFAS function'
            deb([rand1,WFs,rand2,ST,TR,TRz,cft,staltalimit,STATime,LTATime])
            raise Exception('something is broked')
    return TR,usedJdayHours
                                 
def _applyFilter(ST,filt,decimate=False,dtype='double',fillZeros=False):# Apply a filter/decimateion to an obspy trace object and trim 
    ST.sort()
    if dtype=='single': #cast into single if desired
        for num,tr in enumerate(ST):
            ST[num].data=tr.data.astype(np.float32)
    Nc=list(set([x.stats.channel for x in ST]))
    if len(ST)>len(Nc): #if data is fragmented only keep largest chunk
        if fillZeros:
            ST=_mergeChannelsFill(ST)
        else:
            ST=_mergeChannels(ST)
    if decimate:
        ST.decimate(decimate)
    startTrim=max([x.stats.starttime for x in ST])
    endTrim=min([x.stats.endtime for x in ST])
    ST=ST.slice(starttime=startTrim,endtime=endTrim) 
    ST.detrend('linear')
    if isinstance(filt,list) or isinstance(filt,tuple):
        ST.filter('bandpass',freqmin=filt[0],freqmax=filt[1],corners=filt[2],zerophase=filt[3])
    return ST              
            
def _mapjdays(row,ConDir):
    station=row[1].Station
    years=glob.glob(os.path.join(ConDir,station,'*'))
    jdays=[0]*len(years)
    for a in range(len(jdays)):
        jdays[a]=glob.glob(os.path.join(years[a],'*'))
    jdays=np.concatenate(np.array(jdays))
    return jdays
    
def _replaceNanWithMean(self,arg): # Replace where Nans occur with closet non-Nan value
    ind = np.where(~np.isnan(arg))[0]
    first, last = ind[0], ind[-1]
    arg[:first] = arg[first+1]
    arg[last + 1:] = arg[last]
    return arg  
    
def fast_normcorr(t, s): # Fast normalized Xcor
    if len(t)>len(s): #switch t and s if t is larger than s
        t,s=s,t
    n = len(t)
    nt = (t-np.mean(t))/(np.std(t)*n)
    sum_nt = nt.sum()
    a = pd.rolling_mean(s, n)[n-1:]
    b = pd.rolling_std(s, n)[n-1:]
    b *= np.sqrt((n-1.0) / n)
    c = np.convolve(nt[::-1], s, mode="valid")
    result = (c - sum_nt * a) / b    
    return result

def _calcAlpha(MPtem,MPcon): 
    """
    calculate alpha using iterative method outlined in Gibbons and Ringdal 2007.
    This is a bit rough and dirty at the moment, there are better methods to use
    """
    #xtem=MPcon[Nc*(trigIndex)-4:Nc*(trigIndex)+len(MPtem)+4]
    #xcor=np.correlate(xtem,MPtem,mode='valid')
    #deb([xtem,xcor,Nc,trigIndex])
    
    X,Y=MPtem,MPcon
    #return np.linalg.norm(Y)/np.linalg.norm(X)
    #deb([X,Y])
    alpha0=np.dot(X,Y)/np.dot(X,X)
    for a in range(10): 
        if a==0:
            alpha=alpha0
            alphaPrevious=alpha
        res=Y-alpha*X
        Weight=1/(.000000001+np.sqrt(np.abs(res)))
        alpha=np.dot(Y,np.multiply(X,Weight))/np.dot(X,np.multiply(X,Weight))
        if (abs(alpha-alphaPrevious))/(alphaPrevious)<0.01: #If change is less than 1% break, else go 10 iterations
            break
        alphaPrevious=alpha
    #print (X.argmax(),Y.argmax(),alpha)    
    return abs(alpha)   

##subspace Detex
class SSDetex(object):
    """
    dummy class to run subspace detections for all events in EveDir with corresponding waveforms in 
    ConDir
    """
    def __init__(self,TRDF,UTCstart,UTCend,ConDir,EveDir,subspaceDB,trigCon,triggerLTATime,triggerSTATime,multiprocess,filt,
                 decimate,delOldCorrs,extrapolateTimes,calcHist,dtype,estimateMags,pks,eventDir,eventCorFile,UTCSaves,fillZeros):
        self.__dict__.update(locals()) # Instantiate all input variables
        if not os.path.exists(ConDir):
            raise Exception('%s does not exist, make sure you are in the correct directory'%ConDir)
        if not eventDir:
            self.ContinousDataLength=detex.util.getContinousDataLength(Condir=ConDir) #get continous data length, will throw error if not all stations have the same length
        else:
            self.ContinousDataLength=detex.util.getEveDataLength(EveDir=eventDir)
        if os.path.exists(subspaceDB) and not delOldCorrs:
            user_input=raw_input('%s already exists, delete it? ("y" or "yes" else do not delete)\n'%subspaceDB)
            if user_input=='yes' or user_input=='y':
                detex.util.DoldDB(subspaceDB)    
        elif delOldCorrs and os.path.exists(subspaceDB):
            detex.util.DoldDB(subspaceDB)
        
        if pks: #Try to read in picks
            try:
                self.pks_df=pd.read_pickle(pks)
            except:
                print 'reading %s failed' % pks
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
            DFsta=TRDF[sta] #make data frame from all subspaces/singles that share a station. This is done to reduce IO costs in reading continous data
            DFsta.reset_index(inplace=True,drop=True)                    
            self.hist[sta]=[None]*len(TRDF)
            if multiprocess==True: #TODO fix this, multiprocessing is broken
                p = multiprocessing.Process(target=self._CorStations(DFsta,sta))
                jobs.append(p)
                p.start()
            else:
                self.hist[sta]=self._CorStations(DFsta,sta)
        if self.eventDir:
            try:
                DFeve=pd.concat(self.eventCorList,ignore_index=True)
                DFeve.to_pickle(self.eventCorFile)
            except ValueError:
                print ('No events in Df concatenated')
        if isinstance(UTCSaves,collections.Iterable):
            try:
                DFutc=pd.concat(self.UTCSaveList,ignore_index=True)
                DFutc.to_pickle('UTCsaves.pkl')
            except ValueError:
                print 'DFutc empty, not saving'
                
    def _CorStations(self,DFsta,sta): 
        channels=list(set([z for x in DFsta.Channels for y in x.values() for z in y]))
        channels.sort()
        samplingRate=list(set([y['sampling_rate'] for x in DFsta.Stats for y in x.values()])) #get sampling rates
        Names=DFsta.Name.values
        threshold={x.Name:x.Threshold for num,x in DFsta.iterrows()} #dictionary of required coeficients 
        Names.sort()
        histdic={na:[0.0]*(len(self.hist['Bins'])-1) for na in Names}
        if len(samplingRate)>1:  # Make sure there is only one sampling rate for all subspaces using current station station
            raise Exception('More than one sampling rate found for station %s in subspace detection, aborting' % DFsta.Station[0])
        else:
            samplingRate=samplingRate[0]
        contrim={x[1].Name:-(self.ContinousDataLength % 3600)+np.shape(np.array(DFsta.SVD[x[0]].values()))[1]/(samplingRate*len(channels)) for x in DFsta.iterrows()}
        
        staPksDf=None        
        if self.pks:
            staPksDf=self.pks_df[self.pks_df.Station==sta]
        if not self.eventDir:
            histdict=self._CorConDat(threshold,histdic,sta,channels,contrim,Names,DFsta,samplingRate,staPksDf)
        else:
            if os.path.exists(self.eventDir):
                histdict=self._CorEventDat(threshold,histdic,sta,channels,contrim,Names,DFsta,samplingRate,staPksDf)
            else:
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
            print 'No events found for %s'%sta
        else:
            ssArrayTD,ssArrayFD,reqlen,offsets,mags,eventWaveForms,events,WFU,UtU,pwave,swave=self._loadMPSubSpace(DFsta,self.ContinousDataLength,sta,channels,samplingRate,returnFull=True,PKS=staPksDf)
            for fileToCorr in tempFiles: #loop through each chunk of continous data
                CorDF,MPcon,ConDat=self._getRA(ssArrayTD,ssArrayFD,fileToCorr,len(channels),reqlen,contrim,Names)
                if not isinstance(CorDF,pd.DataFrame): #if something is broken skip hour
                    print '%s failed' % fileToCorr 
                    
                    continue
                for name,row in CorDF.iterrows(): # iterate through each subspace/single
                    self.eventCorList.append(pd.DataFrame([[sta,name,row.File,row.MaxSD,row.MaxSTALTA]],columns=['Station','Subspace','File','SD','SD_STALTA']))
                    if self.calcHist and len(CorDF)>0: #If calculating histogram of Statistic of detection 
                        try:
                            histdic[name]=histdic[name]+np.histogram(row.SSdetect,bins=self.hist['Bins'])[0]
                        except:
                            print 'binning failed'
                    if isinstance(self.UTCSaves,collections.Iterable):
                       self._makeUTCSaveDF(row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD)
                    if self._evalTriggerCondition(row,name,threshold): # Trigger Condition
                        Sar=self._CreateCoeffArray(row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD,WFU,UtU,staPksDf,pwave,swave)
                        if len(Sar)>300:
                            print 'over 300 events found in single continous data chunk, perphaps minCoef is too low?' 
                        if len(Sar)>0:
                            DF=DF.append(Sar,ignore_index=True)
                        if len(DF)>500:
                            numdets+=500
                            detex.util.saveSQLite(DF,self.subspaceDB,'ss_df')
                            DF=pd.DataFrame()
            if len(DF)>0:
                detex.util.saveSQLite(DF,self.subspaceDB,'ss_df')
            print ('%s completed, %d potential detection(s) recorded' %(sta,len(DF)+numdets))
        return histdic

        
            
    def _CorConDat(self,threshold,histdic,sta,channels,contrim,Names,DFsta,samplingRate,staPksDf):
        """
        Function to use when subspace is to be run over continous data
        """
        conrangejulday,conrangeyear=self._getContinousRanges(self.ConDir,sta,self.UTCstart,self.UTCend) 
        numdets=0
        if len(conrangeyear)==0:
            print ('No data for %s, check Continous data directory' %(sta))
        DF=pd.DataFrame() # Initialize emptry data frame, will later be dumped to SQL database
        ssArrayTD,ssArrayFD,reqlen,offsets,mags,eventWaveForms,events,WFU,UtU,pwave,swave=self._loadMPSubSpace(DFsta,self.ContinousDataLength,sta,channels,samplingRate,returnFull=True,PKS=staPksDf)
        for a in range(len(conrangeyear)): # for each year
            for b in conrangejulday[a]: # for each julian day   
                FilesToCorr=glob.glob(os.path.join(self.ConDir,sta,str(conrangeyear[a]),str(b),sta+'*'))
                for fileToCorr in FilesToCorr: #loop through each chunk of continous data
                    CorDF,MPcon,ConDat=self._getRA(ssArrayTD,ssArrayFD,fileToCorr,len(channels),reqlen,contrim,Names)
                    if not isinstance(CorDF,pd.DataFrame): #if something is broken skip hour                        
                        print '%s failed' % fileToCorr 
                        continue
                    for name,row in CorDF.iterrows(): # iterate through each subspace/single
                        if self.calcHist and len(CorDF)>0: 
                            try:
                                histdic[name]=histdic[name]+np.histogram(row.SSdetect,bins=self.hist['Bins'])[0]
                            except:
                                print 'binning failed'
                        if isinstance(self.UTCSaves,collections.Iterable):
                            self._makeUTCSaveDF(row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD)
                        if self._evalTriggerCondition(row,name,threshold): # Trigger Condition
                            #deb(row)
                            Sar=self._CreateCoeffArray(row,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD,WFU,UtU,staPksDf,pwave,swave)
                            if len(Sar)>300:
                                print 'over 300 events found in single continous data chunk, perphaps minCoef is too low?' 
                            if len(Sar)>0:
                                DF=DF.append(Sar,ignore_index=True)
                            if len(DF)>500:
                                detex.util.saveSQLite(DF,self.subspaceDB,'ss_df')
                                DF=pd.DataFrame()
                                numdets+=500
        if len(DF)>0:
            detex.util.saveSQLite(DF,self.subspaceDB,'ss_df')
        print ('%s completed, %d potential detection(s) recorded' %(sta,len(DF)+numdets))
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
        
    def _loadMPSubSpace(self,DFsta,ContinousDataLength,sta,channels,samplingRate,returnFull=False,PKS=None): # function to load subspace representations
        if 'UsedSVDKeys' in DFsta.columns: #test if input TRDF row is subspace
            ssArrayTD,rele,ssArrayFD,offsets,mags,eventWaveforms,eves,WFU,UtUdict,pwave,swave={},{},{},{},{},{},{},{},{},{},{} # intitialize empty dicts
            if not all ([isinstance(x.UsedSVDKeys,list) for num,x in DFsta.iterrows()]):
                raise Exception ('SVD not defined, run SVD on subspace stream class before calling detex')
            if not all([set(y)==set(DFsta.Channels[0].values()[0]) for x in DFsta.Channels for y in x.values()]): #make sure are channels are the same for all events
                raise Exception ('all stations in subspace do not have the same channels')
            Nc=len(channels) #num of channels
            for a in DFsta.iterrows():
                events=a[1].Events
                U=np.array([a[1].SVD[x] for x in a[1].UsedSVDKeys]) #used SVD basis
                ssArrayTD[a[1].Name]=U 
                UtU=np.dot(np.transpose(U),U)
                rele[a[1].Name]=int(ContinousDataLength*samplingRate*Nc+np.max(np.shape(ssArrayTD[a[1].Name])))
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
                pwave[a[1].Name],swave[a[1].Name]=self._getPandSWaves(PKS,a,events)
                mags[a[1].Name]=mag
                eves[a[1].Name]=events
                eventWaveforms[a[1].Name]=WFs
                WFU[a[1].Name]=np.dot(WFs,UtU) # events projected into subspace
                UtUdict[a[1].Name]=UtU
        
        if returnFull:
            return ssArrayTD,ssArrayFD,rele,offsets,mags,eventWaveforms,eves,WFU,UtUdict,pwave,swave
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
    def _CreateCoeffArray(self,CorSeries,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD,WFU,UtU,staPKsDf,pwave,swave):  
        #WFlen=np.shape(WFU[name])[1] #get length of window in samples
        dpv=0
        if self.trigCon==0:
            Ceval=CorSeries.SSdetect.copy()
        elif self.trigCon==1:
            Ceval=CorSeries.STALTA.copy()
        Sar=pd.DataFrame(columns=['SD','SD_STALTA','STMP','Name','Sta','MSTAMPmin','MSTAMPmax','Mag','SNR','ProEnMag'])
        count=0
        while Ceval.max()>=threshold[name]: 
            trigIndex=Ceval.argmax()
            coef=CorSeries.SSdetect[trigIndex]
            #times=times+[float(trigIndex)/sr+starttime]
            if self.extrapolateTimes:  #extrapolate times
                times=self._subsampleExtrapolate(Ceval,trigIndex,CorSeries.SampRate,CorSeries.TimeStamp)
                times1=float(trigIndex)/CorSeries.SampRate+CorSeries.TimeStamp
                if abs(times1-(times))>1.0/CorSeries.SampRate:
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
                mag,SNR=np.NaN,np.NaN
            
            if self.pks: #If useing picks
                pass
                #P,S=self._estimateArrivals(trigIndex,CorSeries,events[name],MPcon,pwave[name],swave[name],WFU[name])
            else:
                P,S=np.NaN,np.NaN
            if count>4000: #kill switch to prevent infinite loop (just in case)
                raise Exception (' _CreatCoeffArray loop exceeds limit of 4000 events in one continous data chunk')
            MSTAMPmax,MSTAMPmin=times-np.min(offsets[name]),times-np.max(offsets[name])
            if not MSTAMPmin>0:
                deb([CorSeries,name,threshold,sta,offsets,mags,eventWaveForms,MPcon,events,ssArrayTD,WFU,UtU,staPKsDf,pwave,swave])
            Sar.loc[count]=[coef,SLValue,times,name,sta,MSTAMPmin,MSTAMPmax,stdMag,SNR,ProEnmag]
            count+=1
        #deb(Sar)
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
                print CCs.max(),ind
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
        ConDat=MPcon[trigIndex*CorSeries.Nc:trigIndex*CorSeries.Nc+WFlen] #Continous data chunk that triggered  subspace
        ssCon=np.dot(UtU,ConDat) # Continous data chunk projected into subspace
        proEn=np.var(ssCon)/np.var(WFU,axis=1)
       # alphas=[_calcAlpha(x,ssCon) for x in WFU]
        
        if trigIndex*CorSeries.Nc>5*WFlen:
            rollingstd=pd.rolling_std(MPcon[trigIndex*CorSeries.Nc-5*WFlen:trigIndex*CorSeries.Nc],WFlen)[WFlen-1:]
        else:
            rollingstd=pd.rolling_std(MPcon[trigIndex*CorSeries.Nc:trigIndex*CorSeries.Nc+WFlen+6*WFlen],WFlen)[WFlen-1:]
        baseNoise=np.median(rollingstd)
        touse=mags>-10
        eventCors=np.array([fast_normcorr(x,ConDat)[0] for num,x in enumerate(eventWaveForms)]) #correlation coefs between each event and continous data

#        estMags=np.array([mags[x]+np.log10(alphas[x]) for x in range(len(mags))]) 
        projectedEnergyMags=np.sum([(mags[x]+np.log10(np.sqrt(proEn[x])))*np.square(eventCors[x]) for x in range(len(proEn)) if mags[x]>-10])/np.sum(np.square(eventCors[touse]))    
        stdMags=np.sum([(mags[x]+np.log10(np.std(ConDat)/np.std(eventWaveForms[x])))*np.square(eventCors[x]) for x in range(len(proEn)) if mags[x]>-10])/np.sum(np.square(eventCors[touse]))    
        SNR=np.std(ConDat)/baseNoise
#        mag=np.log10(alpha)+mags[bc]
        return projectedEnergyMags,stdMags,SNR
        
    def _subsampleExtrapolate(self,Ceval,trigIndex,sr,starttime) :
        """ Method to estimate subsample time delays using cosine-fit interpolation
        Cespedes, I., Huang, Y., Ophir, J. & Spratt, S. 
        Methods for estimation of sub-sample time delays of digitized echo signals. 
        Ultrason. Imaging 17, 142–171 (1995)"""
        ind=Ceval.argmax()
        if trigIndex != ind:
            raise Exception('something is messed up, trigIndex and CC.argmax no equal')
        if ind==0 or ind==len(Ceval)-1: # If max occurs at beg or end of CC set as beg or end, no extrapolation
            tau=float(ind)/sr + starttime
        else:
            alpha=np.arccos((Ceval[ind-1]+Ceval[ind+1])/(2*Ceval[ind]))
            tau=-(np.arctan((Ceval[ind-1]-Ceval[ind+1])/(2*Ceval[ind]*np.sin(alpha)))/alpha)*1.0/sr+ind*1.0/sr+starttime
            if -np.arctan((Ceval[ind-1]-Ceval[ind+1])/(2*Ceval[ind]*np.sin(alpha)))/alpha >1:
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
            trig= Corrow.MaxSD
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
    
    def _getContinousRanges(self,ConDir,sta,UTCstart,UTCend):
        # get lists of all hours and days for which continous data is avaliable
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
                #FilesToCorr is now a len(chans) by len(sharedHours) matrix populated with paths to continous waveforms
                FilesToCorr[a2][a3]=glob.glob(os.path.join(ConDir,sta,year,jday,chans[a2],'*'+'T'+sharedHours[a3]+'.sac'))
        return FilesToCorr, sharedHours
    
    def _getRA(self,ssArrayTD,ssArrayFD,fileToCorr,Nc,reqlen,contrim,Names):     
        CorDF=pd.DataFrame(index=Names,columns=['SSdetect','STALTA','TimeStamp','SampRate','MaxSD','MaxSTALTA','Nc','File'])
        CorDF['File']=os.path.basename(fileToCorr)
        try:
            conStream=_applyFilter(obspy.core.read(fileToCorr),self.filt,self.decimate,self.dtype,fillZeros=self.fillZeros)
        except ValueError:
            return None,None,None
        
        CorDF.SampRate=conStream[0].stats.sampling_rate
        MPcon,ConDat,TR=multiplex(conStream,Nc,returnlist=True,retTR=True)
        CorDF.TimeStamp=TR[0].stats.starttime.timestamp
        if isinstance(contrim,dict):
            ctrim=np.median(contrim.values())
        else:
            ctrim=contrim
        if ctrim<0: #Trim continous data to avoid overlap
            MPconcur=MPcon[:int(ctrim*CorDF.SampRate[0]*Nc)]
        else:
            MPconcur=MPcon
        MPconFD=scipy.fftpack.fft(MPcon,n=2**int(np.median(reqlen.values())).bit_length())
 
        
        for name,row in CorDF.iterrows(): #loop through each hour in this Jday and calculate statistic
            if len(MPcon)<=np.max(np.shape(ssArrayTD[name])): # make sure the template is shorter than continous data else skip
                MPcon=None            
            if isinstance(MPcon,np.ndarray): #If channels not equal in length and multiplexing fails delete hour, else continue with CCs                  
                ssd=self._MPXSD(MPconcur,reqlen[name],ssArrayTD[name],ssArrayFD[name],Nc,MPconFD)
#                try:                
#                    maxind=ssd.argmax()
#                except ValueError:
#                    return None,None,None
                CorDF.SSdetect[name]=ssd
                if len(ssd)<10:
                    return None,None,None
                CorDF.MaxSD[name]=ssd.max()
                if CorDF.MaxSD[name]>1.1: # If an infinity value occurs, zero it. 
                    ssd[np.isinf(ssd)] = 0
                    CorDF.SSdetect[name]=ssd
                    CorDF.MaxSD[name]=ssd.max()
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

    def _MPXSD(self,MPcon,reqlen,ssArrayTD,ssArrayFD,Nc,MPconFD): # multiplex subspace detection statistic function
#        MPconFD=scipy.fftpack.fft(MPcon,n=2**reqlen.bit_length())
        n = np.int32(np.shape(ssArrayTD)[1]) #length of each basis vector
        a = pd.rolling_mean(MPcon, n)[n-1:] #rolling mean of continous data
        b = pd.rolling_var(MPcon, n)[n-1:]  # rolling var of continous data
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

           