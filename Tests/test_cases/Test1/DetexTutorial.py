
# coding: utf-8

"""
The test control file for test2 (the tutorial data set)
"""
def main():
    import detex #import detex module
    import numpy as np
    detex.getdata.getAllData() #download all data from iris
    
    cl = detex.subspace.createCluster()
    
    cl.dendro() #create a dendrogram to visualize grouping structure on each station
    
    cl.updateReqCC(.55)
    cl.dendro()
    
    cl['TA.M17A'].updateReqCC(.38) # set required correlation coef. for only station TA.M17A
    cl['TA.M17A'].dendro() # visualize grouping
    
    cl.simMatrix()
    
    ss= detex.subspace.createSubSpace() 
    
    ss.attachPickTimes()
    
    ss.SVD()
    
    ss.detex(useSingles=True) # run subspace detections and also run unclustered events as 1D subspaces (IE waveform correlation)
    
    import matplotlib.pyplot as plt #import matplotlib for visualization
    import json # used to convert string arrays as loaded from sql to numpy arrays
    
    hist=detex.util.loadSQLite('SubSpace.db','ss_hist') # load the ss_hist table of the SubSpace.db sqlite database
    
    for ind,row in hist.iterrows(): #loop over datarame
        hist.loc[ind,'Value']=np.array(json.loads(row.Value)) #convert string arrays into numpy arrays 
    
    avbins=(hist.iloc[0].Value[:-1]+hist.iloc[0].Value[1:])/2.0 # middle of bin values for histograms
    
    
    ## Plot each histogram
    for ind, row in hist.iterrows(): # loop again through dataframe
        if ind==0: # skip if index of dataframe is 0 (these are the bin values)
            continue 
        plt.plot(avbins,row.Value,label=row.Sta+':'+row.Name) #plot
    plt.xlabel('Detection Statistic') #label x
    plt.ylabel('Occurrence Rate') # label y
    plt.title('Detection Statistic') # lable title
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5)) #add legend outside of plot
    plt.semilogy() #use semilog on y axis
    plt.xlim([0,.5]) #set x lim
    
    plt.show() #show plot
        
    
    reload(detex.results)
    res=detex.results.detResults(requiredNumStations=2,veriBuffer=60*10,veriFile='veriFile.csv') 
    
    res.Dets
    
    res.Autos
    
    res.Vers
    
    import pandas as pd
    log=pd.read_csv('veriFile.csv')
    log
    
    import detex
    res=detex.results.detResults(requiredNumStations=1,veriBuffer=60*10,veriFile='veriFile.csv')
    res
    
    res.writeDetections(eventDir='DetectedEvents',updateTemKey=False)

