import numpy as np

#---------------------------------------------------------------------------
def get_FV_host_virus(foldCounter,allV,allH, data):
    # Working with feature vector
    virus ={}
    host ={}
    fileName = 'EMBED/'+data+'/EmbeddingFold_'+str(foldCounter)+'.txt'

    ## ReadDT feature vectore that came after applying n2v on allGraph including just R_train part
    with open(fileName,'r') as f:
        #line =f.readline()# to get rid of the sizes
        for line in f:
            line = line.split()
            line[0]= line[0].replace(":","")
            
            key = line[0]
            
            line.pop(0)
            if key in allV:
                virus[key] = line
            else:
            #key in allH and its feature:
                host[key] = line
                
    ### Create FV for host and for virus
    FV_host = []
    FV_virus = []

    for t in allV:
        FV_virus.append(virus[t])

    for d in allH:
        FV_host.append(host[d])

    # host node2vec FV, and virus node2vec FV
    FV_virus = np.array(FV_virus, dtype = float)
    FV_host = np.array(FV_host, dtype = float)
    
    return FV_virus, FV_host
#------------------------------------------------------------------------------------
