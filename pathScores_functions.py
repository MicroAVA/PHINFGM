
import numpy as np

######################### MetaPath Functions ################################

# This function is used for path score of length 4 to add HH or VV matrix multiplication
def HHH_VVV_sim(simM):
    
    np.fill_diagonal(simM,0)
    m = np.einsum('ij,jk->ijk', simM, simM)
    
    sumM = np.sum((m[:, :, None] ), axis = 1)
    maxM = np.max((m[:, :, None] ), axis = 1)
    #avgM = np.mean((m[:, :, None] ), axis = 1)
    
    sumM = np.squeeze(sumM)
    maxM = np.squeeze(maxM)
    #avgM = np.squeeze(avgM)
               
    return (sumM,maxM)#,avgM)
#----------------------------------------------------------

# host similarity matrix * training VHIs matrix
def metaPath_Hsim_HV(Hsim,HV,length, mul=False):
    
    np.fill_diagonal(Hsim,0)
    m = np.einsum('ij,jk->ijk', Hsim, HV)

    if(mul):
        m = m**(length)

    sumM = np.sum((m[:, :, None] ), axis = 1)
    maxM = np.max((m[:, :, None]), axis = 1)
    #avgM = np.mean((m[:, :, None] ), axis = 1)

    # to convert from 3-d matrix to 2-d matrix
    sumM = np.squeeze(sumM)
    maxM = np.squeeze(maxM)
    #avgM = np.squeeze(avgM)

    return (sumM,maxM)#,avgM)
#------------------------------------------------------------

#  Training VHIs matrix * virus similarity matrix
def metaPath_HV_Vsim(Vsim,HV, length, mul=False):

    np.fill_diagonal(Vsim,0)
    
    m = np.einsum('ij,jk->ijk', HV,Vsim)

    if(mul):
        m = m**(length)

    sumM = np.sum((m[:, :, None]), axis = 1)
    maxM = np.max((m[:, :, None] ), axis = 1)
    #avgM = np.mean((m[:, :, None] ), axis = 1)
    

    sumM = np.squeeze(sumM)
    maxM = np.squeeze(maxM)
    #avgM = np.squeeze(avgM) 

    return (sumM,maxM)#, avgM)
#-------------------------------------------------------------

def metaPath_HHVV(HV,Hsim,Vsim, mul=False):

    sumHHV,maxHHV = metaPath_Hsim_HV(Hsim,HV, 3,mul)
    sumHHVV,_ = metaPath_HV_Vsim(Vsim,sumHHV,3,mul)
    _,maxHHVV = metaPath_HV_Vsim(Vsim,maxHHV,3,mul)
    # _,_,avgDDTT = metaPath_DT_Tsim(Tsim,avgDDT,3)
    
    return sumHHVV,maxHHVV
#-------------------------------------------------------------

def metaPath_HVHV(HV):
    
    VH = np.transpose(HV)
    HH = HV.dot(VH)

    sumM, maxM = metaPath_Hsim_HV(HH,HV,3)
    
    return sumM,maxM
#----------------------------------------------------------------

def metaPath_HVVHV(HV,Vsim,mul=False):
    VH = np.transpose(HV)
    VH_HV = VH.dot(HV)

    sumVHV, maxVHV = metaPath_Hsim_HV(Vsim, VH_HV,4,mul)
    sumHVVHV,_= metaPath_HV_Vsim(sumVHV, HV, 4,mul)
    _,maxHVVHV=metaPath_HV_Vsim(maxVHV, HV, 4,mul)
    return sumHVVHV, maxHVVHV

def metaPath_HVHHV(HV,Hsim,mul=False):
    VH = np.transpose(HV)
    HV_VH = HV.dot(VH)

    sumHVH, maxHVH = metaPath_HV_Vsim(Hsim, HV_VH, 4,mul)
    sumHVHHV, _ = metaPath_Hsim_HV(sumHVH, HV, 4,mul)
    _, maxHVHHV = metaPath_Hsim_HV(maxHVH, HV, 4,mul)
    return sumHVHHV, maxHVHHV
