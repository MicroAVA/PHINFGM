import argparse
import numpy as np
import collections
#-----------------------------------------

def parse_args():

	parser = argparse.ArgumentParser(description="Run VHIs code")
	parser.add_argument('--data', type=str, default='312_747',  help='choose one of the datasets 728_129 32_119 312_747 1380_221')

	return parser.parse_args()
#-----------------------------------------

def tree():
    return collections.defaultdict(tree)
#-----------------------------------------

def get_host_virus_names(HV):
	# remove the host and virus names from the matrix
	VHIs = np.zeros((HV.shape[0]-1,HV.shape[1]-1))

	host = []
	virus = []
	for i in range(1,HV.shape[0]):
	    for j in range(1,HV.shape[1]):
	        virus.append(HV[i][0])
	        host.append(HV[0][j])
	        VHIs[i-1][j-1] = HV[i][j]

	# to remove duplicate elements       
	virus = sorted(list(set(virus)))
	host = sorted(list(set(host)))
	VHIs = np.array(VHIs, dtype=np.float64)

	print('Number of host:',len(host))
	print('Number of virus:', len(virus))

	return host, virus, VHIs
#-------------------------------------------------------------------------

def built_multiple_similarity_matrix(sim_files,Dtype, data, length ):
    
    SimF = np.loadtxt(sim_files, delimiter='\n',dtype=str ,skiprows=0)
    Sim = []
    for i in range(0,len(SimF)):
        simMat = 'Input/'+data+'/'+Dtype+'sim/'+str(SimF[i]) 
        Sim.append(np.loadtxt(simMat, delimiter='\t',dtype=np.float64,skiprows=1,usecols=range(1,length+1)))
        
    return Sim
#---------------------------------------------------------------------------

def load_datasets(data):
	
	# read the interaction matrix
	HostVirusF = "Input/"+data+"/"+data+"_admat_dgc.txt"
	HostVirus = np.genfromtxt(HostVirusF, delimiter='\t',dtype=str)

	# get all host and virus names with order preserving
	all_host, all_virus, VHIs = get_host_virus_names(HostVirus)

	# read all files of similarties
	Vsim_files = 'Input/'+data+'/Vsim/selected_Vsim_files.txt'
	Hsim_files  = 'Input/'+data+'/Hsim/selected_Hsim_files.txt'

	# built the similarity matrices of multiple similarities
	H_sim = built_multiple_similarity_matrix(Hsim_files, 'H', data, len(all_host))
	V_sim = built_multiple_similarity_matrix(Vsim_files, 'V', data, len(all_virus))

	## Create R (host, virus, label) with known and unknown interaction
	R = tree()

	# Get all postive host virus interaction R
	with open('Input/'+data+'/R_'+data+'.txt','r') as f:
	    for lines in f:
	        line = lines.split()
	        line[0]= line[0].replace(":","")
	        R[line[1]][line[0]] = 1
	#######################################################################
	#build the BIG R with all possible pairs and assign labels
	label = []
	pairX = []
	for d in all_host:
		for t in all_virus:
			p = d, t
            # add negative label to non exit pair in R file
			if R[d][t] != 1:
				R[d][t] = 0
				l = 0
			else:
				l = 1

			label.append(l)
			pairX.append(p)

    # prepare X = all (hr, vr) pairs, Y = labels
	X = np.asarray(pairX)
	Y = np.asarray(label)
	print('dimensions of all pairs', X.shape)

	return all_host, all_virus, H_sim, V_sim, VHIs, R, X, Y
#----------------------------------------------------------------------------------------
