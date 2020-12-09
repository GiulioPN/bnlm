import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import bnplm as bm


def getDataFolder():
	'''Helper function: return the path of the data folder'''
	data_path = os.getcwd()+"/data/"
	return data_path

def getOutputFolder():
	'''Helper function: return the path of the output folder'''
	output_path = os.getcwd()+"/output/"
	return output_path

def simulationExample():
	'''A simple test for the hpyp model, in Train-Test mode, 100 samples and all parameters setted to 1.'''
	print("Start simple test with 100 samples and parameters 1,1,1,1")
	train_data = getDataFolder()+"train.txt"
	test_data = getDataFolder()+"test.txt"
	estimations = getOutputFolder() + "estimations_simulationExample.txt"
	perfomance = getOutputFolder() + "perfomance_simulationExample.txt"

	bm.hpypTT(train_data,test_data, 100, 1,1,1,1, estimations, perfomance)

def hpypTT(train_data_file, test_data_file ,samples =100, da = 1, db = 1, sr = 1, ss = 1):
	'''Train test model'''
	print("Start alghorithm with "+str(samples)+" samples and parameters: "+str(da)+","+str(db)+","+str(sr)+","+str(ss))
	train_data = getDataFolder()+train_data_file
	test_data = getDataFolder()+test_data_file
	estimations = getOutputFolder() + "estimations_samples"+str(samples)+"_da"+str(da)+"_db"+str(db)+"_sr"+str(sr)+"_ss"+str(ss)+".txt"
	perfomance = getOutputFolder() + "perfomance_samples"+str(samples)+"_da"+str(da)+"_db"+str(db)+"_sr"+str(sr)+"_ss"+str(ss)+".txt"

	bm.hpypTT(train_data, test_data, samples, da, db, sr, ss, estimations, perfomance)


