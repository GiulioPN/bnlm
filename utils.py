import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import bnplm as bm


def hpypSimulationTWT(samples =500, burnin=400, da = 1, db = 1, sr = 1, ss = 1, lambda_in=0.7, nSim=1):
	'''Train test model'''
	print("Start alghorithm with "+str(samples)+" samples and parameters: "+str(da)+","+str(db)+","+str(sr)+","+str(ss))
	train_corpus = "/vagrant/tests/data/triello2/corpus-TWT_train.txt"
	test_corpus = "/vagrant/tests/data/triello2/corpus-TWT_test.txt"
	bm.hpypTrainTest(train_corpus, test_corpus, samples, burnin, da, db, sr, ss, nSim)

def shpypSimulationTWT(samples =10000, burnin=9000, da = 1, db = 1, sr = 1, ss = 1, lambda_in=0.7, nSim=1):
	'''Train test model'''
	print("Start alghorithm with "+str(samples)+"/"+str(burnin)+" lambda: "+str(lambda_in)+" samples and parameters: "+str(da)+","+str(db)+","+str(sr)+","+str(ss))
	train_corpus1 = "/vagrant/tests/data/triello2/corpus1-tweets_train.txt"
	train_corpus2 = "/vagrant/tests/data/triello2/corpus2-tvshows_train.txt"
	train_corpus3 = "/vagrant/tests/data/triello2/corpus3-wikipedia_train.txt"
	train_corpus = []
	train_corpus.append(train_corpus1)
	train_corpus.append(train_corpus2)
	train_corpus.append(train_corpus3)

	test_corpus1 = "/vagrant/tests/data/triello2/corpus1-tweets_test.txt"
	test_corpus2 = "/vagrant/tests/data/triello2/corpus2-tvshows_test.txt"
	test_corpus3 = "/vagrant/tests/data/triello2/corpus3-wikipedia_test.txt"
	test_corpus = []
	test_corpus.append(test_corpus1)
	test_corpus.append(test_corpus2)
	test_corpus.append(test_corpus3)

	bm.shpypTrainTest(train_corpus, test_corpus, samples, burnin, da, db, sr, ss, lambda_in, nSim)

def shpypSimulation(train_corpus, test_corpus, samples =10000, burnin=9000, da = 1, db = 1, sr = 1, ss = 1, lambda_in=0.7, nSim=1):
	'''Train test model'''
	bm.shpypTrainTest(train_corpus, test_corpus, samples, burnin, da, db, sr, ss, lambda_in, nSim)






