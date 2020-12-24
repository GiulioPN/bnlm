import os
import sys
sys.path.append('../')
import utils as utl
os.chdir('../')

if __name__ == "__main__":
	utl.hpypTT("all-train-postprocessWithLemNoSW.txt", "sotu_test2-postprocessWithLemNoSW.txt", 300, 1, 1, 1, 1,"all-train_SotuTestPostprcssWithLemNoSW")