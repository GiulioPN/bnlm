import os
import sys
sys.path.append('../')
import utils as utl
os.chdir('../')

if __name__ == "__main__":
    utl.hpypTT("all-train.txt", "sotu_test.txt", 100, 1, 1, 1, 1)