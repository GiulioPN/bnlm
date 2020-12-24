import os
import sys
sys.path.append('../')
import utils as utl
os.chdir('../')

if __name__ == "__main__":
    utl.hpypTT("brown09.txt", "brown01.txt", 100, 1, 1, 1, 1, "brownRaw")
    utl.hpypTT("brown09-postprocess.txt", "brown01-postprocess.txt", 100, 1, 1, 1, 1,"brownPostprcss")