import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import dlModels as dlm
import time
import sys

if __name__ == "__main__":
    
    # PMU / MCA stuff

    
    target_func = 'Xtra Cyc/Insn'
    lo = 0.0
    hi = 40.0

    start = time.time()
    pre = pp.PreProcess("./data/TimedLBR_SuperBlocks-pmu-mca-v2-winnow.xlsx", target_func)
    #pre = pp.PreProcess("./data/data-mca-pmu-v1-6-23.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols,mcols=None) 
    pre.setLimits(lo, hi)
    X,y,c_names,sblk_names,indices = pre.prepareData()
    logTrans = False
    
    un = usup.UnSupervised(X, y, c_names)
    un.featureClustering()
    end = time.time()
    print("Total time taken = ",end-start)