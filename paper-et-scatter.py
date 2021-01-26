import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import sys

if __name__ == "__main__":
    

    # PMU / MCA stuff
    pcols1=range(42,61)
    mcols1=range(65,90)
    
    pcols2 = range(9,28)
    mcols2 = range(30,55)
    
    pcols3 = range(9,28)
    mcols3 = range(30,55)
    
    target_func = 'Xtra Cyc/Insn'
    
    lo = 0.0
    hi = 200.0
    
    logTrans = False
    

    print("Small unique ")
    print("\n")
    pre = pp.PreProcess("./data/dataset-small-unique-winnow.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols1,mcols=None)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    sp1 = sup.Supervised(X2,y2,c_names2,logTrans=logTrans)
    sp1.ETRegressionModels(500,6)
    

    
    


