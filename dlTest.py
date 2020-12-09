import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import dlModels as dlm
import sys

if __name__ == "__main__":
    
    # PMU / MCA stuff
    pcols=range(42,61)
    mcols=range(65,90)
    
    target_func = 'Mean Cyc/Insn'
    lo = 0.0
    hi = 200.0

    pre = pp.PreProcess("./data/TimedLBR_SuperBlocks-pmu-mca-v2-winnow.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols,mcols=mcols) 
    pre.setLimits(lo, hi)
    X,y,c_names,sblk_names,indices = pre.prepareData()
    logTrans = False
    classes,thresh = pre.createClassificationData(85)
    
    dl =dlm.dlModels(X,classes,c_names,logTrans = logTrans)
    #dl.buildFCNRegressor()
    dl.buildFCNClassifier()
    
    
    #sp =sup.Supervised(X,y,c_names,logTrans = logTrans)
    #sp.randomForestRegressionModels(1000,8)


