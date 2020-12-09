import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import sys

if __name__ == "__main__":
    
    #pre = pp.PreProcess("../../Data/data-v2-pmu-03-10-parsed.xlsx", 'Xtra Cyc/Insn')
    #pre.setColumns(range(2,5),range(32,175))
    
    #pre = pp.PreProcess("../../Data/data-v2-pmu-04-21-parsed.xlsx", 'Xtra Cyc/Insn')
    #pre.setColumns(range(2,5),range(32,58))

    # PMU / MCA stuff
    pcols=range(42,61)
    mcols=range(65,90)
    
    target_func = 'Xtra Cyc/Insn'
    lo = 0.0
    hi = 200.0

    pre = pp.PreProcess("./data/TimedLBR_SuperBlocks-pmu-mca-v2-winnow.xlsx", target_func, skipRows=0)
    pre.setColumns(icols=range(2,5),pcols=pcols,mcols=mcols)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    classes, thresh = pre.createClassificationData(90)    

    sp = sup.Supervised(X2,y2,c_names2,logTrans = False,cls=classes)
    #sp.roc_auc()
    #sp.prec_recall()
    #sp.bias_variance_decomp_regression()
    sp.run_bias_variance_tradeoff_regression()
    
    
    
    


