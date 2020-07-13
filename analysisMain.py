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
    pcols=range(32,58)
    mcols=range(58,83)
    
    target_func = 'Mean Cyc/Insn'
    lo = 0.0
    hi = 1.0

    pre1 = pp.PreProcess("../../Data/data-mca-pmu-v1-6-23.xlsx", target_func)
    pre1.setColumns(icols=range(2,5),pcols=None,mcols=mcols) 
    pre1.setLimits(lo, hi)
    X1,y1,c_names1,sblk_names1,indices1 = pre1.prepareData()

        
    pre2 = pp.PreProcess("../../Data/data-mca-pmu-v1-6-23.xlsx", target_func)
    pre2.setColumns(icols=range(2,5),pcols=pcols,mcols=None)
    pre2.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre2.prepareData()
    
    pre3 = pp.PreProcess("../../Data/data-mca-pmu-v1-6-23.xlsx", target_func)
    pre3.setColumns(icols=range(2,5),pcols=pcols,mcols=mcols)
    pre3.setLimits(lo, hi)
    X3,y3,c_names3,sblk_names3,indices3 = pre3.prepareData()
    
    #pre.draw_histogram(30)
    
    logTrans = False
    
    sp1 =sup.Supervised(X1,y1,c_names1,logTrans = logTrans)
    sp2 =sup.Supervised(X2,y2,c_names2,logTrans = logTrans)
    sp3 =sup.Supervised(X3,y3,c_names3,logTrans = logTrans)
    
    #sp.correlationAnalysis(k=0)
    #sp.plotCorrelations()
    #sp.linearRegressionsModels()
    #sp.plot_scatter()

    print()
    print("MCA Only")
    sp1.randomForestRegressionModels(500,6)
    print()
    print("PMU Only")
    sp2.randomForestRegressionModels(500,6)
    print()
    print("MCA + PMU")
    sp3.randomForestRegressionModels(500,6)    
    #sp.plot_scatter()
    
    
    


