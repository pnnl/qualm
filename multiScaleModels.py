import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
from time import time

if __name__ == "__main__":
    
    start = time()
    
    # PMU / MCA stuff
    pcols=range(42,61)
    mcols=range(65,90)
    
    target_func = 'Mean Cyc/Insn'

    pre = pp.PreProcess("./data/TimedLBR_SuperBlocks-pmu-mca-v2-winnow.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols,mcols=mcols) 
    #pre.setLimits(0.0, 50.00)
    
    X,y,c_names,sblk_names,indices = pre.prepareData()
    #pre.plotClassImbalance()
    
    classes = pre.createClassificationData(85)
        
    sp = sup.Supervised(X,y,c_names,logTrans = False, cls=classes)
    sp.setParamsMultiStage(0.2,0.5,0.2,0.2)

    #sp.classification_full_data()
    sp.multiStageClassReg()

    print("Total time taken : ", time()-start)
    
    
    


