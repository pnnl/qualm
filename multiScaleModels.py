import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
from time import time

if __name__ == "__main__":
    
    start = time()
    
    #pre = pp.PreProcess("../../Data/data-v2-pmu-03-10-parsed.xlsx", 'Xtra Cyc/Insn')
    #pre.setColumns(range(2,5),range(32,175))
    
    #pre = pp.PreProcess("../../Data/data-v2-pmu-04-21-parsed.xlsx", 'Xtra Cyc/Insn')
    #pre.setColumns(range(2,5),range(32,58))

    #pre.setLimits(0.0, 50.00)
    
    # MCA stuff
    pre = pp.PreProcess("../../Data/data-mca-pmu-v1-6-23.xlsx", 'Mean Cyc/Insn')
    pre.setColumns(range(2,5),range(58,83))
    
    X,y,c_names,sblk_names,indices = pre.prepareData()
    #pre.plotClassImbalance()
    
    classes = pre.createClassificationData(90)
        
    sp = sup.Supervised(X,y,c_names,logTrans = True, cls=classes)
    sp.setParamsMultiStage(0.2,0.5,0.2,0.2)
    sp.multiStageClassReg()

    print("Total time taken : ", time()-start)
    
    
    


