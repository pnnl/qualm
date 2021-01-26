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
    logTrans = False
    large = 200.0
    percentile = 90
    B = 1
        
    
    print("Small unique ")
    print("\n")
    pre = pp.PreProcess("./data/dataset-small-unique-winnow.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols1,mcols=None)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    classes, thresh = pre.createClassificationData(percentile)
    print("Threshold = ",thresh)
        
    pre = pp.PreProcess("./data/dataset-small-unique-winnow.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols1,mcols=None)    
    if (B == 0):
        pre.setLimits(0.0,thresh)
    else:
        pre.setLimits(thresh,large)
        logTrans = True    
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    sp1 = sup.Supervised(X2,y2,c_names2,logTrans=logTrans)
    sp1.regression_allRFModels()
    
    
    logTrans = False
    print("Large unique ")
    print("\n")    
    pre = pp.PreProcess("./data/dataset-large-unique-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols2,mcols=None)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    classes, thresh = pre.createClassificationData(percentile)
    print("Threshold = ",thresh)

    pre = pp.PreProcess("./data/dataset-large-unique-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols2,mcols=None)    
    if (B == 0):
        pre.setLimits(0.0,thresh)
    else:
        pre.setLimits(thresh,large)    
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    sp2 = sup.Supervised(X2,y2,c_names2,logTrans=logTrans)
    sp2.regression_allRFModels()
    
    
    
    print("Large duplicates ")
    print("\n")      
    pre = pp.PreProcess("./data/dataset-large-duplicate-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols3,mcols=None)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    classes, thresh = pre.createClassificationData(percentile)
    print("Threshold = ",thresh)

    pre = pp.PreProcess("./data/dataset-large-duplicate-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols3,mcols=None)
    if (B == 0):
        pre.setLimits(0.0,thresh)
    else:
        pre.setLimits(thresh,large)    
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    sp3 = sup.Supervised(X2,y2,c_names2,logTrans=logTrans)
    sp3.regression_allRFModels()
    
    


