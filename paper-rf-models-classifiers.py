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
    large = 200.0
    percentile = 90
    B = 1
        
    
    print("Small unique ")
    print("\n")
    pre = pp.PreProcess("./data/dataset-small-unique-winnow.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols1,mcols=mcols1)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    classes1, thresh = pre.createClassificationData(percentile)
    print("Threshold = ",thresh)
    sp1 = sup.Supervised(X2,y2,c_names2,cls=classes1)
    sp1.rf_models_CV_class()
    
    
    print("Large unique ")
    print("\n")    
    pre = pp.PreProcess("./data/dataset-large-unique-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols2,mcols=mcols2)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    classes2, thresh = pre.createClassificationData(percentile)
    print("Threshold = ",thresh)
    sp2 = sup.Supervised(X2,y2,c_names2,cls=classes2)
    sp2.rf_models_CV_class()
    
    
    
    print("Large duplicates ")
    print("\n")      
    pre = pp.PreProcess("./data/dataset-large-duplicate-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols3,mcols=mcols3)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    classes3, thresh = pre.createClassificationData(percentile)
    print("Threshold = ",thresh)
    sp3 = sup.Supervised(X2,y2,c_names2,cls=classes3)
    sp3.rf_models_CV_class()
    
    


