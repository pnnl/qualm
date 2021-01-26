import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

    pre = pp.PreProcess("./data/dataset-small-unique-winnow.xlsx", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols1,mcols=mcols1)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    sp1 = sup.Supervised(X2,y2,c_names2)
    feat1,correl1 = sp1.correlationAnalysis(k=15)
    
    
    pre = pp.PreProcess("./data/dataset-large-unique-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols2,mcols=mcols2)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    sp2 = sup.Supervised(X2,y2,c_names2)
    feat2,correl2 =sp2.correlationAnalysis(k=15)    
    
   
    pre = pp.PreProcess("./data/dataset-large-duplicate-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols3,mcols=mcols3)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    sp3 = sup.Supervised(X2,y2,c_names2)
    feat3,correl3 =sp3.correlationAnalysis(k=15)
    
    
    df = pd.DataFrame(list(zip(feat1,correl1,feat2,correl2,feat3,correl3)), columns=['Ft-sm-u','Cl-sm-u','Ft-lg-u','Cl-lg-u','Ft-lg-d','Cl-lg-d'])
    df.to_csv("correls.csv",index=False)