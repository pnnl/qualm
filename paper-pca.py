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
    usup1 = usup.UnSupervised(X2,y2,c_names2)
    s1,pc1 = usup1.doPCA(X2)
    
    
    pre = pp.PreProcess("./data/dataset-large-unique-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols2,mcols=mcols2)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    usup2 = usup.UnSupervised(X2,y2,c_names2)
    s2,pc2 = usup2.doPCA(X2)
    
   
    pre = pp.PreProcess("./data/dataset-large-duplicate-winnow.csv", target_func)
    pre.setColumns(icols=range(2,5),pcols=pcols3,mcols=mcols3)
    pre.setLimits(lo, hi)
    X2,y2,c_names2,sblk_names2,indices2 = pre.prepareData()
    usup3 = usup.UnSupervised(X2,y2,c_names2)
    s3,pc3 = usup3.doPCA(X2)
    
    print(s1,s2,s3)
    
    plt.figure()
    plt.plot(np.linspace(1,len(s1),len(s1),dtype=int),100.*s1,color = 'r', linewidth=4, marker = 'o', mS = 8, label = 'SM-UNI')
    plt.plot(np.linspace(1,len(s2),len(s2),dtype=int),100.*s2,color = 'b', linewidth=4, marker = '+', mS = 8, label = 'LG-UNI')
    plt.plot(np.linspace(1,len(s3),len(s3),dtype=int),100.*s3,color = 'g', linewidth=4, marker = '*', mS = 8, label = 'LG-DUP')
    plt.grid()
    #plt.title('Variance explained by the principal components',fontsize = 25)
    plt.xlabel('Number of principal components',fontsize=36)
    plt.ylabel('Percentage variance explained',fontsize=36)
    ax=plt.gca()
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24) 
    plt.legend(fontsize=32)
    #plt.show() 
    
