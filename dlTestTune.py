import numpy as np
import supervised as sup
import unsupervised as usup
import preprocess as pp
import dlModels as dlm
import sys

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

def TuneCLassifier(config):
    
    f1_ret = dl.buildFCNClassifierForTuning(config["lr"],config["dropout"])
    tune.report(f1_score=f1_ret)
    
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
    labels,thresh = pre.createClassificationData(85)
    dl = dlm.dlModels(X,labels,c_names,logTrans = logTrans)
        
    sched = AsyncHyperBandScheduler()
    analysis = tune.run(
        TuneCLassifier,
        metric="f1_score",
        mode="max",
        name="exp",
        scheduler=sched,
        stop={
            "f1_score": 0.85,
            "training_iteration": 100
        },
        resources_per_trial={"cpu": 1},
        num_samples=100,
        config={
            "lr": tune.loguniform(1e-4, 1e-2),
            "dropout": tune.uniform(0.2, 0.8),
        })
    
    print("Best config is:", analysis.best_config)
        
    
    


