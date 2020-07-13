The repo consists of three main classes (and the associated methods) and example driver scripts 
to perform analysis.

a. preproces.py : Contains all the methods related to data pre-processing, plotting distributions etc.
b. supervised.py : Contains all the methods related to supervised learning (regression, classification)
c. unsupervised.py : Contains all the methods related to un-supervised learning (PCA, clustering etc.)

The main driver scripts utilized thus far are:

a. analysisMain.py : Used for a typical analyusis pipeline
b. multiScaleModels.py : Used to execute the proposed multi-stage model

The data excel file, the associated target function, the range of columns for the features 
are all typically included as part of the driver scripts and are usually self explanatory.
Currently the scripts support both MCA and PMU features and options exist to model either or both.
Based on the flags, appropriate normalizations are performed. 
       
