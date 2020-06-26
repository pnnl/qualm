import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from time import time
from collections import defaultdict

import sys

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import mutual_info_regression

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor

import sklearn.metrics
from sklearn.metrics import r2_score,mean_squared_error,confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.model_selection import cross_validate


from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

class Supervised():
    
    
    def __init__(self,X,y,c,logTrans = False, cls = None):
        
        self.X = MinMaxScaler().fit_transform(X)
        self.colNames = c
        self.logTransform = logTrans
        self.classes = cls
        
        if(self.logTransform == True):
            self.y = np.log(1.00+y)
        else:
            self.y = y
            
        self.alpha = 0.2
        self.beta = 0.5
        self.gamma = 0.2
        self.delta = 0.2
    
    
    def correlationAnalysis(self,cor_type = 'Corr', k = 20):
          

        imp_order = None
        
        if (cor_type == 'Corr'):
            
            df = pd.DataFrame(self.X)
            df["y"] = self.y
            cor = df.corr()
            corrs_raw = abs(cor["y"])
            corrs_raw = corrs_raw[:-1]
            #colNames = colNames[~np.isnan(corrs_raw)]
            #corrs_raw = corrs_raw[~np.isnan(corrs_raw)]
            self.corrs = np.sort(corrs_raw.values)[::-1]
            #Remove the NaNs
    
            print("Correlation ordering : ")
            imp_order = (np.argsort(corrs_raw.values)[::-1])
            
            for idx in range(15):
                print(self.colNames[imp_order[idx]]," : ",round(self.corrs[idx],3))
                
        elif (cor_type == "MI"):
            mi_raw = mutual_info_regression(self.X,self.y)
            self.corrs = np.sort(mi_raw)[::-1]
        
     
        #Do filtering if needed
        Xred = None
        if (k > 0):
            
            print("Doing filtering based on correlations.")
            if ( k > self.X.shape[1]):
                sys.exit("Reduced number of features must be less than total number of features.")
            else:
                imp_ord_red = imp_order[:k]
                Xred = self.X[:,imp_ord_red]
                
        return Xred
    
    def plotCorrelations(self):
        
        plt.figure()
        plt.plot(np.linspace(1,len(self.corrs),len(self.corrs),dtype=int),self.corrs,color='r',marker = 'o', mS = 4)
        plt.grid()
        plt.title('Correlation (Linear) | Features and Target',fontsize = 25)
        plt.xlabel('Feature Index (sorted)',fontsize=20)
        plt.ylabel('Correlation (Linear) value',fontsize=20)
        ax=plt.gca()
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)     
        plt.show()
            

    def plot2DSlices(self):
        
        plt.figure()
        ax = plt.gca()
        #nFeat = X.shape[1]
        nFeat = 4
        for idx in range(4,nFeat+4):
            plt.subplot(2,2,idx-3)
            plt.scatter(self.X[:,idx-1],self.y)
            plt.xlabel(self.colNames[idx-1],fontsize = 22)
            plt.ylabel('Target value',fontsize = 22)
            plt.grid()
        
        plt.suptitle("Target vs normalized features.", fontsize = 24)    
        plt.show()        
            
    def linearRegressionsModels(self):

        #y = transform_target(y)
        
        #Split into training validation and test
        x_train, x_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=1)
        
        #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)    
    
        #Train the Linear regression models on training set
        reg_ols = LinearRegression().fit(x_train, y_train)
        reg_lasso = Lasso(alpha=0.1).fit(x_train, y_train)
        reg_ridge = Ridge().fit(x_train, y_train)
        
        #Make all 3 predictions
        y_pred_val_ols = reg_ols.predict(x_val)
        y_pred_val_ridge = reg_ridge.predict(x_val)
        y_pred_val_lasso = reg_lasso.predict(x_val)
        
        #MSE for test/validation set
        
        self.ytrue = y_val
        self.ypred = y_pred_val_ridge        
        
        if(self.logTransform == True):
            
            y_val = np.exp(y_val)-1.00
            y_pred_val_ols = np.exp(y_pred_val_ols)-1.00    
            y_pred_val_ridge = np.exp(y_pred_val_ridge)-1.00
            y_pred_val_lasso = np.exp(y_pred_val_lasso)-1.00
            
            self.ytrue = y_val
            self.ypred = y_pred_val_ridge
        
        print("R2 on validation set for OLS : ", r2_score(y_val,y_pred_val_ols))
        print("R2 on validation set for Ridge : ", r2_score(y_val,y_pred_val_ridge))
        print("R2 on validation set for Lasso : ", r2_score(y_val,y_pred_val_lasso))
            
        print("RMS error OLS : ", np.sqrt(mean_squared_error(y_val, y_pred_val_ols)))
        print("RMS error Ridge : ", np.sqrt(mean_squared_error(y_val, y_pred_val_ridge)))
        print("RMS error Lasso : ", np.sqrt(mean_squared_error(y_val, y_pred_val_lasso)))
    
    def plot_scatter(self):
        
        plt.figure()
        plt.scatter(self.ytrue,self.ypred)
        xy = np.linspace(np.min(self.ytrue),np.max(self.ytrue),100)
        plt.plot(xy,xy, linewidth = 3, linestyle = '--', color = 'k')
        plt.xlabel('Target value (true)',fontsize = 22)
        plt.ylabel('Target value (predicted)',fontsize = 22) 
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)    
        plt.grid()
        plt.show()
    
    def randomForestRegressionModels(self,nTrees,dDepth):
       
        
        print("Number of examples / features overall : ", self.X.shape)

        #Split into training validation and test
        x_train, x_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.25, random_state=0)
        
        #Train the RF regression model on training set
        rf_reg = RandomForestRegressor(n_estimators=nTrees, max_depth=dDepth, random_state=0)
        print("Fitting the RF model")
        rf_reg.fit(x_train, y_train)
        #print(rf_reg.feature_importances_)    
        y_pred_val_rf = rf_reg.predict(x_val)
        
        self.ytrue = y_val
        self.ypred = y_pred_val_rf
        
        if (self.logTransform == True):
            y_val = np.exp(y_val)-1.00
            y_pred_val_rf = np.exp(y_pred_val_rf)-1.00
            
            self.ytrue = y_val
            self.ypred = y_pred_val_rf            
       
        print("R2 for random forest model : ", r2_score(self.ytrue,self.ypred))                  
        print("RMS error Random Forest : ", np.sqrt(mean_squared_error(y_val,y_pred_val_rf)))    
     
        print("Doing Cross Validation")
        scorers = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        scores = cross_validate(rf_reg, self.X, self.y, cv=5, scoring=scorers)
        for key,item in scores.items():
            print(key)
            print(-np.mean(item))
            print("")
        
    def setParamsMultiStage(self,a,b,g,d):
        
        self.alpha = a
        self.beta = b
        self.gamma = g
        self.delta = d
        
       
    def multiStageClassReg(self):
        
        #Create split for train and test
        print("Splitting Train/ Test")
        x_train, x_test, l_train, l_test , y_train, y_test = train_test_split(self.X, self.classes, self.y, test_size=self.alpha, random_state=1)
    
        #Create split for classification and regression
        #Note that in the below split, y_class are not utilized
        print("Splitting Train into Classification / Regression ")
        x_class, x_reg, l_class, l_reg , y_class, y_reg = train_test_split(x_train, l_train, y_train, test_size=self.beta, random_state=2)
        
        #Now create traning and test sets for classification
        print("Splitting Classification into CLassification Train / Classification Test")
        x_class_train, x_class_test, l_class_train, l_class_test = train_test_split(x_class, l_class, test_size=self.gamma, random_state=3)
        print("#Samples for classification training and testing : ",len(l_class_train),len(l_class_test))
        
        print("RF model for classification")
        rf_class = RandomForestClassifier(n_estimators=1000,random_state=0)
        rf_class = self.classification_task(rf_class,x_class_train,l_class_train,x_class_test,l_class_test)
        
        print("Extra Trees model for classification")
        et_class = ExtraTreesClassifier(n_estimators=1000, random_state=0)
        et_class = self.classification_task(et_class,x_class_train,l_class_train,x_class_test,l_class_test)

        print("XGBoost model for classification") 
        xg_class = XGBClassifier(n_estimators=1000,seed=0)
        xg_class = self.classification_task(xg_class,x_class_train,l_class_train,x_class_test,l_class_test)
       
        #Now do the regression training / testing
        #First filter regression data based on classification label
        print("Total # Samples for Regression (target = 0/1) : ",len(y_reg)) 
        
        x_reg = x_reg[(l_reg == 1.0),:]
        y_reg = y_reg[(l_reg == 1.0)]

        print("Total # Samples for Regression (target = 1) : ",len(y_reg))   
        print("Splitting Regression into Regression Train / Regression Test")
        x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(x_reg, y_reg, test_size=self.delta, random_state=4)        
        print("#Samples for Regression training and testing : ",len(y_reg_train),len(y_reg_test))
        
        print("Random forest model for regression : ")
        rf_reg = RandomForestRegressor(n_estimators=1000, random_state=0)
        rf_reg = self.regression_task(rf_reg, x_train, y_train, x_test, y_test)
        
        print("Extra Trees model for regression")
        et_reg = ExtraTreesRegressor(n_estimators=1000,random_state=0)
        et_reg = self.regression_task(et_reg, x_train, y_train, x_test, y_test)
        
        print("XGBoost model for regression")
        xg_reg = XGBRegressor(n_estimators=1000,random_state=0)
        xg_reg = self.regression_task(xg_reg, x_train, y_train, x_test, y_test)        
        
        #Final testing and error performance
        
        class_models = [rf_class,et_class,xg_class]
        reg_models = [rf_reg, et_reg, xg_reg]
        desc = ["RF","ET","XG"]
        
        for idx1 in range(3):
            for idx2 in range(3):
                
                print("\n")
                title_str = desc[idx1] + " for classification and " + desc[idx2] + " for regression"
                print(title_str)
                print("\n")
                self.combined_task(class_models[idx1], reg_models[idx2], x_test, l_test, y_test)
                
        #print("XGBoost (class) and RF (reg) : ")
        #print(" ")        
        #self.combined_task(xg_class, rf_reg, x_test, l_test, y_test)        
        
    def classification_task(self, model,x_train,y_train,x_test,y_test):
        
        print("Model training.")
        model.fit(x_train,y_train)
        pred_labels = model.predict(x_test)
        conf_mat = confusion_matrix(y_test,pred_labels)
        print("Confusion matrix for the current model : ")
        print(conf_mat)
        print("Classification accuracy : ", accuracy_score(y_test,pred_labels)) 
        print("Precision score = ",precision_score(y_test,pred_labels))
        print("Recall score = ",recall_score(y_test,pred_labels))
        print("F1 score = ",f1_score(y_test,pred_labels))
        
        return model
        
        
    def regression_task(self, model,x_train,y_train,x_test,y_test):
        
        print("Model training.")
        model.fit(x_train,y_train)
        #print(model.feature_importances_)    
        print("R2 for the model : ", model.score(x_test,y_test))
        y_pred = model.predict(x_test)
        
        if (self.logTransform == True):
            y_test = np.exp(y_test)-1.00
            y_pred = np.exp(y_pred)-1.00
                
        print("RMS error for the model : ", np.sqrt(mean_squared_error(y_test,y_pred)))    
     
        return model    
    
    def combined_task(self,c_model,r_model,x_test,l_test,y_test):
        
        l_pred = c_model.predict(x_test)
        conf_mat = confusion_matrix(l_test,l_pred)
        print("Confusion matrix for the current model : ")
        print(conf_mat)
        
        print("Final Classification accuracy for the current model : ", accuracy_score(l_test,l_pred))    
        print("Final Precision score = ",precision_score(l_test,l_pred))
        print("Final Recall score = ",recall_score(l_test,l_pred))
        print("Final F1 score = ",f1_score(l_test,l_pred))
        
       
        #Get those with label = 1 
        x_test_l1 = x_test[(l_pred == 1),:]
        y_test_l1 = y_test[(l_pred == 1)]
        y_pred_l1 = r_model.predict(x_test_l1)
        
        print("# Testing points  for the final phase : ",len(y_test),len(y_test_l1))
        
        print("Final R2 for the model : ", r_model.score(x_test_l1,y_test_l1))
        
        if (self.logTransform == True):
            y_test_l1 = np.exp(y_test_l1)-1.00
            y_pred_l1 = np.exp(y_pred_l1)-1.00
                
        print("RMS error for the model : ", np.sqrt(mean_squared_error(y_test_l1,y_pred_l1)))    
        