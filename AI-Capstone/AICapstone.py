import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
from imblearn.over_sampling import SMOTE as SM_OV
from imblearn.over_sampling import SMOTENC as SM_NC
from imblearn.under_sampling import RandomUnderSampler as SM_UN
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import sklearn.preprocessing as skl_pre
import imblearn.over_sampling as Over_Sampling
import imblearn.under_sampling as Under_Sampling
from IPython.display import clear_output

from sklearn.linear_model import LogisticRegression as LRE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC
import sklearn.metrics as skl_met
import base64
from io import BytesIO


from scipy.special import expit, logit
from xgboost import XGBClassifier  as XGB
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model

from sklearn import metrics
from keras import backend as K
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
import time
import copy
import sys
from datetime import datetime
from joblib import Parallel, delayed
import os
from scipy.stats import multivariate_normal
import plotly.graph_objects as go
#from chart_studio.plotly import iplot
from matplotlib.colors import ListedColormap

import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorboard
from keras.utils.vis_utils import plot_model
#from ann_visualizer.visualize import ann_viz

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, IsolationForest

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler, StandardScaler, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, GridSearchCV, train_test_split, cross_validate, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import mean_squared_error, matthews_corrcoef
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import chi2

from matplotlib.colors import LogNorm


def plot_time_amount(X,x_name='Time',y_name='Amount', bins_x=48, bins_y=24,s=400,x_log=False,y_log=False): 
    #x_column = (X[x_column].values/X[x_column].max()*bins_x).astype(int)
    #y_column = (X[y_column].values/X[y_column].max()*bins_y).astype(int)
    enc_x=skl_pre.MinMaxScaler()
    enc_y=skl_pre.MinMaxScaler()

    x_column = enc_x.fit_transform(X[x_name].values.reshape(-1,1)).ravel()
    y_column = enc_y.fit_transform(X[y_name].values.reshape(-1,1)).ravel()


    x_binned=np.histogram(x_column,bins=bins_x)
    y_binned=np.histogram(y_column,bins=bins_y)

    xy_binned=np.histogram2d(x_column,y_column,bins=[bins_x,bins_y])


    fig = plt.figure(constrained_layout=False,figsize=[10,10])
    gs1 = fig.add_gridspec(nrows=2, ncols=2, wspace=0.01,hspace=0.01, width_ratios=[3,1], height_ratios=[1,3])


    ax1  = fig.add_subplot(gs1[0,0])
    ax12 = fig.add_subplot(gs1[1,0])
    ax2  = fig.add_subplot(gs1[1,1])
    
    # ax0  = fig.add_subplot(gs1[0,1])
    # ax0.set_xticks([])
    # ax0.set_yticks([])
    
    ax1.set_xlim(-1/(2*bins_x),1-1/(2*bins_x))
    ax2.set_ylim(-1/(2*bins_y),1-1/(2*bins_y))
    ax12.set_xlim(-1/(2*bins_x),1-1/(2*bins_x))
    ax12.set_ylim(-1/(2*bins_y),1-1/(2*bins_y))
    ax12.set_ylim(-1/(2*bins_y),1-1/(2*bins_y))
    
    ax1.bar( x_binned[1][:-1]  ,height=x_binned[0]   ,width =0.95/bins_x,alpha=0.2,color="green")
    ax2.barh(y_binned[1][:-1]  ,width =y_binned[0]   ,height=0.95/bins_y,alpha=0.2,color="green")
    
    x=np.linspace(0/bins_x,1-0/bins_x,100)
    y=np.interp(x,x_binned[1][:-1] ,x_binned[0])
    ax1.plot(x,y )


    x=np.linspace(0/bins_y, 1-0/bins_y,100)
    y=np.interp(x,y_binned[1][:-1] ,y_binned[0])
    ax2.plot(y,x)
    
    #ax2.barh(X_Amount['Bin'].values,width =X_Amount['Value'].values,height=0.9,alpha=0.3)
    #for i,x in enumerate(xy_binned[0]):
        #for j,y in enumerate(xy_binned[1]):
            #ax12.rectangle((x,y), 1/bins_x, 1/bins_y, fc='blue',ec="red")

    y,x=np.meshgrid(y_binned[1][:-1],x_binned[1][:-1], )
    im=ax12.pcolor(x,y, xy_binned[0],cmap="Greens", norm=LogNorm())

    colorbar = fig.colorbar(im, ax=ax2,shrink=0.9)

    ax1.grid(True)
    ax2.grid(True)
    ax12.grid(True)

    ax12.set_xlabel("Time")
    ax12.set_ylabel("Amount")
    
    ax1.set_ylabel("Freq.")
    ax2.set_xlabel("Freq.")
    
    if x_log:
        ax1.set_ylabel("Freq. (log)")
        ax1.set_yscale('log')
    
    if y_log:    
        ax2.set_xlabel("Freq. (log)")
        ax2.set_xscale('log')

    idx=np.array([i/5 for i in range(5)])

    ax1.set_xticks(idx,[f"" for j in idx],rotation=90)
    ax2.set_yticks(idx,[f"" for j in idx])


    ax12.set_xticks(idx,[f"{j:3.0f}" for j in enc_x.inverse_transform(idx.reshape(-1,1)).ravel()],rotation=90)
    ax12.set_yticks(idx,[f"{j:3.0f}" for j in enc_y.inverse_transform(idx.reshape(-1,1)).ravel()])
    ax12.grid(True)
    return render_fig_html()
    
    

def is_multivariate_gaussian(data, num_components):
    # Fit a Gaussian Mixture Model to the data
    gmm = GaussianMixture(n_components=num_components, random_state=0)
    gmm.fit(data)

    means = gmm.means_.ravel().tolist()
    covariances = gmm.covariances_.ravel().tolist()
    
    
    # Evaluate the log-likelihood of the data under the GMM model
    log_likelihood = gmm.score(data)

    # Calculate the degrees of freedom based on the number of features and components
    n_features = data.shape[1]
    df = num_components * (n_features + n_features*(n_features+1)//2)

    # Calculate the BIC score for model comparison
    n_samples = data.shape[0]
    bic = -2 * log_likelihood + np.log(n_samples) * df

    # Perform the likelihood ratio test for comparing GMM against a single Gaussian
    chi2_threshold = chi2.ppf(0.95, df)
    is_combination_gaussian = bic < chi2_threshold

    return {"-log(p)":-log_likelihood,"Mean":means,"Var":covariances} # is_combination_gaussian,bic
    return {"-log(p)":-log_likelihood,"Mean":[f"{x:3.3f}"for x in means],"Var":[f"{x:3.3f}"for x in covariances]} # is_combination_gaussian,bic




def get_Best_Gaussian(X,xmin=2,xmax=8):
    d0={}
    d1={}
    for i in range(xmin,xmax):
        d1[i]=is_multivariate_gaussian(X.reshape(-1,1), num_components=i)
        d0[i]=d1[i].copy()
        d0[i]["Mean"]=[f"{x:3.3f}"for x in d0[i]["Mean"]]
        d0[i]["Var"]=[f"{x:3.3f}"for x in d0[i]["Var"]]

    d1=pd.DataFrame(d1).T
    d0=pd.DataFrame(d0).T
    display(d0)
    d1.sort_values(by="-log(p)",inplace=True,ascending=False)
    
    print(f"Best estimate in number of gaussian mixture : {d1.reset_index().T.to_dict()[0]} ")
    return d1,d1.reset_index().T.to_dict()[0]

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def list_to_str(lst,join=""):
    st=""
    for s in lst:
        st=st+s+f" {join} "
    st=st[:-len(join)-2]
    return st

def print_out(x,VERBOSE=0):
    if VERBOSE>0:
        print(x)
        clear_output(wait=True)
    else :
        print(x)

def confusion_matrix_plot(conf_mat,labels={0:"Normal",1:"Fraud"}):
    n=len(conf_mat) 
    fig,ax=plt.subplots(1,1,figsize=(4, 4),layout="tight")
    for i in range(n):
        for j in range(n):
            ax.text(i ,j, conf_mat[i ,j],ha="center")
    im=ax.imshow(conf_mat,cmap='coolwarm')
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix")
    ax.set_xticks([i for i in range(n)],[labels[i] for i in range(n)])
    ax.set_yticks([i for i in range(n)],[labels[i] for i in range(n)],rotation=90)
    colorbar = fig.colorbar(im, ax=ax,shrink=0.7)
    return render_fig_html()        
        
def Over_Sample(df,random_state=42):
    sm = SM_OV(random_state=random_state)
    X=df[x_cols+[y_col]]
    y=df[y_col]
    X_res, y_res = sm.fit_resample(X, y)
    #X_res[y_col]=y
    return X_res

def Under_Sample(df,random_state=42):
    sm = SM_UN(random_state=random_state)
    X=df[x_cols+[y_col]]
    y=df[y_col]
    X_res, y_res = sm.fit_resample(X, y)
    #X_res[y_col]=y
    return X_res

def print_imbalance(datasets=["Train","Hidden"]):
    for dt in datasets:
        print("\n")
        print(f"\n>>> Class Counts: {dt} Data")
        print(str(df_data[dt]["Class"].value_counts()))
        x=np.array(np.bincount(df_data[dt]["Class"].values))
        x=x/np.min(x)
        print("Imbalance Ratio:")
        print(x)

def print_stat(datasets=["Train","Hidden"]):
    for dt in datasets:
        print("\n")
        print(dt + " Data ")
        print("\n>>> Shape: ")
        print(" "*4,df_data[dt].shape)
        print("\n>>> Columns: ")
        print(" "*4,df_data[dt].columns)
  
def render_fig_html():
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    plt.close()
    return f'<img src=\'data:image/png;base64,{encoded}\'>'      
 
class DataIndex():
    def __init__(self,df_data={},y_col="Class", w_col="Weight",
                 ENCODE_COLUMNS=None, Encoder="StandardScaler",VERBOSE=0,
                 CATEGORICAL_FEATURES=[],
                 OVER_SAMPLER_METHOD="RandomOverSampler",
                 UNDER_SAMPLER_METHOD="RandomUnderSampler"):

        self.ValuesDict={} 
        self.ClassWeights={}   
        self.y_col=y_col
        self.w_col=w_col
        self.VERBOSE=VERBOSE
        self.OVER_SAMPLER_METHOD =OVER_SAMPLER_METHOD
        self.UNDER_SAMPLER_METHOD=UNDER_SAMPLER_METHOD
        self.ENCODE_COLUMNS=ENCODE_COLUMNS        
        self.CATEGORICAL_FEATURES=CATEGORICAL_FEATURES
        
        self.ROWS=list(df_data.keys())
        
        df_data["TEST"][y_col]=[-1 for i in range(len(df_data["TEST"]))] 
        
        # add weights
        for rows in self.ROWS:           
            df_data[rows]=self.Add_Tags_To_Data(df_data[rows],rows)
        self.X=pd.concat([df_data[rows] for rows in self.ROWS]) 
        self.COLUMNS=list(set(self.X.columns)-set([y_col,w_col,"__TAG"]))
        
        if ENCODE_COLUMNS==None:        
            self.ENCODE_COLUMNS=self.COLUMNS
            
        self.NON_COLUMNS=[y_col,w_col]
        
        for col in self.X.columns:
            self.ValuesDict[col]={}
            self.ValuesDict[col]["ENC"]=None
            self.ValuesDict[col]["EncoderType"]=Encoder
        
    def Add_Tags_To_Data(self,df,rows):
        y=df[self.y_col]
        df["__TAG"]=[rows for i in range(len(df))] 
        df[self.y_col]=df[self.y_col].astype(int)
        df[self.w_col]=self.Get_SampleWeight(y)        
        self.ClassWeights[rows]=self.Get_ClassWeight(y)  
        return df
         
    def Encode_Data(self,ENCODE_COLUMNS=None):        
        if ENCODE_COLUMNS==None:        
            ENCODE_COLUMNS=self.ENCODE_COLUMNS
        else:
            self.ENCODE_COLUMNS=ENCODE_COLUMNS
            
        for col in self.ENCODE_COLUMNS:   
            X=self.get_X()
            if self.ValuesDict[col]["ENC"]==None:
                enc=eval("skl_pre."+self.ValuesDict[col]["EncoderType"]+"()")                
                self.ValuesDict[col]["ENC"]=enc
                enc.fit(X[col].values.reshape(-1,1))
                print_out(f"Encoding: {col}",self.VERBOSE)
                r=enc.transform(self.X[col].values.reshape(-1,1))[:,0]
                self.X[col]=r
            else:
                print_out(f"Previously Encoded: {col}",self.VERBOSE)
        
    def OverSampler_Data(self,OVER_SAMPLER_METHOD=None):
        X=self.get_X(rows="TRAIN")        
        if OVER_SAMPLER_METHOD==None:
            OVER_SAMPLER_METHOD=self.OVER_SAMPLER_METHOD
        else:
            self.OVER_SAMPLER_METHOD=OVER_SAMPLER_METHOD            
        OVER_SAMPLER_METHOD="Over_Sampling."+OVER_SAMPLER_METHOD
        if OVER_SAMPLER_METHOD=="SMOTENC":
            OVER_SAMPLER_METHOD=OVER_SAMPLER_METHOD+"(categorical_features=CATEGORICAL_FEATURES)"
        else:
            OVER_SAMPLER_METHOD=OVER_SAMPLER_METHOD+"()"
        OVER_SAMPLER=eval(OVER_SAMPLER_METHOD)        
        X_res, y_res = OVER_SAMPLER.fit_resample(X=X[self.COLUMNS+[self.y_col]],y= X[self.y_col])
        X_res[self.y_col]=y_res.values    
        
        X_res=self.Add_Tags_To_Data(X_res,rows="OVER" )        
        self.X=pd.concat([self.X,X_res])
        print_out(f"Over sampling complete: Shape {X_res.shape} Ratio : {self.ClassWeights['OVER']} ")
      
    def UnderSampler_Data(self,UNDER_SAMPLER_METHOD=None):
        X=self.get_X(rows="TRAIN")        
        if UNDER_SAMPLER_METHOD==None:
            UNDER_SAMPLER_METHOD=self.UNDER_SAMPLER_METHOD
        else:
            self.UNDER_SAMPLER_METHOD=UNDER_SAMPLER_METHOD            
        UNDER_SAMPLER_METHOD="Under_Sampling."+UNDER_SAMPLER_METHOD
        if False:
            UNDER_SAMPLER_METHOD=UNDER_SAMPLER_METHOD+"(categorical_features=CATEGORICAL_FEATURES)"
        else:
            UNDER_SAMPLER_METHOD=UNDER_SAMPLER_METHOD+"()"
        OVER_SAMPLER=eval(UNDER_SAMPLER_METHOD)        
        X_res, y_res = OVER_SAMPLER.fit_resample(X=X[self.COLUMNS+[self.y_col]],y= X[self.y_col])
        X_res[self.y_col]=y_res.values            
        
        X_res=self.Add_Tags_To_Data(X_res,rows="UNDER" )        
        self.X=pd.concat([self.X,X_res])
        print_out(f"Under sampling complete: Shape {X_res.shape} Ratio : {self.ClassWeights['UNDER']} ")
        
    def get_X(self,rows=None):
        X=self.X
        if rows==None:
            X=X[X["__TAG"]=="TRAIN"]
        elif type(rows)==type(["0"]) :
            st=['(X["__TAG"].values=="'+row+'")' for row in rows]
            X=X[eval(list_to_str(st,join="+")   ) ]
        else:
            X=X[X["__TAG"]==rows]
        return X
           
    def Get_ClassWeight(self,y):    
        ClassWeight=y.value_counts(dropna=False).to_dict()
        w_max= np.max(list(ClassWeight.values()))        
        for i in ClassWeight.keys():
            ClassWeight[i]=w_max/ClassWeight[i]
        return ClassWeight        
        
    def Get_SampleWeight(self,y):
        ClassWeight=self.Get_ClassWeight(y)
        SampleWeight=y.map(lambda x: ClassWeight[x])
        return SampleWeight        
            
        
class ML_TRAINER():
    def __init__(self,DI,x_cols=None,default_kwargs={},fit_kwargs={},predict_kwargs={},
                 model_name="model_LRE",Train="TRAIN",Test="TEST",Valid="VALID",Cutoff=0.5,SEARCH=False,USE_WEIGHT=True,verbose=0):
        self.model_name=model_name
        model=eval(model_name)
        self.model=model(default_kwargs)
        
        self.Test=Test
        self.Train=Train
        self.Valid=Valid    
        self.x_cols=x_cols
        self.default_kwargs=default_kwargs   
        self.fit_kwargs=fit_kwargs      
        self.predict_kwargs=predict_kwargs 
        self.Cutoff=Cutoff 
        self.verbose=verbose 
        
        self.y={}
        self.performance={}
        self.score_values={} 
        self.score_values_TRAIN_VALID={} 
        self.SEARCH=SEARCH
        self.USE_WEIGHT=USE_WEIGHT
        
        if  x_cols ==None or len(x_cols)==0 : 
            self.x_cols=DI.COLUMNS
            
        self.y_col=DI.y_col  
        self.w_col=DI.w_col     
        self.UNDER_SAMPLER_METHOD=DI.UNDER_SAMPLER_METHOD  
        self.OVER_SAMPLER_METHOD=DI.OVER_SAMPLER_METHOD     
        self.cutoff=0.5    
        model_text=self.model.model_name
        
        
        model_text=model_text+f" with sample weights."
        
        
        if USE_WEIGHT==False:
            model_text=model_text+f" without sample weights."            
        if self.Train=="OVER":
            model_text=model_text+f" with over-sampling using {self.OVER_SAMPLER_METHOD}"
        if self.Train=="UNDER":
            model_text=model_text+f" with under-sampling using {self.UNDER_SAMPLER_METHOD}"
        self.model_text=model_text
        
    def fit(self,DI):        
        X_Train=DI.X[DI.X["__TAG"].values==self.Train][self.x_cols]
        y_Train=DI.X[DI.X["__TAG"].values==self.Train][self.y_col]
        w_Train=DI.X[DI.X["__TAG"].values==self.Train][self.w_col]
        
        X_Valid=DI.X[DI.X["__TAG"].values==self.Valid][self.x_cols]
        y_Valid=DI.X[DI.X["__TAG"].values==self.Valid][self.y_col]
        w_Valid=DI.X[DI.X["__TAG"].values==self.Valid][self.w_col]
        
        
        X_Test=DI.X[DI.X["__TAG"].values==self.Test][self.x_cols]
             
        fit_kwargs={}
        fit_kwargs["fit"]=self.fit_kwargs
        
        fit_kwargs["X_Train"]=X_Train
        fit_kwargs["y_Train"]=y_Train
        
        fit_kwargs["X_Valid"]=X_Valid
        fit_kwargs["y_Valid"]=y_Valid
        
        fit_kwargs["w_Valid"]=w_Valid 
        fit_kwargs["w_Train"]=w_Train     
        
        if self.USE_WEIGHT==False:
            fit_kwargs["w_Valid"]=None 
            fit_kwargs["w_Train"]=None       
        self.model.fit(fit_kwargs)
        print_out(self.model_name + " training complete!.",self.verbose)  
        
        if self.SEARCH:
            self.score_values=self.score(X_Valid, y_Valid)
        else:  
            self.predict_3(X_Train, y_Train, X_Valid, y_Valid,X_Test,)
            self.score_values=self.score(X_Valid, y_Valid)
            self.score_values_TRAIN_VALID["Valid"] =  self.score_values   
            self.score_values_TRAIN_VALID["Train"] =  self.score(X_Train, y_Train)   
            self.score_values_TRAIN_VALID=pd.DataFrame(self.score_values_TRAIN_VALID)
            self.model_performance()
            
    def calculate_performance(self,DI):        
        X_Train=DI.X[DI.X["__TAG"].values==self.Train][self.x_cols]
        y_Train=DI.X[DI.X["__TAG"].values==self.Train][self.y_col]
        w_Train=DI.X[DI.X["__TAG"].values==self.Train][self.w_col]
        
        X_Valid=DI.X[DI.X["__TAG"].values==self.Valid][self.x_cols]
        y_Valid=DI.X[DI.X["__TAG"].values==self.Valid][self.y_col]
        w_Valid=DI.X[DI.X["__TAG"].values==self.Valid][self.w_col]
        
        
        X_Test=DI.X[DI.X["__TAG"].values==self.Test][self.x_cols]
             
        fit_kwargs={}
        fit_kwargs["fit"]=self.fit_kwargs
        
        fit_kwargs["X_Train"]=X_Train
        fit_kwargs["y_Train"]=y_Train
        
        fit_kwargs["X_Valid"]=X_Valid
        fit_kwargs["y_Valid"]=y_Valid
        
        fit_kwargs["w_Valid"]=w_Valid 
        fit_kwargs["w_Train"]=w_Train     
        
        if self.USE_WEIGHT==False:
            fit_kwargs["w_Valid"]=None 
            fit_kwargs["w_Train"]=None       
        

        self.predict_3(X_Train, y_Train, X_Valid, y_Valid,X_Test,)
        self.score_values=self.score(X_Valid, y_Valid)
        self.score_values_TRAIN_VALID["Valid"] =  self.score_values   
        self.score_values_TRAIN_VALID["Train"] =  self.score(X_Train, y_Train)   
        self.score_values_TRAIN_VALID=pd.DataFrame(self.score_values_TRAIN_VALID)
        self.model_performance()           
            
            
        
    def predict_3(self, X_Train, y_Train, X_Valid, y_Valid,X_Test):        
        
        predict_kwargs=self.predict_kwargs
         
        self.y["Train_True"]=y_Train.values
        self.y["Valid_True"]=y_Valid.values     
        predict_kwargs["X_Train"]=X_Train
        predict_kwargs["X_Valid"]=X_Valid
                      
        self.y["Train_Prob"]=self.predict_proba(X_Train)[:,1]
        self.y["Valid_Prob"]=self.predict_proba(X_Valid)[:,1]
        self.y["Test_Prob"]=self.predict_proba(X_Test)[:,1]  
              
        self.y["Train_Pred"]=self.predict_class(self.y["Train_Prob"]).astype(int)
        self.y["Valid_Pred"]=self.predict_class(self.y["Valid_Prob"]).astype(int)
        self.y["Test_Pred"]=self.predict_class(self.y["Test_Prob"]).astype(int)

    def predict(self, X): 
        prob=self.model.predict_proba(X)
        val=self.predict_class(prob).astype(int)
        return val
    
    def predict_proba(self, X): 
        prob=self.model.predict_proba(X)
        return prob
                
    def score(self,  X_Valid, y_Valid):                           
        prob_pred=self.predict_proba(X_Valid)[:,1]
        val_pred=self.predict_class(prob_pred)
        score_values={}
        score_values["roc_auc_score"]=skl_met.roc_auc_score(y_Valid,prob_pred)
        score_values["f1_score"]=skl_met.f1_score(y_Valid,val_pred)
        score_values["mean_squared_error"]=skl_met.mean_squared_error(y_Valid,val_pred)
        score_values["log_loss"]=skl_met.log_loss(y_Valid,prob_pred)
        score_values["accuracy_score"]=skl_met.accuracy_score(y_Valid,val_pred)
        return score_values
        
    def predict_class(self, y):
        return (y>self.Cutoff)*1
        
        
    def model_performance(self):
        if len(self.performance)==0:        
            for q_str in ["Train","Valid"]:
                # classification_report
                self.performance["classification_report_"+q_str]=metrics.classification_report(self.y[q_str+"_True"], 
                                                                                            self.y[q_str+"_Pred"],
                                                                                            output_dict=True)
                # confusion_matrix
                self.performance["confusion_matrix_"+q_str]=metrics.confusion_matrix(self.y[q_str+"_True"], 
                                                                                            self.y[q_str+"_Pred"])
                # AUC
                fpr, tpr, thresholds = metrics.roc_curve(self.y[q_str+"_True"], self.y[q_str+"_Prob"])
                roc_auc = metrics.auc(fpr, tpr)            
                self.performance["RocCurve_"+q_str]=[fpr, tpr, thresholds,roc_auc]
            
    def display_metrics(self):     
        # classification_report
        metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        
        
        report={}
        report["classification_report"]={}
        report["confusion_matrix"]={}
        for q_str in ["Train","Valid"]:
            report["classification_report"][q_str]=pd.DataFrame(self.performance["classification_report_"+q_str]).to_html()
            
        # classification_report
        for q_str in ["Train","Valid"]:
            report["confusion_matrix"][q_str]=confusion_matrix_plot(self.performance["confusion_matrix_"+q_str])
            
        # AUC        
        fig,ax=plt.subplots(1,1)
        legend=[]
        for q_str in ["Train","Valid"]:
            ax.plot(self.performance["RocCurve_"+q_str][0],self.performance["RocCurve_"+q_str][1])
            legend.append(f"{q_str} AUC:{self.performance['RocCurve_'+q_str][3]:3.3f}")
        ax.grid(True)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC Curve:")
        ax.legend(legend)
        
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        plt.close()
        report["RocCurve"] = f'<img src=\'data:image/png;base64,{encoded}\'>'


        html=f"<table ><tr><th colspan='5' ><h1>{self.model_text}</h1></th></tr>"
        html=html+  f"<tr><th></th><th>Classification Report</th><th>Confusion matrix</th><th>ROC Curve</th><th>Performance</th></tr>"
        html=html+  f"<tr><th>Train</th><td>{report['classification_report']['Train']}</td><td>{report['confusion_matrix']['Train']}</td><td rowspan='2'>{report['RocCurve']}</td><td rowspan='2'>{self.score_values_TRAIN_VALID.to_html()}</td></tr>"  
        html=html+  f"<tr><th>Valid</th><td>{report['classification_report']['Valid']}</td><td>{report['confusion_matrix']['Valid']}</td></tr>"   
        html=html+  f"</table>"  
        display(HTML(html))   

    def decision_function(self, X):
        return self.model.decision_function(X=X)
    def calibrate_cutoff(self,x_min=0.45,x_max=0.55,metric='f1_score',showplot=True,pos=-1,bins=201):
        y_true=self.y["Valid_True"]
        y_prob=self.y["Valid_Prob"]
        y_pred=self.y["Valid_Pred"]
        
        cutoffs = np.linspace(x_min,x_max,bins)
        # Calculate F1-scores for each cutoff
        r_scores=[]
        
        
        for cutoff in cutoffs:
            y_p=(y_prob > cutoff)*1
            try:
                if metric=='f1_score':
                    x=skl_met.f1_score(y_p,y_true)
                if metric=='roc_auc_score':
                    x=skl_met.roc_auc_score(y_p,y_prob)
                if metric=='mean_squared_error':
                    x=skl_met.mean_squared_error(y_p,y_true)
                
                r_scores.append([cutoff,x])
            except:
                r_scores.append([cutoff,np.nan])
        r_scores=np.array(r_scores)
        if showplot:
            fig,ax=plt.subplots(1,1,figsize=(4, 4),layout="tight")                    
            ax.plot(r_scores[:,0],r_scores[:,1])
            ax.set_xlabel("Cutoff")
            ax.set_ylabel(metric)
            plt.show()
            
        x=r_scores[np.argsort(r_scores[:,1])[pos]]
        return {"Cutoff":x[0],metric:x[1]}
        

class GridSearch():
    def __init__(self,model_name="",DI=None,Train="TRAIN",predict_kwargs={},fit_kwargs={},default_kwargs={},Vary={},verbose=0,metric="roc_auc_score"):
        ParameterGrid=self.make_grid_params(Vary)
        models_grid={}
        models_scores={}
        self.metric=metric
        self.verbose=verbose          
        self.ParameterGrid=ParameterGrid
                 
        for n in ParameterGrid.keys():
            def_kwargs=default_kwargs.copy()
            for key in ParameterGrid[n]:
                def_kwargs[key]=ParameterGrid[n][key]
                
            print_out(ParameterGrid[n],verbose)
            models_grid[n]=ML_TRAINER(model_name=model_name,DI=DI,Train=Train,
                                                 default_kwargs=default_kwargs,fit_kwargs=fit_kwargs,
                                                 predict_kwargs=predict_kwargs,SEARCH=True);
            models_grid[n].fit(DI=DI);
            score_values=models_grid[n].score_values
            score_values["Parameters"]=ParameterGrid[n]
            models_scores[n]=score_values
        self.models_grid=models_grid     
        self.score_df=pd.DataFrame(models_scores).T     

    def best_parameter(self):
        score_df=self.score_df
        n=score_df[self.metric].values.argmax()
        return self.ParameterGrid[n],    score_df.iloc[n]         

    def make_grid_nos(self, N,lengths):
        NS=[]
        for i in range(N):
            number=[]
            N0=1
            for j in range(len(lengths)):
                number.append(np.mod(int(i/N0),lengths[j]))
                N0=N0*lengths[j]
            NS.append(number)
        return NS
    def make_grid_params(self,Vary):
        ParameterGrid={}
        i=0
        lengths=[len(Vary[key]) for key in Vary.keys()]
        N=np.prod(lengths)
        NS=self.make_grid_nos(N,lengths)
        keys=list(Vary)
        for i,K in enumerate(NS):
            ParameterGrid[i]={}
            for j,key in enumerate(list(Vary.keys())):
                #print(K[j])
                ParameterGrid[i][key]=Vary[key][K[j]]
        return ParameterGrid



class Model_LRE():
    def __init__(self,default_kwargs):
        self.model = LRE(**default_kwargs)
        self.model_name = "Logistic Regression"
    def fit(self, fit_kwargs):
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def predict_proba(self, X):
        return self.model.predict_proba(X=X)
    def decision_function(self, X):
        return self.model.decision_function(X=X)
           
class Model_SVC():
    def __init__(self,default_kwargs):
        self.model = SVC(**default_kwargs)
        self.model_name = "Support Vector Machine"
    def fit(self, fit_kwargs):
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def predict_proba(self, X):
        return self.model.predict_proba(X=X)
    def decision_function(self, X):
        return self.model.decision_function(X=X)
    
class Model_GNB():
    def __init__(self,default_kwargs):
        self.model = GNB(**default_kwargs)
        self.model_name = "GaussianNB"
    def fit(self, fit_kwargs):
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def predict_proba(self, X):
        return self.model.predict_proba(X=X)
    def decision_function(self, X):
        return self.model.predict_log_proba(X=X)
    
class Model_RFC():
    def __init__(self,default_kwargs):
        self.model = RFC(**default_kwargs)
        self.model_name = "Random Forest Classifier"
    def fit(self, fit_kwargs):
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def predict_proba(self, X):
        return self.model.predict_proba(X=X)
    def decision_function(self, X):
        return self.model.predict_log_proba(X=X)
    
class Model_XGB():
    def __init__(self,default_kwargs):
        self.model = XGB(**default_kwargs)
        self.model_name = "XGBoost Classifier"
    def fit(self, fit_kwargs):
        fit_kwargs["fit"]["eval_set"]=[[fit_kwargs["X_Valid"],fit_kwargs["y_Valid"]]]
        fit_kwargs["fit"]["sample_weight_eval_set"]=[fit_kwargs["w_Valid"]]
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def predict_proba(self, X):
        p=self.model.predict_proba(X=X)
        return p
    def decision_function(self, X):
        return logit(self.predict_proba(X))
    
class Model_BGC():
    def __init__(self,default_kwargs):
        self.model = BaggingClassifier(**default_kwargs)
        self.model_name = "Bagging Classifier"
    def fit(self, fit_kwargs):
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def predict_proba(self, X):
        return self.model.predict_proba(X=X)
    def decision_function(self, X):
        return self.model.predict_log_proba(X=X)
    
class Model_ETC():
    def __init__(self,default_kwargs):
        self.model = ExtraTreesClassifier(**default_kwargs)
        self.model_name = "Extra Trees Classifier"
    def fit(self, fit_kwargs):
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def predict_proba(self, X):
        return self.model.predict_proba(X=X)
    def decision_function(self, X):
        return self.model.predict_log_proba(X=X)
    
class Model_ABC():
    def __init__(self,default_kwargs):
        self.model = AdaBoostClassifier(**default_kwargs)
        self.model_name = "AdaBoost Classifier"
    def fit(self, fit_kwargs):
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def predict_proba(self, X):
        return self.model.predict_proba(X=X)    
    def decision_function(self, X):
        return self.model.predict_log_proba(X=X)
    
class Anoml_IFC():
    def __init__(self,default_kwargs):
        self.model = IsolationForest(**default_kwargs)
        self.model_name = "Isolation Forest"
    def fit(self, fit_kwargs):
        self.model.fit(X=fit_kwargs["X_Train"],sample_weight=fit_kwargs["w_Train"],**fit_kwargs["fit"])
    def decision_function(self, X):
        return self.model.decision_function(X=X)        
    
    def predict_proba(self, X):
        x=expit(self.decision_function(X))
        y=np.stack([x,1-x]).T
        return y
    
class Model_ANN():
    def __init__(self,default_kwargs):
        self.model = ANN_net(**default_kwargs)
        self.model_name = "Artificial Neural Network Classifier"
    def fit(self, fit_kwargs):
        validation_data=([fit_kwargs["X_Valid"],fit_kwargs["y_Valid"]])
        
        self.model.fit(X=fit_kwargs["X_Train"],y=fit_kwargs["y_Train"],sample_weight=fit_kwargs["w_Train"],
                       validation_data=validation_data,**fit_kwargs["fit"])
    def decision_function(self, X):
        return self.model.decision_function(X=X)        
    
    def predict_proba(self, X):
        x=expit(self.decision_function(X))
        y=np.stack([x,1-x]).T
        return y
    
class ANN_net():
    def __init__(self, optimizer='Adam',learn_rate=0.01,units =64, dropout=0.2,layers=3,epochs=10,batch_size=2500,input_shape=30,verbose=0,activation='relu'):
        
        self.optimizer=optimizer
        #print(optimizer)
        self.learn_rate=learn_rate
        self.units_layers=[units]*layers
        self.dropout_layers=[dropout]*layers
        self.epochs=epochs
        self.layers=layers
        self.batch_size=batch_size
        self.input_shape=input_shape
        self.verbose=verbose
        self.activation=activation
        self.model=self.create_ANN()
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        
    def create_ANN(self, ):
        model = Sequential()              
        for i,u,d in zip([i for i in range(self.layers)],self.units_layers,self.dropout_layers):
            if i==0:
                model.add(Dense(units =self.units_layers[0],activation=self.activation,input_shape=(self.input_shape,)))
                #model.add(tf.keras.layers.BatchNormalization())  
            model.add(Dense(units=u,kernel_initializer='normal',activation=self.activation))
            model.add(Dropout(d))        
        model.add(Dense(units=1,activation='sigmoid'))
        if self.optimizer=="Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)
        elif self.optimizer=="SGD":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learn_rate)
        elif self.optimizer=="RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learn_rate)
        elif self.optimizer=="Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.learn_rate)
        elif self.optimizer=="Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.learn_rate)
        elif self.optimizer=="Adamax":
            optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learn_rate)
        elif self.optimizer=="Nadam":
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.learn_rate)
            
            
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, 
                      metrics=[tf.keras.metrics.AUC(),'accuracy',"mse",tf.keras.metrics.MeanAbsoluteError()])
        return model
    
    def fit(self,X,y,sample_weight=None,validation_data=None ):
        self.history =self.model.fit(X,y=y,sample_weight=sample_weight,verbose=self.verbose,
                                     epochs=self.epochs,batch_size=self.batch_size,
                                     validation_data=validation_data)
        
        old_key=[]
        new_key=[]
        
        for key in self.history.history.keys():
            if "auc" in key:
                ky=key.split("_")[:-1]
                ky="_".join(ky)
                old_key.append(key)
                new_key.append(ky)

        for ky,key in zip(new_key,old_key):    
            self.history.history[ky]=self.history.history.pop(key)

    
    def predict_proba(self,X):
        y=self.model.predict(X)
        return np.concatenate([y,1-y],axis=1)
    def decision_function(self, X):
        return logit(self.predict_proba(X)[:,1])
    
    
        
        # fit_kwargs["X_Train"]=X_Train
        # fit_kwargs["y_Train"]=y_Train
        # fit_kwargs["w_Train"]=w_Train,tf.keras.metrics.AUC(),tf.keras.metrics.MeanAbsoluteError()
        # fit_kwargs["X_Valid"]=X_Valid
        # fit_kwargs["y_Valid"]=y_Valid
        # fit_kwargs["w_Valid"]=w_Valid