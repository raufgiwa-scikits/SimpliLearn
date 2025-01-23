


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
from tensorflow.keras import layers,Model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Flatten,Add,Concatenate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
import numpy as np
import pandas as pd

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfk_layers
import tensorflow.keras as tfk
import tensorflow_lattice as tfl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf


from sklearn.base import BaseEstimator, TransformerMixin
from six import with_metaclass

class gami_Layer_Support():
    def __init__(self,Dimension=1):        
        super(gami_Block_Support,self).__init__()
    def coefficients(self): 
        return  {"w":self.a,"c":self.b}
    def calibrate(self):
        if len(np.array(self.bounds).shape)==1:
            V =self.bounds[1]-self.bounds[0]
            C1=NIntegrate_1D(f=self.call,p=1,bounds=self.bounds,N= 101)    
            C2=NIntegrate_1D(f=self.call,p=2,bounds=self.bounds,N= 101)   
            self.b = C1/V
            self.a = (C2-(self.b**2)*V)**0.5   
        elif len(np.array(self.bounds).shape)==2:        
            V=(self.bounds[0][1]-self.bounds[0][0])*(self.bounds[1][1]-self.bounds[1][0])        
            C1=NIntegrate_2D(f=self.call,p=1,bounds=self.bounds,N= [101,101])    
            C2=NIntegrate_2D(f=self.call,p=2,bounds=self.bounds,N= [101,101])  
            self.b = C1/V
            self.a = (C2-(self.b**2)*V)**0.5 
    def test_calibrate(self):
        if len(np.array(self.bounds).shape)==1:
            C1=NIntegrate_1D(f=self.f,p=1,bounds=self.bounds,N= 101)    
            C2=NIntegrate_1D(f=self.f,p=2,bounds=self.bounds,N= 101)   
        elif len(np.array(self.bounds).shape)==2:        
            C1=NIntegrate_2D(f=self.f,p=1,bounds=self.bounds,N= [101,101])    
            C2=NIntegrate_2D(f=self.f,p=2,bounds=self.bounds,N= [101,101])
        return {"C1":C1,"C2":C2}  
    def f(self,x):
        return (self.call(x)-self.b)/self.a            
            
            
                   
        
class gami_Block_Support():
    def __init__(self,):        
        super(gami_Block_Support,self).__init__()
        0
    def Record_log(self, epoch, train_data,data_valid, valid_list):
        if self.record==True: 
            train_pred=self.call(train_data)               
            train_true=train_data["target"] 
            train_weight=train_data["weight"] 
            valid_pred=[self.call(data_valid[i])  for i in valid_list  ]
            valid_true=[data_valid[i]["target"]   for i in valid_list  ]
            valid_weight=[data_valid[i]["weight"] for i in valid_list  ]
            loss_train=[self.loss_fn(train_true, train_pred,train_weight).numpy()]
            loss_valid=[self.loss_fn(valid_true[i], valid_pred[i],valid_weight[i]).numpy() for i in valid_list ]             
            self.Record_Losses[epoch]=[loss_train,loss_valid]
        if self.verbose>2:
            print(f"Epoch {epoch+1}/{self.epochs}: {len(self.trainable_variables)}: {[loss_train,loss_valid]}",end="\r") 
        elif self.verbose>1:
            print(f"Epoch {epoch+1}/{self.epochs}: {len(self.trainable_variables)}",end="\r")  
    def sub_dataset(self,data,idx):
        sub_data={}
        for var in self.meta_data:
            sub_data[var]=data[var][idx]
        for var in ["target", "weight"]:
            sub_data[var]=data[var][idx]    
        return sub_data 
    def sub_dataset_valid(self,data):
        sub_data_valid=[self.sub_dataset(data=data,idx=idx) for idx in self.meta_target["valid"]]
        return sub_data_valid  
    def train_epoch(self,data, train_idx_list):
        for idx in train_idx_list:
            sub_data=self.sub_dataset(data,idx)                        
            self.train_step(sub_data)
    def train_step(self,sub_data):
        with tf.GradientTape() as tape:
            predictions = self.call(sub_data)[:,0]
            loss = self.loss_fn(sub_data["target"], predictions, sub_data['weight'])                    
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # incorrect
        return predictions,loss,gradients    
    def compute_feature_partial_loss(self, data):
        feature_importance_mean = {}
        yp=(tf.reduce_sum(self.call_outputs(data),axis=-1))
        losses =[(self.loss_fn(data["target"],yp-self.call_outputs(data)[:,i],data['weight'])).numpy() for i in range(len(self.meta_data))]
        losses
                
        for k,var in enumerate(list(self.meta_data)):
            feature_importance_mean[var]=losses[k]
        return feature_importance_mean       
 
    def compute_feature_importance_mean(self, data):
        feature_importance_mean = {}
        for var in self.meta_data:
            feature_output = self.Block_Networks[var].call(data[var])
            importance_mean = tf.reduce_mean(tf.abs(feature_output)).numpy()
            feature_importance_mean[var] = importance_mean
        return feature_importance_mean  
    def compute_feature_importance_var(self, data):
        feature_importance_var = {}
        for var in self.meta_data:
            feature_output = self.Block_Networks[var].call(data[var])
            importance_var = tf.math.reduce_variance(feature_output).numpy()
            feature_importance_var[var] = importance_var
        return feature_importance_var  
 
    def get_weights(self,data=None):
        Single_Weights=[]
        wgts=[np.ravel(c.numpy()) for c in self.Output_Layer.variables][0]
        Single_Weights={"Coefficients":{},"Bias":0}
        for n,var in enumerate(list(self.meta_data)):
            Single_Weights["Coefficients"][var]=wgts[n]
        Single_Weights["Bias"]=[np.ravel(c.numpy()) for c in self.Output_Layer.variables][1][0]
        return Single_Weights 

    def test_calibrate(self):
        self.calibrate()
        return [self.Block_Networks[var].test_calibrate() for var in self.Block_Networks]
    def coefficients(self):
        self.calibrate()
        return [self.Block_Networks[var].coefficients() for var in self.Block_Networks]
    def calibrate(self):
        for var in self.Block_Networks:
            self.Block_Networks[var].calibrate()
    def predict(self,data):
        single_outputs = self.call(data)[:,0].numpy()
        return single_outputs   
   
   
def RandomGenerator(num_samples=9, depth=0, data_point_dim=2, distribution="uniform",seed=12457,p0=0,p1=1,Categorical=0,Time_Series=0):    
    if seed is not None:
        np.random.seed(seed)
    input_dim=[num_samples, data_point_dim,]     
    if depth>1:
        input_dim=[num_samples, depth, data_point_dim] 
    if distribution == "uniform":    
        low  = p0  # Default lower bound is 0
        high = p1  # Default upper bound is 1
        x= np.random.uniform(low,high,input_dim)
    elif distribution == "normal":
        mean = p0  # Default mean is 0
        std = p1  # Default standard deviation is 1
        x= np.random.normal(mean,std,input_dim)
    elif distribution == "exponential":
        scale = p1  # Default scale parameter (1/lambda) is 1
        x= np.random.exponential(scale,input_dim)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")
    if Categorical>1:
        x=(x/np.max(np.ravel(np.abs(x)))*(Categorical-1)).astype(int)
    return {"data":x,"input_dim":x.shape,"Categorical":Categorical,"Time_Series":Time_Series}

def NIntegrate_2D(f,p=1, bounds=[[0,1],[0,1]], N=[101,101]):
    
    x_bounds, y_bounds=bounds
    nx, ny=N
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds    
    dx=(x_max-x_min)/(nx-1)
    dy=(y_max-y_min)/(ny-1)
    xy= np.array([[x,y] for x in np.linspace(x_min,x_max,nx+0) for y in np.linspace(y_min,y_max,ny+0)])
    u=f([xy[:,0].reshape(-1,1),xy[:,1].reshape(-1,1)])
    y=np.ravel(u).reshape(nx,ny)  
    y=y**p  
    integral =y[1:-1,1:-1].sum()+(y[0,1:-1].sum()+y[-1,1:-1].sum()+y[1:-1,0].sum()+y[1:-1,-1].sum())/2+(y[0,0]+y[0,-1]+y[-1,0]+y[-1,-1])/4
    integral = integral*dx*dy
    return integral

def NIntegrate_1D(f,p=1, bounds=[0,1], N=101):
    x_min, x_max= bounds
    x = np.linspace(x_min, x_max, N).reshape(-1,1)  # Divide the interval into n subintervals
    u=f(x)
    y=np.ravel(u)
    y=y**p  
    dx=(x_max-x_min)/(N-1)
    integral = dx * (y[0] + y[-1] + 2 * y[1:-1].sum() )/2
    return integral




class IdentityEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting needed for identity transformation

    def transform(self, X):
        return X  # Return the input data as is

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class MonotonicityConstraint(tf.keras.constraints.Constraint):
    def __init__(self,monotonicity):
        self.monotonicity = monotonicity
    def __call__(self,w):
        return self.monotonicity*tf.nn.relu(w*self.monotonicity) 
    def get_config(self):
        return {'monotonicity': self.monotonicity}
    
class MaxNormConstraint(tf.keras.constraints.Constraint):
    def __init__(self,min_norm=-1,max_norm=1,axis=0):
        self.max_norm = max_norm
        self.min_norm = min_norm
        self.axis = axis
    def __call__(self,w):
        return w * tf.clip_by_value(tf.norm(w,axis=self.axis,keepdims=True),self.min_norm,self.max_norm) / (tf.norm(w,axis=self.axis,keepdims=True) + tf.keras.backend.epsilon())
    def get_config(self):
        return {'max_norm': self.max_norm,'axis': self.axis}
    
class NormAndMonotonicityConstraint(tf.keras.constraints.Constraint):
    def __init__(self,norm=None,monotonicity=None,axis=0):
        self.norm = norm
        self.monotonicity = monotonicity
        self.axis = axis

    def __call__(self,w):
        if self.monotonicity:
            w=self.monotonicity*tf.nn.relu(w*self.monotonicity) 
        
        if self.monotonicity:
            norms = tf.norm(w,axis=self.axis,keepdims=True)
            w=w * tf.clip_by_value(norms,self.norm[0],self.norm[1]) / (norms + tf.keras.backend.epsilon())
        return w   
    
    def get_config(self):
        return {
            'max_norm': self.norm,'monotonicity': self.monotonicity,'axis': self.axis
        }

class MonotonicityAndUnitNormConstraint(tf.keras.constraints.Constraint):
    def __init__(self, monotonicity=+0,norm=False, axis=0):
        self.monotonicity = monotonicity
        self.norm = norm
        self.axis = axis
    def __call__(self, w):        
        if self.monotonicity>0:
            w = tf.nn.relu(w)  # Enforce non-negative weights
        elif self.monotonicity <0:
            w = -tf.nn.relu(-w)  # Enforce non-positive weights
        if self.norm:
            norms = tf.norm(w, axis=self.axis, keepdims=True)
            w = w / (norms + tf.keras.backend.epsilon())  # Normalize weights to have unit norm
            
        return w

    def get_config(self):
        return {'monotonicity': self.monotonicity, 'axis': self.axis}

def Sequential_Network(arch_layers=[],input_dim=1,end=True,name="seq",
                       kernel_initializer='zeros',
                       bias_initializer='zeros', 
                       use_bias=True,
                       Monotonicity=0,Norm=None):    
    Hidden_Layers = tfk.Sequential()
    for n,arch_layer in enumerate(arch_layers):
        layer_type = arch_layer["Net"]
        layer_net = arch_layer["Val"]
        if layer_type.lower() == "lstm":
            layer = tfk_layers.LSTM(layer_net,activation="tanh",return_sequences=False,name=f"{name}:LSTM:{n}", #use_bias=use_bias, 
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer )      
        if layer_type.lower() == "categnet":
            layer = CategNet(input_dim,name=f"{name}:CategNet:{n}",use_bias=use_bias)            
        if layer_type.lower() == "dense":
            layer = tfk_layers.Dense(layer_net,activation="relu",name=f"{name}:DENSE:{n}", #use_bias=use_bias, 
                                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        if layer_type.lower() == "drop":
            layer = tfk_layers.Dropout(layer_net,name=f"{name}:DROP:{n}")
        if layer_type.lower() == "tfl-pwl":
            if "Monotonicity" in list(arch_layer):
                if arch_layer["Monotonicity"] == -1:
                    arch_layer["Monotonicity"] = "increasing"
                if arch_layer["Monotonicity"] == 1:
                    arch_layer["Monotonicity"] = "decreasing"
                layer = tfl.layers.Lattice(lattice_sizes=[layer_net], monotonicities=arch_layer["Monotonicity"], output_min=0.0,output_max=1.0,name=f"{name}:TFL-PWL:{n}",)
            else:
                layer = tfl.layers.Lattice(lattice_sizes=[layer_net], output_min=0.0,output_max=1.0,name=f"{name}:TFL-PWL:{n}",)        
        Hidden_Layers.add(layer)
    if end:
        output_layer = tfk_layers.Dense(1,activation=tf.identity,name=f"{name}:output", use_bias=use_bias, 
                                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer) 
        constraint = MonotonicityAndUnitNormConstraint(monotonicity=Monotonicity,norm=Norm)        
        output_layer.kernel_constraint = constraint
        output_layer.name = f"{name}:output"
        Hidden_Layers.add(output_layer)
    return Hidden_Layers

def Sequential_Network_call(inputs,Hidden_Layers):
    x = inputs
    for i in range(len(Hidden_Layers)):
        x = Hidden_Layers[i](x)
    return x

class CategNet(tf.keras.layers.Layer):
    def __init__(self,category_num=6,name="",
                        use_bias=False,
                        kernel_initializer='zeros',
                        bias_initializer='zeros', ):
        super(CategNet,self).__init__()
        self.category_num = category_num
        self.categ_bias = self.add_weight(shape=[self.category_num,1],initializer=kernel_initializer)
        self.bias = 0
        if use_bias:
            self.bias = self.add_weight(initializer=bias_initializer)

    def call(self,inputs):
        dummy = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32), depth=self.category_num)
        self.output_original = tf.matmul(dummy,self.categ_bias) + self.bias
        return self.output_original

class MainEffect_Net(tf.keras.layers.Layer,gami_Layer_Support):
    def __init__(self,input_dim,name="name",arch_layers=[{"Net": "Dense", "Val": 128}],Time_Series=False,
                                                use_bias=False, kernel_initializer='zeros', bias_initializer='zeros',
                                                Monotonicity=0,Norm=None,bounds = [0,1]):
        super(MainEffect_Net,self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.arch_layers = arch_layers
        self.Time_Series = Time_Series
        self.Amplitude = 1
        self.bounds = bounds
        
        self.output_bias = self.add_weight(name=f"output_bias:{name}", shape=[1],  initializer=tf.zeros_initializer(), trainable=True)
        self.F = Sequential_Network(arch_layers=arch_layers,input_dim=self.input_dim[0], use_bias=use_bias,end=True,
                                                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,Monotonicity=Monotonicity,Norm=Norm)
    def call(self,inputs,training=False):
        x = tf.cast(inputs,tf.float32)
        x = self.F(x)+ self.output_bias
        return x
            
class PairEffect_Net(tf.keras.layers.Layer,gami_Layer_Support):
    def __init__(self,input_dim=[None]*2,name=["nameA"]*2,arch_layers=[None]*3,Time_Series=[False]*2,
                                                use_bias=True,Norm=None,
                                                kernel_initializer='zeros', bias_initializer='zeros',bounds = [[0,1] ,[0,1] ]):
        super(PairEffect_Net,self).__init__()
        self.input_dim = input_dim
        self.arch_layers = arch_layers
        self.Time_Series = Time_Series
        self.bounds =bounds
        
        self.output_bias = self.add_weight(name=f"output_bias:{name[0]}x{name[1]}", shape=[1],  initializer=tf.zeros_initializer(), trainable=True)
        self.Hidden_Layers0 = Sequential_Network( arch_layers=arch_layers[0],input_dim=input_dim[0][0], end=False, name=name[0]+"2",use_bias=use_bias,
                                                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, )
        self.Hidden_Layers1 = Sequential_Network( arch_layers=arch_layers[1],input_dim=input_dim[1][0], end=False, name=name[1]+"2",use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, )
        self.F = Sequential_Network( arch_layers=arch_layers[2],input_dim=input_dim[0][0] + input_dim[1][0], end=True, name=f"{name[0]} x {name[1]}",
                                                use_bias=use_bias, Norm=Norm,kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, )
    def call(self,inputs,training=False):
        x0 = tf.cast(inputs[0],tf.float32)
        x1 = tf.cast(inputs[1],tf.float32)
        x0 = self.Hidden_Layers0(x0)
        x1 = self.Hidden_Layers1(x1)
        x = tf.concat([x0,x1],axis=-1)
        x = self.F(x)+self.output_bias
        return x

class MainEffectBlock(tf.keras.layers.Layer, gami_Block_Support):
    def __init__(self,meta_data,meta_target, epochs=10, batch_size = 256,
                 task_type="Regression",learning_rate=0.001,record=True,endLayer=False,verbose=3,
                 kernel_initializer =  tf.keras.initializers.Orthogonal(), bias_initializer =  tf.keras.initializers.Zeros()):
        super(MainEffectBlock,self).__init__()
        self.meta_data = meta_data
        self.meta_target = meta_target
        self.Block_Networks = {}
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.epochs = epochs
        self.batch_size = batch_size
        self.meta_target=meta_target        
        self.meta_data=meta_data   
        self.task_type = task_type      
        self.record=record   
        self.endLayer=endLayer
        self.verbose=verbose
        self.output_bias = 0#self.add_weight(name=f"output_bias:MainBlock", shape=[1],  initializer=tf.zeros_initializer(), trainable=True)
        if endLayer:
            self.Output_Layer = tf.keras.layers.Dense(1,activation="linear", use_bias=False, trainable=True, kernel_initializer=tfk.initializers.Ones(),  
                                                  bias_initializer=tfk.initializers.Zeros(),)
        
        self.Record_Losses={}

        for var in meta_data:
            arch_layers = meta_data[var]["arch_layers"]
            input_dim = meta_data[var]["input_dim"]
            Time_Series = meta_data[var]["Time_Series"]
            if "Monotonicity" not in list(meta_data[var]):
                meta_data[var]["Monotonicity"]=0
            if "Norm" not in list(meta_data[var]):
                meta_data[var]["Norm"]=[-1,1]
            
            print(meta_data[var])
            sg = MainEffect_Net(input_dim=input_dim,name=var, arch_layers=arch_layers, Time_Series=Time_Series, 
                                kernel_initializer =  kernel_initializer, use_bias=True,  
                                bias_initializer =  bias_initializer,
                                Monotonicity=meta_data[var]["Monotonicity"],Norm=meta_data[var]["Norm"])
            self.Block_Networks[var] = sg
    def call(self,data):
        single_outputs = self.call_outputs(data)        
        if self.endLayer:
            single_outputs = self.Output_Layer(single_outputs) 
        else:
            single_outputs = tf.reduce_sum(single_outputs,axis=-1)
            single_outputs = tf.reshape(single_outputs,[-1,1])
        return single_outputs+self.output_bias
    def call_outputs(self,data):
        single_outputs = [ self.Block_Networks[var].call(data[var]) for var in self.meta_data ]
        single_outputs = tf.concat(single_outputs,axis=-1)
        return single_outputs
                
    def Train(self,data):
        self.call(data)
        self.Record_Losses={}
        
        print(f"trainable variables : {len(self.trainable_variables)} Total variable : {len(self.variables)}")
        train_idx=self.meta_target["train"]        
        train_idx_list=[x.tolist() for x in np.array_split(train_idx, int(len(train_idx)/self.batch_size))]
        train_idx=self.meta_target["train"]
        train_data=self.sub_dataset(data,train_idx)
        data_valid=self.sub_dataset_valid(data)
        valid_list=range(len(self.meta_target["valid"])) 
        
        for epoch in range(self.epochs):
            self.train_epoch(data, train_idx_list)
            self.Record_log(epoch, train_data,data_valid, valid_list)
    
class PairEffectBlock(tf.keras.layers.Layer, gami_Block_Support):
    def __init__(self,meta_data,meta_data_pair,meta_target, epochs=10, batch_size = 256,
                 task_type="Regression",learning_rate=0.001,record=True,endLayer=False,verbose=3,
                 kernel_initializer =  tf.keras.initializers.Orthogonal(), bias_initializer =  tf.keras.initializers.Zeros()):
        super(PairEffectBlock,self).__init__()
        self.meta_data_pair = meta_data_pair
        self.Block_Networks = {}
        self.batch_size = batch_size
        self.meta_target=meta_target        
        self.meta_data=meta_data   
        self.task_type = task_type  
        self.epochs = epochs    
        self.record=record
        self.verbose=verbose
        self.x_bounds = [0,1]
        self.y_bounds = [0,1]
        self.endLayer=endLayer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
            
        self.output_bias = 0#self.add_weight(name=f"output_bias:MainBlock", shape=[1],  initializer=tf.zeros_initializer(), trainable=True)
        if self.endLayer:
            self.Output_Layer = tf.keras.layers.Dense(1,activation="linear", use_bias=False, trainable=True, 
                                                      kernel_initializer=tfk.initializers.Zeros(),  bias_initializer=tfk.initializers.Ones(),)
        
        self.Record_Losses={}
        
        for var in meta_data_pair:
            name = meta_data_pair[var]["name"]
            arch_layers = meta_data_pair[var]["arch_layers"]
            input_dim = meta_data_pair[var]["input_dim"]
            Time_Series = meta_data_pair[var]["Time_Series"]
            sg = PairEffect_Net( input_dim=input_dim,name=name, arch_layers=arch_layers,Time_Series=Time_Series, kernel_initializer = kernel_initializer , 
                                use_bias=True, bias_initializer = bias_initializer)
            self.Block_Networks[var] = sg
    def call(self,data):
        single_outputs = self.call_outputs(data)        
        if self.endLayer:
            single_outputs = self.Output_Layer(single_outputs) 
        else:
            single_outputs = tf.reduce_sum(single_outputs,axis=-1)
            single_outputs = tf.reshape(single_outputs,[-1,1])
        return single_outputs+self.output_bias
    def call_outputs(self,data):
        single_outputs = [self.Block_Networks[var].call( [data[var] for var in self.meta_data_pair[var]["name"]] ) for var in self.meta_data_pair  ]
        single_outputs = tf.concat(single_outputs,axis=-1)
        return single_outputs
                
    def Train(self,data):
        self.call(data)
        self.Record_Losses={}
        
        print(f"trainable variables : {len(self.trainable_variables)} Total variable : {len(self.variables)}")
        train_idx=self.meta_target["train"]        
        train_idx_list=[x.tolist() for x in np.array_split(train_idx, int(len(train_idx)/self.batch_size))]
        train_idx=self.meta_target["train"]
        train_data=self.sub_dataset(data,train_idx)
        data_valid=self.sub_dataset_valid(data)
        valid_list=range(len(self.meta_target["valid"]))
        loss_train=[]
        loss_valid=[]
        
        for epoch in range(self.epochs):
            self.train_epoch(data, train_idx_list)
            self.Record_log(epoch, train_data,data_valid, valid_list)
    
class OutputLayer(tf.keras.layers.Layer):

    def __init__(self, input_num, interact_num):
        super(OutputLayer, self).__init__()
        self.interaction = []
        self.input_num = input_num
        self.interact_num = interact_num

        self.main_effect_weights = self.add_weight(name="subnet_weights",
                                              shape=[self.input_num, 1],
                                              initializer=tf.keras.initializers.Orthogonal(),
                                              trainable=True)
        self.main_effect_switcher = self.add_weight(name="subnet_switcher",
                                              shape=[self.input_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)

        self.interaction_weights = self.add_weight(name="interaction_weights",
                                  shape=[self.interact_num, 1],
                                  initializer=tf.keras.initializers.Orthogonal(),
                                  trainable=True)
        self.interaction_switcher = self.add_weight(name="interaction_switcher",
                                              shape=[self.interact_num, 1],
                                              initializer=tf.ones_initializer(),
                                              trainable=False)
        self.output_bias = self.add_weight(name="output_bias",
                                           shape=[1],
                                           initializer=tf.zeros_initializer(),
                                           trainable=True)


    def call(self, inputs):
        [input_main_effect,input_interaction]=inputs
        if len(input_interaction) > 0:
            output = (tf.matmul(input_main_effect, self.main_effect_switcher * self.main_effect_weights)
                   + tf.matmul(input_interaction, self.interaction_switcher * self.interaction_weights)
                   + self.output_bias)
        else:
            output = (tf.matmul(input_main_effect, self.main_effect_switcher * self.main_effect_weights)
                   + self.output_bias)
        return output

class GAMI(tf.keras.layers.Layer,gami_Block_Support):
    def __init__(self,meta_data=None,meta_target=None, meta_data_pair=None, epochs=5, batch_size = 256,
                 task_type="Regression",learning_rate=[0.001]*3,record=True, verbose=3):
        super(GAMI, self).__init__()

        self.MB = MainEffectBlock(meta_data=meta_data, meta_target=meta_target, epochs=epochs[0], 
                                  batch_size=batch_size, verbose=verbose, learning_rate=learning_rate[0])
        self.PB = PairEffectBlock(meta_data=meta_data, meta_data_pair=meta_data_pair, meta_target=meta_target, epochs=epochs[1], 
                                  batch_size=batch_size, verbose=verbose, kernel_initializer=tfk.initializers.Zeros(),learning_rate=learning_rate[1])

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate[2])
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.epochs = epochs
        self.batch_size = batch_size
        self.meta_target=meta_target        
        self.meta_data=meta_data   
        self.meta_data_pair=meta_data_pair
        self.Record_Losses=[{},{},{},{}]
        self.task_type = task_type
        self.Train_Stage=0 # 0: Main 1: Pair 2: Fine
        self.record = record
        self.verbose = verbose
        self.output_bias = 0#self.add_weight(name="output_bias", shape=[1], initializer=tf.zeros_initializer(), trainable=True)
        if self.task_type.lower() == "regression":
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif self.task_type.lower() == "classification":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            raise ValueError("The task type is not supported")
         


    def predict(self,data,):
        return self.call(data)
             
    def call(self,data):
        o1 = self.MB.call(data)
        o2 = self.PB.call(data)
        o=o1+o2+self.output_bias        
        return o  # tf.reshape(o,-1) 
      
 
    def Train(self,data,stage=0):
        self.call(data)
        self.MB.trainable=True if stage in [0,2] else False
        self.PB.trainable=True if stage in [1,2] else False
        
        
        print(f"MB: {len(self.MB.trainable_variables)} PB: {len(self.PB.trainable_variables)} All: {len(self.trainable_variables)}   ")
        train_idx=self.meta_target["train"]        
        train_idx_list=[x.tolist() for x in np.array_split(train_idx, int(len(train_idx)/self.batch_size))]
        train_idx=self.meta_target["train"]
        train_data=self.sub_dataset(data,train_idx)
        data_valid=self.sub_dataset_valid(data)
        valid_list=range(len(self.meta_target["valid"])) 
        
        for epoch in range(self.epochs[stage]):
            for idx in train_idx_list:
                sub_data=self.sub_dataset(data,idx)                                           
                if stage==0:
                    self.MB.train_step(sub_data)                                       
                if stage==1:
                    self.PB.train_step(sub_data)
                elif stage==2:
                    self.train_step(sub_data) 
            self.Record_log(epoch, train_data,data_valid, valid_list)
        self.Record_log(epoch, train_data,data_valid, valid_list)
        
        
    def Record_log(self,epoch, train_data,data_valid, valid_list):
        self.Record_Losses={}
        if self.verbose>2:
            train_pred=self.call(train_data)               
            train_true=train_data["target"] 
            train_weight=train_data["weight"] 
            valid_pred=[self.call(data_valid[i])  for i in valid_list  ]
            valid_true=[data_valid[i]["target"]   for i in valid_list  ]
            valid_weight=[data_valid[i]["weight"] for i in valid_list  ]
            loss_train=[self.loss_fn(train_true, train_pred,train_weight).numpy()]
            loss_valid=[self.loss_fn(valid_true[i], valid_pred[i],valid_weight[i]).numpy() for i in valid_list ]             
            self.Record_Losses[epoch]=[loss_train,loss_valid]
            print(f"Epoch {epoch+1}/{self.epochs}: {len(self.trainable_variables)}: {[loss_train,loss_valid]}",)#end="\r") 
        elif self.verbose>1:
            print(f"Epoch {epoch+1}/{self.epochs}: {len(self.trainable_variables)}",end="\r") 

   
        
        
        


def sigmoid(x,l=1):
    return 1/(1+np.exp(-x*l))

def actv(x):
    return 1*(x>0)