


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Flatten, Add, Concatenate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
import numpy as np
import pandas as pd
# import shap

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfk_layers
import tensorflow.keras as tfk
import tensorflow_lattice as tfl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf
# import shap

# D:Dense, L:LSTM, R: RNN,  O: Drop out,
# {"Net":"Dense","Val":128},
# {"Net":"LSTM","Val":128},
# {"Net":"RNN","Val":128},
# {"Net":"Drop out","Val":0.3},
# {"Net":"Drop out","Val":128},
# {"Net":"TFL-PWL","Val":10, "Constraint": +1 },

def_arch_layers=[{"Net":"Dense","Val":128},{"Net":"Drop out","Val":0.3}, {"Net":"Dense","Val":128},{"Net":"Drop out","Val":0.3},]



num_samples=10000
distributions_categories = {
     0: {"distribution":"uniform", "categories":None},
     1: {"distribution":"normal", "categories":None},
     2: {"distribution":"exponential", "categories":None},
     3: {"distribution":"uniform", "categories":5},
     4: {"distribution":"normal", "categories":10},  }

def NumGenerator(num_samples=num_samples, distribution="uniform", seed=None, p0=0,p1=1):

    if seed is not None:
        np.random.seed(seed)

    if distribution == "uniform":    
        low =p0  # Default lower bound is 0
        high = p1  # Default upper bound is 1
        x= np.random.uniform(low, high, num_samples)
    elif distribution == "normal":
        mean = p0  # Default mean is 0
        std = p1  # Default standard deviation is 1
        x= np.random.normal(mean, std, num_samples)
    elif distribution == "exponential":
        scale = p1  # Default scale parameter (1/lambda) is 1
        x= np.random.exponential(scale, num_samples)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")
    return x.reshape(-1,1)

def CatGenerator(num_samples=num_samples, distribution="uniform", seed=None, categories=5):

    if seed is not None:
        np.random.seed(seed)
    x= NumGenerator(num_samples, distribution, seed)
    x=(MinMaxScaler().fit_transform(x)*(categories-1)).astype(int)
    return x


def RandomGenerator(num_samples=num_samples, distribution="uniform", seed=None, categories=None):    
    if categories:
        x = CatGenerator(num_samples, distribution, seed, categories)
    else:
        x = NumGenerator(num_samples, distribution, seed)
    return x

def TimeGenerator(num_samples=num_samples, distributions_categories=distributions_categories, seed=None):

    print(num_samples)
    if seed is not None:
        np.random.seed(seed)
    x=[]
    for C in distributions_categories:
        r=RandomGenerator(num_samples=num_samples, distribution=distributions_categories[C]["distribution"],
                          categories=distributions_categories[C]["categories"])
        x.append(r)
    x=np.hstack(x)    
    x=x.reshape(num_samples,len(list(distributions_categories)),1) 
    return x




class SingleEffect_Net(tf.keras.layers.Layer):    
    def __init__(self, input_dim, Feature_name="Feature_Name", category_num=False, arch_layers= None,Time_Series=False):
        super(SingleEffect_Net, self).__init__()
        self.name=Feature_name
        self.input_dim=input_dim
        self.Feature_name=Feature_name
        self.category_num=category_num
        self.arch_layers=arch_layers       
        self.Time_Series=Time_Series
        # self.inputs = layers.Input(shape=input_dim)      
        self.Hidden_Layers, self.Output_Layer =self.SingleNet()
        # if embedding_dim:    
        #     self.inputs=layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=1)(self.inputs)
        #     self.inputs=layers.Flatten()(self.inputs)
        # x=self.inputs
        # self.Model=Model(self.inputs, self.output)
        # print("Feature_name")      
        
        if self.category_num:  
            self.output_layer_bias = self.add_weight(shape=[1, 1], initializer=tf.zeros_initializer(), trainable=False)
            self.categ_bias = self.add_weight( shape=[self.category_num, 1], initializer=tf.zeros_initializer(), trainable=True)
        
        
    def call(self, x, training=False):  
        print(self.name,self.Time_Series  )

         
        u=x
        if self.category_num:  
            dummy = tf.one_hot(indices=x, depth=self.category_num) 
            u = tf.matmul(dummy, self.categ_bias)+ self.output_layer_bias
            u=tf.reshape(u,[-1,1])
        for net in self.Hidden_Layers:
            u=net(u) 
        y=  self.Output_Layer(u)  
        if self.Time_Series:
            print("Time_Series")
            y=tf.reshape(y,[-1,1])
        return y

    def SingleNet(self,input_dim=None, embedding_dim=None, arch_layers=None): 
        Hidden_Layers=[]
        if input_dim is None:
            input_dim    =self.input_dim
        if embedding_dim is None:
            embedding_dim=self.category_num
        if arch_layers is None:
            arch_layers  =self.arch_layers   
        # if embedding_dim:    
        #     HiddenLayers.append(layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=1))
        #     HiddenLayers.append(layers.Flatten())            
            
        for arch_layer in arch_layers:
            layer_type=arch_layer["Net"]
            layer_net=arch_layer["Val"]      
                              
            if layer_type.lower()=="lstm":
                Hidden_Layers.append(LSTM(layer_net, activation='tanh', return_sequences=False))
                
            if layer_type.lower()=="dense":
                Hidden_Layers.append(layers.Dense(layer_net, activation="relu")) 
                
            if layer_type.lower()=="drop out":
                Hidden_Layers.append(layers.Dropout(layer_net))             
                   
            if layer_type.lower()=="tfl-pwl":         
                if "Constraint" in list(arch_layer):
                    lattice_layer = tfl.layers.Lattice(lattice_sizes=layer_net, monotonicities=arch_layer['Constraint'], output_min=0.0, output_max=1.)
                else:                
                    lattice_layer = tfl.layers.Lattice(lattice_sizes=layer_net, output_min=0.0, output_max=1.)                
                Hidden_Layers.append(lattice_layer)          
                      
        Output_Layer = layers.Dense(1, activation="linear") 
        return Hidden_Layers, Output_Layer    
    
    

class MainEffectBlock(tf.keras.layers.Layer):

    def __init__(self, meta_data):
        super(MainEffectBlock, self).__init__()
        self.meta_data=meta_data
        self.MainEffect_Block_Networks={}
        self.Output_Layer = layers.Dense(1, activation="linear") 
        
        for var in meta_data:
            # print(meta_data[var])
            Feature_name=meta_data[var]["Feature_name"]
            category_num=meta_data[var]["category_num"]
            arch_layers=meta_data[var]["arch_layers"]
            input_dim=meta_data[var]["input_dim"]    
            Time_Series=meta_data[var]["Time_Series"]    
            sg=SingleEffect_Net(input_dim=input_dim, Feature_name=Feature_name, category_num=category_num, arch_layers=arch_layers,Time_Series=Time_Series )
            self.MainEffect_Block_Networks[var]=sg
        # sg.call(data[var])    
    
    def call(self,X):
        single_outputs=[self.MainEffect_Block_Networks[var].call(X[var]) for var in self.meta_data]
        return single_outputs
    
    















       
class MonotonicityConstraint(tf.keras.constraints.Constraint):
    def __init__(self, monotonicity):
        if monotonicity not in [+1, -1]:
            raise ValueError("monotonicity must be +1 or -1")
        self.monotonicity = monotonicity
    def __call__(self, w):
        return tf.nn.relu(w) if self.monotonicity == +1 else -tf.nn.relu(-w)
    def get_config(self):
        return {'monotonicity': self.monotonicity}
    
class MaxNormConstraint(tf.keras.constraints.Constraint):
    def __init__(self, max_norm, axis=0):
        self.max_norm = max_norm
        self.axis = axis
    def __call__(self, w):
        return w * tf.clip_by_value(tf.norm(w, axis=self.axis, keepdims=True), 0, self.max_norm) / (tf.norm(w, axis=self.axis, keepdims=True) + tf.keras.backend.epsilon())
    def get_config(self):
        return {'max_norm': self.max_norm, 'axis': self.axis}
    
class NormAndMonotonicityConstraint(tf.keras.constraints.Constraint):
    def __init__(self, max_norm, monotonicity=+1, axis=0):
        if monotonicity not in [+1, -1]:
            raise ValueError("monotonicity must be +1 or -1")
        self.max_norm = max_norm
        self.monotonicity = monotonicity
        self.axis = axis

    def __call__(self, w):
        # Enforce monotonicity
        if self.monotonicity == +1:
            w = tf.nn.relu(w)
        elif self.monotonicity == -1:
            w = -tf.nn.relu(-w)
        
        # Enforce norm constraint
        norms = tf.norm(w, axis=self.axis, keepdims=True)
        return w * tf.clip_by_value(norms, 0, self.max_norm) / (norms + tf.keras.backend.epsilon())

    def get_config(self):
        return {
            'max_norm': self.max_norm,
            'monotonicity': self.monotonicity,
            'axis': self.axis
        }
        
def make_single_network(arch_layers,name="Net"): 
    Hidden_Layers=[]
    for n,arch_layer in enumerate(arch_layers):
        layer_type=arch_layer["Net"]
        layer_net=arch_layer["Val"]      
                            
        if layer_type.lower()=="lstm":
            layer=tfk_layers.LSTM(layer_net, activation='tanh', return_sequences=False)
            
        if layer_type.lower()=="dense":
            layer=tfk_layers.Dense(layer_net, activation="relu")                       
            if "Monotonicity" in list(arch_layer) and "Norm" in list(arch_layer):
                NormAndMonotonicity=NormAndMonotonicityConstraint(arch_layer["Norm"], arch_layer["Monotonicity"])
                layer.kernel_constraint=NormAndMonotonicity
            
            elif "Monotonicity" in list(arch_layer):
                Monotonicity=MonotonicityConstraint(arch_layer["Monotonicity"])
                layer.kernel_constraint=Monotonicity
                
            elif "Norm" in list(arch_layer):
                Norm=MaxNormConstraint(arch_layer["Norm"])
                layer.kernel_constraint=Norm
            
        if layer_type.lower()=="drop out":
            layer=tfk_layers.Dropout(layer_net)
                
        if layer_type.lower()=="tfl-pwl":         
            if "Monotonicity" in list(arch_layer):
                if arch_layer['Monotonicity']==+1:
                    Monotonicity="increasing"
                if arch_layer['Monotonicity']==-1:
                    Monotonicity="decreasing"     
                           
                layer = tfl.layers.Lattice(lattice_sizes=[layer_net], monotonicities=[arch_layer['Monotonicity']], output_min=0.0, output_max=1.)
            else:                
                layer = tfl.layers.Lattice(lattice_sizes=[layer_net], output_min=0.0, output_max=1.)
        # layer.name=f"{name}:{layer_type.upper()}:{n}"
        Hidden_Layers.append(layer) 
    output_Layer = tfk_layers.Dense(1, activation="linear") 
    output_Layer.name= f"{name}:output"
    Hidden_Layers.append(output_Layer)            
    return Hidden_Layers


def make_call_network(inputs, Hidden_Layers): 

    x=inputs
    for i in range(len(Hidden_Layers)):
        x = Hidden_Layers[i](x)
    return x





class SingleEffect_NetV3(tf.keras.layers.Layer):    
    def __init__(self, input_dim, Feature_name="Feature_Name", #category_num=False, 
                 arch_layers= None,Time_Series=False):
        super(SingleEffect_NetV3, self).__init__()
        self.name=Feature_name
        self.input_dim=input_dim
        print(input_dim)
        self.Feature_name=Feature_name
        # self.category_num=category_num
        self.arch_layers=arch_layers       
        self.Time_Series=Time_Series   
        self.Hidden_Layers=self.make_network(arch_layers, self.input_dim[0])
          
    def call(self, inputs, training=False):        
        x=tf.cast(inputs,tf.float32)
        for i in range(len(self.Hidden_Layers)):
            x = self.Hidden_Layers[i](x)
        self.output_original=x
        return x  

    def make_network(self,arch_layers, input_dim):         
        # arch_layers=self.arch_layers
        name=self.Feature_name
        Hidden_Layers=[]
        for n,arch_layer in enumerate(arch_layers):
            layer_type=arch_layer["Net"]
            layer_net=arch_layer["Val"]      
                                
            if layer_type.lower()=="lstm":
                layer=tfk_layers.LSTM(layer_net, activation='tanh', return_sequences=False)
                
            if layer_type.lower()=="categnet":
                layer=CategNet(input_dim)
                
            if layer_type.lower()=="dense":
                layer=tfk_layers.Dense(layer_net, activation="relu")                       
                if "Monotonicity" in list(arch_layer) and "Norm" in list(arch_layer):
                    NormAndMonotonicity=NormAndMonotonicityConstraint(arch_layer["Norm"], arch_layer["Monotonicity"])
                    layer.kernel_constraint=NormAndMonotonicity
                
                elif "Monotonicity" in list(arch_layer):
                    Monotonicity=MonotonicityConstraint(arch_layer["Monotonicity"])
                    layer.kernel_constraint=Monotonicity
                    
                elif "Norm" in list(arch_layer):
                    Norm=MaxNormConstraint(arch_layer["Norm"])
                    layer.kernel_constraint=Norm
                
            if layer_type.lower()=="drop out":
                layer=tfk_layers.Dropout(layer_net)
                    
            if layer_type.lower()=="tfl-pwl":         
                if "Monotonicity" in list(arch_layer):
                    if arch_layer['Monotonicity']==+1:
                        Monotonicity="increasing"
                    if arch_layer['Monotonicity']==-1:
                        Monotonicity="decreasing"     
                            
                    layer = tfl.layers.Lattice(lattice_sizes=[layer_net], monotonicities=[arch_layer['Monotonicity']], output_min=0.0, output_max=1.)
                else:                
                    layer = tfl.layers.Lattice(lattice_sizes=[layer_net], output_min=0.0, output_max=1.)
            # layer.name=f"{name}:{layer_type.upper()}:{n}"
            Hidden_Layers.append(layer) 
        output_Layer = tfk_layers.Dense(1, activation="linear") 
        output_Layer.name= f"{name}:output"
        Hidden_Layers.append(output_Layer)            
        return Hidden_Layers




class CategNet(tf.keras.layers.Layer):

    def __init__(self,  category_num=6):
        super(CategNet, self).__init__()
        self.category_num = category_num
        self.categ_bias = self.add_weight(shape=[self.category_num, 1], initializer=tf.zeros_initializer(), trainable=True)
        self.bias = self.add_weight(  initializer=tf.zeros_initializer(), trainable=True)

    def call(self, inputs):
        # print(inputs.shape)
        # dummy = tf.one_hot(indices=tf.cast(inputs[:, 0], tf.int32), depth=self.category_num)
        self.output_original = tf.matmul(inputs, self.categ_bias)+self.bias
        output = self.output_original
        return output

 

class MainEffectBlock(tf.keras.layers.Layer):

    def __init__(self, meta_data):
        super(MainEffectBlock, self).__init__()
        self.meta_data=meta_data
        self.MainEffect_Block_Networks={}
        self.Output_Layer = layers.Dense(1, activation="linear") 
        
        for var in meta_data:
            # print(meta_data[var])
            Feature_name=meta_data[var]["Feature_name"]
            # category_num=meta_data[var]["category_num"]
            arch_layers=meta_data[var]["arch_layers"]
            input_dim=meta_data[var]["input_dim"]    
            Time_Series=meta_data[var]["Time_Series"]    
            sg=SingleEffect_NetV3(input_dim=input_dim, Feature_name=Feature_name,# category_num=category_num, 
                                  arch_layers=arch_layers,Time_Series=Time_Series )
            self.MainEffect_Block_Networks[var]=sg
        # sg.call(data[var])    
    
    def call(self,X):
        single_outputs=[self.MainEffect_Block_Networks[var].call(X[var]) for var in self.meta_data]        
        output=self.Output_Layer(tf.concat(single_outputs,axis=-1))        
        return output
    
 

class PairEffectBlock(tf.keras.layers.Layer):

    def __init__(self, meta_data_pair):
        super(PairEffectBlock, self).__init__()
        self.meta_data_pair=meta_data_pair
        self.PairEffect_Block_Networks={}
        self.Output_Layer = layers.Dense(1, activation="linear") 
        
        for var in meta_data_pair:
            # print(meta_data[var])
            Feature_name=meta_data_pair[var]["Feature_name"]
            # category_num=meta_data[var]["category_num"]
            arch_layers=meta_data_pair[var]["arch_layers"]
            input_dim=meta_data_pair[var]["input_dim"]    
            Time_Series=meta_data_pair[var]["Time_Series"]    
            sg=DoubleEffect_NetV3(input_dim=input_dim, Feature_name=Feature_name,# category_num=category_num, 
                                  arch_layers=arch_layers,Time_Series=Time_Series )
            self.PairEffect_Block_Networks[var]=sg
        # sg.call(data[var])    
    
    def call(self,X):       
        single_outputs=[self.PairEffect_Block_Networks[var].call(
            [   X[self.meta_data_pair[var]["Feature_name"][0]],
                X[self.meta_data_pair[var]["Feature_name"][1]]]
            ) for var in self.meta_data_pair]
        
        output=self.Output_Layer(tf.concat(single_outputs,axis=-1))
        
        return output
    
class DoubleEffect_NetV3(tf.keras.layers.Layer):    
    def __init__(self,  input_dim= [None]*2, Feature_name=["Feature_NameA"]*2, arch_layers= [None]*3,Time_Series=[False]*2):
        super(DoubleEffect_NetV3, self).__init__()

        self.name=Feature_name
        self.input_dim=input_dim
        self.Feature_name=Feature_name
        # self.category_num=category_num
        self.arch_layers=arch_layers       
        self.Time_Series=Time_Series  
        
        # print(arch_layers[0])
        self.Hidden_Layers0=self.make_network(arch_layers[0], input_dim[0][0])
        # print(arch_layers[1])
        self.Hidden_Layers1=self.make_network(arch_layers[1], input_dim[1][0])
        self.Hidden_Layers01=self.make_network(arch_layers[2], input_dim[0][0]+input_dim[1][0],end=True)
          
    def call(self,inputs, training=False):   
        
        x0=tf.cast(inputs[0] ,tf.float32)   
        x1=tf.cast(inputs[1] ,tf.float32) 
        
        for i in range(len(self.Hidden_Layers0)):
            x0 = self.Hidden_Layers0[i](x0)
        for i in range(len(self.Hidden_Layers1)):
            x1 = self.Hidden_Layers1[i](x1)
        x01=tf.concat([x0,x1],axis=-1)
        for i in range(len(self.Hidden_Layers01)):
            x01 = self.Hidden_Layers01[i](x01)        
        return x01  
              

    def make_network(self,arch_layers, input_dim=1,end=False):         
        # arch_layers=self.arch_layers
        # name=self.Feature_name
        Hidden_Layers=[]
        for n,arch_layer in enumerate(arch_layers):
            layer_type=arch_layer["Net"]
            layer_net=arch_layer["Val"]      
                                
            if layer_type.lower()=="lstm":
                layer=tfk_layers.LSTM(layer_net, activation='tanh', return_sequences=False)
                
            if layer_type.lower()=="categnet":
                layer=CategNet(input_dim)
                
            if layer_type.lower()=="dense":
                layer=tfk_layers.Dense(layer_net, activation="relu")                       
                if "Monotonicity" in list(arch_layer) and "Norm" in list(arch_layer):
                    NormAndMonotonicity=NormAndMonotonicityConstraint(arch_layer["Norm"], arch_layer["Monotonicity"])
                    layer.kernel_constraint=NormAndMonotonicity
                
                elif "Monotonicity" in list(arch_layer):
                    Monotonicity=MonotonicityConstraint(arch_layer["Monotonicity"])
                    layer.kernel_constraint=Monotonicity
                    
                elif "Norm" in list(arch_layer):
                    Norm=MaxNormConstraint(arch_layer["Norm"])
                    layer.kernel_constraint=Norm
                
            if layer_type.lower()=="drop out":
                layer=tfk_layers.Dropout(layer_net)
                    
            if layer_type.lower()=="tfl-pwl":         
                if "Monotonicity" in list(arch_layer):
                    if arch_layer['Monotonicity']==+1:
                        Monotonicity="increasing"
                    if arch_layer['Monotonicity']==-1:
                        Monotonicity="decreasing"     
                            
                    layer = tfl.layers.Lattice(lattice_sizes=[layer_net], monotonicities=[arch_layer['Monotonicity']], output_min=0.0, output_max=1.)
                else:                
                    layer = tfl.layers.Lattice(lattice_sizes=[layer_net], output_min=0.0, output_max=1.)
            # layer.name=f"{name}:{layer_type.upper()}:{n}"
            Hidden_Layers.append(layer) 
        if end:
            output_Layer = tfk_layers.Dense(1, activation="linear") 
            # output_Layer.name= f"{name}:output"
            Hidden_Layers.append(output_Layer)            
        return Hidden_Layers





















