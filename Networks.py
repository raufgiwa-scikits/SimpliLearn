


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

# import shap



# D:Dense, L:LSTM, R: RNN,  O: Drop out,
# {"Net":"Dense","Val":128},
# {"Net":"LSTM","Val":128},
# {"Net":"RNN","Val":128},
# {"Net":"Drop out","Val":0.3},
# {"Net":"Drop out","Val":128},
# {"Net":"TFL-PWL","Val":10, "Constraint": ['increasing'] },

def_arch_layers=[{"Net":"Dense","Val":128},{"Net":"Drop out","Val":0.3}, {"Net":"Dense","Val":128},{"Net":"Drop out","Val":0.3},]


class SingleEffect_Network2():    
    def __init__(self, input_dim,Feature_name="Feature_Name", embedding_dim=False, arch_layers= def_arch_layers, **kwargs):
        
        self.input_dim=input_dim
        self.feature_name=Feature_name
        self.embedding_dim=embedding_dim
        self.arch_layers=arch_layers
        self.inputs, self.outputs =self.SingleNet()
        self.Model=Model(self.inputs, self.outputs)

    def SingleNet(self,input_dim=None, embedding_dim=None, arch_layers=None): 
        if input_dim is None:
            input_dim    =self.input_dim
        if embedding_dim is None:
            embedding_dim=self.embedding_dim
        if arch_layers is None:
            arch_layers  =self.arch_layers
        
        if embedding_dim:
            inputs = layers.Input(shape=input_dim)
            x = layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=1)(inputs)        
            x = layers.Flatten()(x)
        else:
            inputs = layers.Input(shape=input_dim)
            x=inputs
        for arch_layer in arch_layers:
            layer_type=arch_layer["Net"]
            layer_net=arch_layer["Val"]
            
            if layer_type.lower()=="lstm":
                x = LSTM(layer_net, activation='tanh', return_sequences=False)(x)
            if layer_type.lower()=="dense":
                x = layers.Dense(layer_net, activation="relu")(x) 
            if layer_type.lower()=="drop out":
                x = layers.Dropout(layer_net)(x)
            if layer_type.lower()=="tfl-pwl":
                x = layers.Dropout(layer_net)(x)
                
                
        outputs = layers.Dense(1, activation="linear")(x)    
        return inputs, outputs


class SingleEffect_Net():    
    def __init__(self, input_dim, Feature_name="Feature_Name", embedding_dim=False, arch_layers= def_arch_layers):
        
        self.input_dim=input_dim
        self.Feature_name=Feature_name
        self.embedding_dim=embedding_dim
        self.arch_layers=arch_layers
        
        self.inputs = layers.Input(shape=input_dim)  
        
        
        self.HiddenLayers, self.Output_Layer =self.SingleNet()
        
        if embedding_dim:    
            self.inputs=layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=1)(self.inputs)
            self.inputs=layers.Flatten()(self.inputs)
            
        x=self.inputs
         
        print(Feature_name)
        for net in self.HiddenLayers:
            x=net(x)        
        self.output=  self.Output_Layer(x)  
        self.Model=Model(self.inputs, self.output)
        
        
        

    def SingleNet(self,input_dim=None, embedding_dim=None, arch_layers=None): 
        HiddenLayers=[]
        if input_dim is None:
            input_dim    =self.input_dim
        if embedding_dim is None:
            embedding_dim=self.embedding_dim
        if arch_layers is None:
            arch_layers  =self.arch_layers
              
            
        # if embedding_dim:    
        #     HiddenLayers.append(layers.Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=1))
        #     HiddenLayers.append(layers.Flatten())
            
            
        for arch_layer in arch_layers:
            layer_type=arch_layer["Net"]
            layer_net=arch_layer["Val"]
                        
            if layer_type.lower()=="lstm":
                HiddenLayers.append(LSTM(layer_net, activation='tanh', return_sequences=False))
            if layer_type.lower()=="dense":
                HiddenLayers.append(layers.Dense(layer_net, activation="relu")) 
            if layer_type.lower()=="drop out":
                HiddenLayers.append(layers.Dropout(layer_net))                
            if layer_type.lower()=="tfl-pwl":         
                if "Constraint" in list(arch_layer):
                    lattice_layer = tfl.layers.Lattice(lattice_sizes=layer_net, monotonicities=arch_layer['Constraint'], output_min=0.0, output_max=1.)
                else:                
                    lattice_layer = tfl.layers.Lattice(lattice_sizes=layer_net, output_min=0.0, output_max=1.)                
                HiddenLayers.append(lattice_layer)                
        Outputs = layers.Dense(1, activation="linear") 
        return HiddenLayers, Outputs



from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
import numpy as np
import pandas as pd
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
    
    








































































