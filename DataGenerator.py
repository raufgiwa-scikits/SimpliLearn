


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












  

















