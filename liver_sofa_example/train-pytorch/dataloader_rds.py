import torch
from torch.utils.data import Dataset
import pyreadr
import pandas as pd
import numpy as np

class tabularData(Dataset):
    
    
    def __init__(self, filename, target = "target", ignore = []):
        
        #read data file
        data = pyreadr.read_r(filename)[None]
        
        #convert to torch tensors
        self.target     = torch.tensor(data[target].values, dtype = torch.float32).unsqueeze(1)
        self.predictors = torch.tensor(np.array(data.drop(columns = target)), dtype = torch.float32)
        
        print("Data read successfully!")
    
    
    
    def __len__(self):
    
        return len(self.target)
    
    
    
    def __getitem__(self, idx):
        
        return self.predictors[idx], self.target[idx]
    
    
    
    def in_dim(self):
        
        return self.predictors.shape[1]
