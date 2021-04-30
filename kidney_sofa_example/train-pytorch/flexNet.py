import torch
import torch.nn as nn
#import torch.nn.functional as F




def hexagon(n_layers, mid_layer_width):
    
    mid_index = int((n_layers + n_layers % 2) / 2)
    
    if mid_index >1:
        decreasing_factor = (1 / 2) ** (1 / (mid_index - 1))
    else:
        decreasing_factor = 1
    
    widths = []
    for i in range(mid_index):
        widths.append(round(mid_layer_width * (decreasing_factor ** i)))
    
    if n_layers % 2 == 0:
        return widths[::-1] + widths
    else:
        return widths[:0:-1] + widths



def linear_block(in_dim, out_dim, dropout_rate, activation_fct):
    
    input_module = [nn.Identity, nn.Dropout]
    
    activation = {"relu": nn.ReLU,
                  "sigmoid": nn.Sigmoid,
                  "regression": nn.Identity,
                  "multi_class": nn.LogSoftmax}
                 
    return nn.Sequential(
                    input_module[0 < dropout_rate < 1](p = dropout_rate),
                    nn.Linear(in_dim, out_dim),
                    activation[activation_fct]())



class FlexNet(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dims, model, dropout_rate, activation_fct):
        
        super(FlexNet, self).__init__()
        self.network = nn.ModuleList()
        
        self.network.append(linear_block(in_dim = input_dim,
                                        out_dim = hidden_dims[0],
                                        dropout_rate = 0,
                                        activation_fct = activation_fct))
        
        for i in range(1, len(hidden_dims)):
            self.network.append(linear_block(in_dim = hidden_dims[i-1],
                                            out_dim = hidden_dims[i],
                                            dropout_rate = dropout_rate,
                                            activation_fct = activation_fct))
        
        out_dim = {"regression": 1,
                   "multi_class": 5}
                   
        self.network.append(linear_block(in_dim = hidden_dims[-1],
                                        out_dim = out_dim[model],
                                        dropout_rate = 0,
                                        activation_fct = model))
                                        
        self.network = nn.Sequential(*self.network)
    
    
    def forward(self, x):
        
        return self.network(x)
