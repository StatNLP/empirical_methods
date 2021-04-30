import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataloader_rds
import pandas as pd
import numpy as np
import pyreadr
import flexNet


def hexagon(n_layers, mid_layer_width):
    '''
    n_layers > 3
    '''
    
    mid_index = int((n_layers + n_layers % 2) / 2)
    decreasing_factor = (1 / 2) ** (1 / (mid_index - 1))
    
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
        
        print("Model successfully initialized!")
    
    
    def forward(self, x):
        
        return self.network(x)



#----- set global paramters -----
torch.manual_seed(0)

#train input
data_dir = "data/"
train_set = "dataset-training_std_nocirc.rds"

#output
model_name = "myfirstModel"
model_dir = "models/"
predictions_dir = "predictions/"

# set hyperparameters
arch_model_type = "multi_class"
arch_hidden_dims = hexagon(3,256)
arch_dropout = 0
arch_activation_fct = "sigmoid"


opt_learning_rate = 1 / 100
opt_momentum = .9

mini_batch_size = 32
epoches = 10
#--------------------------------



#----- load training data -----
trainset = dataloader_rds.tabularData(data_dir + train_set)
train_loader = torch.utils.data.DataLoader(trainset, batch_size = mini_batch_size, shuffle = True)
#------------------------------



#----- initialize network -----
net = FlexNet(input_dim = trainset.in_dim(),
              hidden_dims = arch_hidden_dims,
              model = arch_model_type, 
              dropout_rate=arch_dropout,
              activation_fct = arch_activation_fct)
#------------------------------



#----- define loss function -----
if arch_model_type == "regression":
    criterion = nn.MSELoss(reduction = "mean")
elif arch_model_type == "multi_class":
    criterion = nn.NLLLoss()
else:
    raise Exception("No loss function defined. Invalid model type specified!")
#--------------------------------



#----- create a stochastic gradient descent optimizer ------
optimizer = optim.SGD(net.parameters(), lr=opt_learning_rate, momentum=opt_momentum)
#-----------------------------------------------------------



for epoch in range(epoches):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        #get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        if arch_model_type == "multi_class":
            labels = labels.to(dtype=torch.long).squeeze()
        
        #zero the parameter gradients
        optimizer.zero_grad()
        
        #forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] average mini batch loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')



#save trained model (before evaluation set model to eval mode with model.eval())
torch.save({"model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoches": epoch},
            model_dir + model_name + ".pt")

#load saved model
#model2 = Net(input_dim = len(data_liver[0][0]))
#savedModel = torch.load("models/myFirstModel.pt")
#model2.load_state_dict(savedModel['model_state_dict'])


#read test data and predict scores using the trained model
test_data = dataloader_rds.tabularData("data/dataset-test_std_nocirc.rds")
net.eval() #switch model to evaluation mode before predictions
if arch_model_type == "regression":
    predictions = pd.DataFrame(net(test_data[:][0]).detach().numpy())
elif arch_model_type == "multi_class":
    predictions = pd.DataFrame(torch.argmax(net(test_data[:][0]), dim=1).detach().numpy())
else:
    print("WARNING: No predications are made!")
pyreadr.write_rds(predictions_dir + model_name + ".rds", predictions)
