import torch
import torch.nn as nn
import torch.optim as optim

import dataloader_rds
import flexNet

import pandas as pd
import numpy as np
import pyreadr
from joblib import Parallel, delayed




#----- set global paramters -----
# hyper parameter grid
model_settings = pd.read_feather("model_specifications.feather")


#input
data_dir = "../data/"
#train_set = "dataset-training_std_nocirc.rds"
#test_set = "dataset-test_std_nocirc.rds"
ablation_set = "dataset-ablation_zeroBili_std.rds"


#output
model_dir = "../models/"
predictions_dir = "../predictions/"
#--------------------------------


def trainModel(pars):
    #----- set input data      ------
    train_set = pars.train_set
    test_set = pars.test_set
    #----- set hyperparameters ------
    torch.manual_seed(pars.random_seed)
    
    arch_model_type = pars.model_type
    arch_hidden_dims = flexNet.hexagon(int(pars.hidden_number), int(pars.hidden_size_max))
    arch_dropout = pars.dropout
    arch_activation_fct=pars.activation_fn
    
    opt_learning_rate = pars.learning_rate
    opt_momentum = .9
    
    mini_batch_size = int(pars.batch_size)
    epoches = int(pars.epoches)
    
    model_name = pars.model_name
    #--------------------------------
    
    
    
    #----- load training data -----
    trainset = dataloader_rds.tabularData(data_dir + train_set)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = mini_batch_size, shuffle = True)
    #------------------------------
    
    
    
    #----- initialize network -----
    net = flexNet.FlexNet(input_dim = trainset.in_dim(),
                          hidden_dims = arch_hidden_dims,
                          model = arch_model_type, 
                          dropout_rate = arch_dropout,
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
        #if i % 1000 == 999:    # print every 1000 mini-batches
        #    print('[%d, %5d] average mini batch loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 1000))
        #    running_loss = 0.0
    
    print('Finished Training of Model_' + model_name + " !")
    
    
    
    #save trained model
    torch.save({"model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoches": epoch},
                model_dir + "model_" + model_name + ".pt")
            
    #switch model to eval mode before calc predictions
    net.eval()
    
    #caclulate train set predictions
    trainset = dataloader_rds.tabularData(data_dir + train_set)
    
    if arch_model_type == "regression":
        predictions = pd.DataFrame(net(trainset[:][0]).detach().numpy())
    elif arch_model_type == "multi_class":
        predictions = pd.DataFrame(torch.argmax(net(trainset[:][0]), dim=1).detach().numpy())
    else:
        print("WARNING: No predications are made for Model_" + model_name + "!")
    
    pyreadr.write_rds(predictions_dir + "train_" + model_name + ".rds", predictions)
    
    
    
    #calc ablation set predictions
    ablationset = dataloader_rds.tabularData(data_dir + ablation_set)
    
    if ablationset.in_dim() == trainset.in_dim():
    
        if arch_model_type == "regression":
            predictions = pd.DataFrame(net(ablationset[:][0]).detach().numpy())
        elif arch_model_type == "multi_class":
            predictions = pd.DataFrame(torch.argmax(net(ablationset[:][0]), dim=1).detach().numpy())
        else:
            print("WARNING: No predications are made for Model_" + model_name + "!")
    
        pyreadr.write_rds(predictions_dir + "ablation_" + model_name + ".rds", predictions)
    
    
    #calc test set predictions
    testset = dataloader_rds.tabularData(data_dir + test_set)
    
    
    if arch_model_type == "regression":
        predictions = pd.DataFrame(net(testset[:][0]).detach().numpy())
    elif arch_model_type == "multi_class":
        predictions = pd.DataFrame(torch.argmax(net(testset[:][0]), dim=1).detach().numpy())
    else:
        print("WARNING: No predications are made for Model_" + model_name + "!")
    
    pyreadr.write_rds(predictions_dir + "test_" + model_name + ".rds", predictions)
    
    return None



Parallel(n_jobs=1)(delayed(trainModel)(pars) for pars in model_settings.itertuples())
