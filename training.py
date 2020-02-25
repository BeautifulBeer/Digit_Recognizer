#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F


# In[2]:


def training(model, datasets, optimizer, scheduler, epochs=10):
    phases = ['train', 'valid']
    
    training_loss = []
    validation_loss = []
    
    dataset_size = {'train' : len(datasets['train'].dataset), 'valid' : len(datasets['valid'].dataset)}
    
    for epoch in range(epochs):                
        print(f'-------------- Epoch {epoch} --------------')        
        
        for phase in phases:
            
            epoch_accuracy = 0.0
            epoch_losses = 0.0
            
            if phase == 'train':
                model.train()
                optimizer.zero_grad()
                
            if phase == 'valid':
                model.eval()
                
            for data in datasets[phase]:
                inputs, labels = data
                labels = labels.type(torch.LongTensor)
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                output = model(inputs)
                _, preds = torch.max(output, 1)
                loss = F.nll_loss(output, labels, reduction='none')
                
                epoch_losses += loss.sum()
                epoch_accuracy += torch.sum(preds == labels).item()
                
                if phase == 'train':
                    loss.sum().backward()
                    optimizer.step()
                        
            epoch_losses = epoch_losses / dataset_size[phase]
            epoch_accuracy = epoch_accuracy / dataset_size[phase]
            print(f'{phase} Epoch Losses : {epoch_losses:.4f}')
            print(f'{phase} Epoch Accuracy : {epoch_accuracy:.4f}')
            
            if phase == 'train':
                training_loss.append(epoch_losses)
            if phase == 'valid':
                validation_loss.append(epoch_losses)
        
        scheduler.step()
    return training_loss, validation_loss


# In[ ]:




