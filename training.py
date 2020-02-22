#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F


# In[ ]:


def training(model, datasets, optimizer, scheduler, epochs=10):
    phases = ['valid']
    
    training_loss = []
    validation_loss = []
    
    for epoch in range(epochs):                
        print(f'-------------- Epoch {epoch} --------------')
        epoch_corrects = 0
        epoch_losses = 0.0
        
        for phase in phases:
            if phase == 'train':
                model.train()
                optimizer.zero_grad()
                
            if phase == 'valid':
                model.eval()
                
            for data in datasets[phase]:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
            
                output = model(inputs)
                loss = F.nll_loss(output, labels)
                print(output)
                print('-----------------')
                print(labels)
                print('-----------------')
                print(loss)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        
        scheduler.step()
            
        

