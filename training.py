#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import time
import torch.nn.functional as F


# In[1]:


def training(model, datasets, datasets_size, optimizer, scheduler, epochs=10):
    phases = ['train', 'valid']
    
    best_valid_accuracy = -1.0
    
    training_loss = []
    validation_loss = []
        
    for epoch in range(epochs):                
        print(f'-------------- Epoch {epoch} --------------')        
        
        for phase in phases:
            epoch_begin = time.time()
            epoch_accuracy = 0.0
            epoch_losses = 0.0
            
            if phase == 'train':
                model.train()
                
            if phase == 'valid':
                model.eval()
                
            for data in datasets[phase]:
                if phase == 'train':
                    optimizer.zero_grad()
                    
                inputs, labels = data
                labels = labels.type(torch.LongTensor)
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                output = model(inputs)
                _, preds = torch.max(output, 1)
                loss = F.nll_loss(output, labels)
                
                epoch_losses += loss.sum().detach()
                epoch_accuracy += torch.sum(preds == labels).detach().item()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                        
            epoch_losses = epoch_losses / datasets_size[phase]
            epoch_accuracy = 100. * epoch_accuracy / datasets_size[phase]
            print(f'{phase} Epoch Losses : {epoch_losses:.5f} :: Accuracy : {epoch_accuracy:.5f} :: Time : {(time.time() - epoch_begin):.4f}s')
            
            if phase == 'train':
                training_loss.append(epoch_losses)
            if phase == 'valid':
                if best_valid_accuracy < epoch_accuracy:
                    best_valid_accuracy = epoch_accuracy
                    torch.save(model.state_dict(), 'final_model.pth')
                validation_loss.append(epoch_losses)
        
        scheduler.step()
    return training_loss, validation_loss

