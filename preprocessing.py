#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
from torch.utils.data import Dataset
import torchvision


# In[43]:


class DigitDataset(Dataset):
    def __init__(self, csv_data, transform, phase='training'):
        self.csv_data = csv_data
        self.transform = transform
        raw_tensor = torch.tensor(csv_data)
        self.phase = phase
        if self.phase == 'training':
            self.data, self.labels = raw_tensor[:, 1:].view(-1, 28, 28), raw_tensor[:, 0]
        if self.phase == 'testing':
            self.data = raw_tensor.view(-1, 28, 28)
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.phase == 'training':
            return self.transform(self.data[idx]), self.labels[idx]
        if self.phase == 'testing':
            return self.transform(self.data[idx])
        


# In[ ]:





# In[ ]:





# In[ ]:




