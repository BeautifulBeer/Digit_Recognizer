#!/usr/bin/env python
# coding: utf-8

# In[28]:


import torch
import pandas as pd
import os
from torch.utils.data import Dataset


# In[43]:


class DigitDataset(Dataset):
    def __init__(self, csv_path, phase='training'):
        csv_file = pd.read_csv(csv_path)
        raw_tensor = torch.tensor(csv_file.values)
        print(len(raw_tensor))
        self.phase = phase
        if self.phase == 'training':
            self.data, self.labels = raw_tensor[:, 1:].view(-1, 28, 28), raw_tensor[:, 0]
        if self.phase == 'testing':
            self.data = raw_tensor.view(-1, 28, 28)
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.phase == 'training':
            return self.data[idx], self.labels[idx]
        if self.phase == 'testing':
            return self.data[idx]
        


# In[ ]:





# In[ ]:





# In[ ]:




