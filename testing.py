#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[5]:


def testing(model, dataset):
    result = []
    if torch.cuda.is_available():
            model.cuda()
    for data in dataset:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        output = model(inputs)
        _, preds = torch.max(output, 1)
        result += [int(element) for element in preds.tolist()]
    return result

