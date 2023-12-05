#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pickle
import numpy as np
import pandas as pd


# In[3]:


with open('data/generated/cell_all_train_shap.pkl', 'rb') as f:
    shap_values = pickle.load(f)


# In[9]:


shap_values = np.array(shap_values)


# In[10]:


shap_values.shape


# In[28]:


print(shap_values[0].sum(axis=0))


# In[35]:


cols = list(pd.read_csv('data/dataset_full.csv', index_col=0).columns[2:])


# In[39]:


pd.DataFrame(shap_values.sum(axis=1), columns=cols)


# In[44]:


fi_classes = pd.DataFrame(shap_values.__abs__().sum(axis=1), columns=cols)
fi_classes


# In[61]:


fi_classes.to_csv('data/generated/cell_fi_classes.csv')


# In[60]:


fi = pd.DataFrame([shap_values.__abs__().sum(axis=0).sum(axis=0)], columns=cols)
fi


# In[62]:


fi.to_csv('data/generated/cell_fi.csv')


# In[62]:





# In[ ]:




