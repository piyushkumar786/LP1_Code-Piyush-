#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  #provides in-memory 2d table object called Dataframe
import numpy as np   #objects for multi-dimensional arrays, 
import matplotlib.pyplot as plt  #multi-platform data visualization


# In[2]:


'''Dictionary-like object, the interesting attributes are: 'data', the data to learn, 'target', the classification labels, 'target_names', the meaning of the labels, 'feature_names', the meaning of the features, 'DESCR', the full description of the dataset, 'filename', the physical location of iris csv dataset'''


# In[3]:


import sklearn


# In[4]:


from sklearn import datasets


# In[5]:


iris=datasets.load_iris()


# In[6]:


iris


# In[8]:


data_set = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


# In[10]:


data_set


# In[19]:


print('\n', 'DATA SET INFORMATION'.center(45, '_'))
print(data_set.info())


# In[20]:


print('\n', 'STATISTICAL INFORMATION'.center(45, '_'))
print(data_set.describe())


# In[41]:


print('\n', 'COLUMNS DTYPE (IF NOMINAL)'.center(45, '_'))
print(data_set.select_dtypes(include=['category']))


# In[22]:


print('\n', 'COLUMNS DTYPE (ALL)'.center(45, '_'))
print(data_set.dtypes)


# In[23]:


print('\n', 'DATA SET MEMORY USAGE'.center(45, '_'))
print(data_set.memory_usage())


# In[24]:


def num_missing(x):
  return sum(x.isnull())


# In[25]:


print('\n', 'MISSING VALUE CHECK'.center(45, '_'))
print("Missing values per column:")
print(data_set.apply(num_missing, axis=0)) 


# In[31]:


data_set[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']].plot.hist(bins=10,
                                                                                                      title='All features')
plt.show()


# In[32]:


data_set[['sepal length (cm)', 'sepal width (cm)']].plot.hist(bins=10, title='Sepal Features')
plt.show()


# In[33]:


data_set[['petal length (cm)','petal width (cm)']].plot.hist(bins=10, title='Petal Features')
plt.show()


# In[42]:


new_data=data_set[["petal length (cm)", "petal width (cm)",'petal length (cm)','petal width (cm)']] 
print(new_data)


# In[39]:



plt.figure(figsize = (10, 7)) 
new_data.boxplot() 


# In[44]:


data_set.isnull().sum()


# In[ ]:



