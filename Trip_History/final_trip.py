#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data_set = pd.read_csv('/home/piyush/Desktop/Roll_no-38(LP1)/Assignmaent_no-11 - (trip history-DA)/2010-capitalbikeshare-tripdata.csv')


# In[3]:


print(data_set.info())


# In[4]:


X = data_set.iloc[:, [3, 5]].values
y = data_set.iloc[:, -1].values
print(X[:5])
print(y[:5])


# In[5]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y =LabelEncoder()
y = labelencoder_y.fit_transform(y)
print("Sample y:",y[:5])
print("0 :",labelencoder_y.classes_[0])
print("1 :",labelencoder_y.classes_[1])


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, random_state=0)
classifier.fit(X_train, y_train)


# In[8]:


y_pred = classifier.predict(X_test)


# In[9]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[10]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=100),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=100))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree (Train set)')
plt.xlabel('CapitalBS Features')
plt.ylabel('Member Type')
plt.legend()
plt.show()


# In[16]:


X = data_set.iloc [:, [0,3,5]]
Y = data_set.iloc [:,-1]

X = X.values
Y = Y.values


# In[17]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split (X,Y,test_size=0.25)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit (train_X,train_Y)


# In[20]:


answer = knn.predict (test_X)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_Y,answer))


# In[ ]:



