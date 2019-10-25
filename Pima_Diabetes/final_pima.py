#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# render the plot inline, instead of in a separate window
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv ("/home/piyush/Desktop/Roll_no-38(LP1)/Assignmaent_no-10 - (pima-DA)/PimaIndiansDiabetes.csv")


# In[3]:


df.shape # take a look at the shape


# In[4]:


df.head(5) # take a look at the first and last few lines


# In[5]:


df.tail(5)


# In[6]:


df.isnull().values.any() #looks like we don't have any nulls


# In[7]:


df.shape # take a look at the shape


# In[8]:


def plot_corr(df,size=11): 
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Yellow means that they are highly correlated.
                                           
    """
    corr = df.corr() # calling the correlation function on the datafrmae
    fig, ax = plt.subplots(figsize=(size,size))
    ax.matshow(corr) # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)),corr.columns) # draw x tickmarks
    plt.yticks(range(len(corr.columns)),corr.columns) # draw y tickmarks


# In[9]:


plot_corr(df)


# In[10]:


df.corr()


# In[12]:


del df['Age']


# In[13]:


df.head()


# In[32]:


num_obs = len(df)
num_true = len(df.loc[df['Class'] == 1])
num_false = len(df.loc[df['Class'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, (num_true/num_obs) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/num_obs) * 100))


# In[19]:




from sklearn.model_selection import train_test_split

feature_col_names = ['TimesPregnant', 'GlucoseConcentration', 'BloodPrs', 'SkinThickness', 'Serum', 'BMI', 'DiabetesFunct']
predicted_class_names = ['Class']

X = df[feature_col_names].values # these are factors for the prediction
y = df[predicted_class_names].values # this is what we want to predict

split_test_size = 0.3

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = split_test_size,random_state=42)


# In[20]:


print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))


# In[22]:




print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['Class'] == 1]), (len(df.loc[df['Class'] == 1])/len(df.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['Class'] == 0]), (len(df.loc[df['Class'] == 0])/len(df.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))


# In[23]:


df.head()


# In[24]:




print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(len(df.loc[df['TimesPregnant'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['GlucoseConcentration'] == 0])))
print("# rows missing thickness: {0}".format(len(df.loc[df['BloodPrs'] == 0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['SkinThickness'] == 0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['Serum'] == 0])))
print("# rows missing diab_pred: {0}".format(len(df.loc[df['DiabetesFunct'] == 0])))



# In[25]:


from sklearn.preprocessing import Imputer
# For all readings == 0, impute with mean
fill_0 = Imputer(missing_values=0,strategy="mean",axis=0)
X_train= fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


# In[26]:


from sklearn.naive_bayes import GaussianNB
# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())


# In[27]:


# predict values using training data
nb_predict_train = nb_model.predict(X_train)
# import the performance metrics library from scikit learn
from sklearn import metrics
# check naive bayes model's accuracy
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train)))
print()


# In[28]:


nb_predict_test=nb_model.predict(X_test)
from sklearn import metrics
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,nb_predict_test)))


# In[29]:


print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test,nb_predict_test)))
print("")


# In[31]:


print("Classification Report")
print("{0}".format(metrics.classification_report(y_test,nb_predict_test)))


# In[ ]:



