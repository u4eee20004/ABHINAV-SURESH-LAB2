#!/usr/bin/env python
# coding: utf-8

# # ABHINAV SURESH - AM.EN.U4EEE20004 - MACHINE LEARNING - LAB 2

# In[2]:


import numpy as np
import pandas as pd
col_names=["sepal_length","sepal_width","petal_length","petal_width","type"]#mentioning the names of the columns in the table
data=pd.read_csv("iris_dataset.csv",skiprows=1,header=None,names=col_names)#adding the added file to the program 
data.head(10)#upto what values do the program table to show in the output


# In[3]:


X=data.iloc[:,:-1].values#location of the values of the data in x
Y1=data.iloc[:,-1].values#locaton of the values of the data in Y1
Y=Y1.reshape(-1,1) #for reshapeing the values given by the data
from  sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=41)#splitting the values for each tests
from sklearn import tree #for Decision and random
classifier=tree.DecisionTreeClassifier(min_samples_split=3,max_depth=3,criterion="entropy")#things to add in the trees are added
classifier.fit(X_train,Y_train) #for training the model
classifier.score(X_test,Y_test) #score for 1
tree.plot_tree(classifier)# for plotting the tree given below


# In[ ]:




