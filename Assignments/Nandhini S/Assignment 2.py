#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset = pd.read_csv('Churn_Modelling.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# # PERFORM VARIOUS VISUALIZATIONS

# # UNIVARIANT ANALYSIS

# In[6]:


plt.figure()
dataset.Balance.plot(kind='hist' ,figsize=(12,6))


# # distribution plot

# In[8]:


plt.figure() 
sns.displot(dataset.Tenure)


# In[11]:


plt.figure()
sns.countplot(dataset.Gender)


# In[12]:


dataset.groupby(['Gender'])['EstimatedSalary'].mean().plot(kind='bar')


# # BIVARIANT ANALYSIS

# In[14]:


dataset.groupby(['Gender'])['EstimatedSalary'].mean().plot(kind='bar',ylabel='avg salary',figsize=(12,8))


# # MULTI-VARIANT ANALYSIS

# In[15]:


plt.scatter(x='CreditScore',y='EstimatedSalary',data=dataset,c='g',s=50)
plt.scatter(x='CreditScore',y='Balance',data=dataset,c='b',marker='*')


# # DESCRIPTIVE STATISTICS ON THE DATASET

# In[16]:


dataset.describe()


# # HANDLING THE MISSING VALUES

# In[17]:


dataset.isnull().sum()


# In[18]:


dataset.plot()


# In[19]:


dataset['Age'].value_counts()


# In[20]:


dataset['NumOfProducts'].value_counts()


# In[21]:


dataset['Age'].unique()


# In[22]:


dataset['NumOfProducts'].unique()


# In[23]:


dataset['CreditScore'].unique()


# In[24]:


dataset['Balance'].unique()


# In[25]:


dataset['Balance'].value_counts()


# In[26]:


dataset['EstimatedSalary'].value_counts()


# # OUTLIERS

# In[27]:


dataset['EstimatedSalary'].unique()


# # CHECK FOR CATEGORICAL VALUES AND PERFORM ENCODING

# In[28]:


dataset.info()


# In[29]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoder = LabelEncoder()

dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])


# In[30]:


dataset.head()


# In[31]:


dataset['Geography'].unique()


# In[32]:


x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]


# # SPLITTING DATA INTO INDEPENDENT AND DEPENDENT VARIABLES

# In[33]:


x.head()


# In[34]:


y.head()


# In[35]:


dataset.info()


# In[36]:


dataset.drop('Surname', axis=1, inplace=True)


# In[37]:


x=dataset.iloc[:,6:10]
y=dataset.iloc[:,6:10]


# # SPLITTING THE DATA INTO TRAIN AND TEST

# In[38]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[39]:


X_train.head()


# # #SCALING INDEPENDENT VARIABLES

# In[40]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
print(X_train)


# In[ ]:




