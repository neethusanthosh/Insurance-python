#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


df=pd.read_csv("D:/decoder lectures/casestudy/insurance/insurance.csv")


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.isnull().sum()


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[8]:


a=["sex","smoker","region"]
b=['age','bmi','children','charges']


# In[9]:


for i in a:
    print(df[i].value_counts())
    sns.countplot(x=i,data=df)
    plt.show()


# In[11]:


for i in b:
    sns.displot(df[i],kde=False)
    print(df[i].describe())
    print(df[i].skew())
    plt.show()


# In[12]:


#bivariate Analysis
for i in a:
    sns.swarmplot(x=i,y="charges",data=df)
    plt.show()
    sns.boxplot(x=i,y="charges",data=df)
    plt.show()


# In[13]:


for i in b:
    sns.scatterplot(x=i,y="charges",data=df)
    print(df[[i,"charges"]].corr())
    plt.show()


# In[14]:


sns.scatterplot(x="age",y="charges",hue="smoker",data=df)
plt.show()


# In[15]:


sns.scatterplot(x="age",y="charges",hue="sex",data=df)
plt.show()


# In[16]:


sns.scatterplot(x="age",y="charges",hue="region",data=df)
plt.show()


# In[17]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[18]:


sns.scatterplot(x="bmi",y="charges",hue="smoker",data=df)
plt.show()


# In[19]:


sns.scatterplot(x="bmi",y="charges",hue="smoker",data=df)
plt.show()


# In[20]:


sns.scatterplot(x="bmi",y="charges",hue="sex",data=df)
plt.show()


# In[21]:


sns.scatterplot(x="children",y="charges",hue="smoker",data=df)
plt.show()


# In[22]:


df.head()


# In[23]:


from sklearn.preprocessing import LabelEncoder
L1=LabelEncoder()
L2=LabelEncoder()
L3=LabelEncoder()
df["sex"]=L1.fit_transform(df['sex'])
df["smoker"]=L2.fit_transform(df['smoker'])
df["region"]=L3.fit_transform(df['region'])


# In[24]:


print(L1.classes_)
print(L2.classes_)
print(L3.classes_)


# In[25]:


df.head()


# In[26]:


x=df.drop('charges',axis=1)
y=df['charges']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[27]:


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


# In[29]:


algo=MLPRegressor(hidden_layer_sizes=(50,50),max_iter=2000,verbose=True)
algo.fit(xtrain,ytrain)


# In[30]:


ypred=algo.predict(xtest)


# In[31]:


from sklearn.metrics import mean_absolute_error,r2_score
print(mean_absolute_error(ytest,ypred))
print(r2_score(ytest,ypred))


# In[32]:


algo2=LinearRegression()
algo2.fit(xtrain,ytrain)


# In[33]:


ypred=algo2.predict(xtest)


# In[34]:


print(mean_absolute_error(ytest,ypred))
print(r2_score(ytest,ypred))


# In[35]:


algo3=DecisionTreeRegressor()
algo3.fit(xtrain,ytrain)


# In[36]:


ypred=algo3.predict(xtest)
print(mean_absolute_error(ytest,ypred))
print(r2_score(ytest,ypred))


# In[37]:


import joblib
joblib.dump(algo,r"E:\DataScience-data\ann-model.pkl")


# In[38]:


new=np.array([[25,0,38.23,3,1,1]])
algo.predict(new)


# In[39]:


joblib.dump(L1,r"E:\DataScience-data\ann-model-l1.pkl")
joblib.dump(L2,r"E:\DataScience-data\ann-model-l2.pkl")
joblib.dump(L3,r"E:\DataScience-data\ann-model-l3.pkl")


# In[ ]:





# In[ ]:




