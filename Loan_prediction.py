#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


df=pd.read_csv("C:/Users/RIYA/Desktop/jupyter/loan train.csv")


# In[18]:


df


# In[19]:


df.shape


# In[20]:


df.head


# In[21]:


df.info


# In[22]:


df.describe


# In[23]:


print(df.mean)


# In[24]:


pd.crosstab(df['Credit_History'],df['Loan_Status'],margins=True)


# In[25]:


df.boxplot(column='ApplicantIncome')


# In[26]:


df['ApplicantIncome'].hist(bins=20)


# In[27]:


df['CoapplicantIncome'].hist(bins=20)


# In[28]:


df.boxplot(column='ApplicantIncome',by='Education')


# In[29]:


df.boxplot(column='LoanAmount')


# In[30]:


df['LoanAmount'].hist(bins=20)


# In[31]:


df['LoanAmount_log']=np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[32]:


df.isnull().sum()


# In[33]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)


# In[34]:


df['Married'].fillna(df['Married'].mode()[0],inplace=True)


# In[35]:


df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)


# In[36]:


df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)


# In[37]:


df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log = df.LoanAmount_log.fillna(df.LoanAmount_log.mean())


# In[38]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)


# In[39]:


df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)


# In[40]:


df.isnull().sum()


# In[41]:


df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])


# In[43]:


df['TotalIncome_log'].hist(bins=20)


# In[44]:


X=df.iloc[:,np.r_[1:5,9:11,13:15]].values
y=df.iloc[:,12].values


# In[45]:


X


# In[47]:


y


# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[55]:


print(X_train)


# In[56]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[59]:


for i in range(0,5):
    X_train[:,i]=labelencoder_X.fit_transform(X_train[:,i])


# In[62]:


X_train[:,7]=labelencoder_X.fit_transform(X_train[:,7])


# In[63]:


X_train


# In[64]:


labelencoder_y = LabelEncoder()


# In[65]:


y_train=labelencoder_y.fit_transform(y_train)


# In[66]:


y_train


# In[67]:


for i in range (0,5):
    X_test[:,i]=labelencoder_X.fit_transform(X_test[:,i])


# In[68]:


X_test[:,7]=labelencoder_X.fit_transform(X_test[:,7])


# In[69]:


labelencoder_y=LabelEncoder()
y_test=labelencoder_y.fit_transform(y_test)


# In[70]:


X_test


# In[71]:


y_test


# In[72]:


from sklearn.tree import DecisionTreeClassifier
DTClassifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(X_train,y_train)


# In[73]:


y_pred = DTClassifier.predict(X_test)
y_pred


# In[74]:


from sklearn import metrics
print('The accuracy of decision tree is :',metrics.accuracy_score(y_pred,y_test))


# In[75]:


from sklearn.naive_bayes import GaussianNB
NBClassifier = GaussianNB()
NBClassifier.fit(X_train,y_train)


# In[76]:


y_pred=NBClassifier.predict(X_test)


# In[77]:


y_pred


# In[78]:


print('The accuracy of Naive Bayes is', metrics.accuracy_score(y_pred,y_test))


# In[79]:


testdata = pd.read_csv("C:/Users/RIYA/Desktop/jupyter/loan_test.csv")


# In[80]:


testdata.head()


# In[81]:


testdata.info()


# In[82]:


testdata.isnull().sum()


# In[83]:


testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)
testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)
testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)
testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)


# In[84]:


testdata.isnull().sum()


# In[85]:


testdata.boxplot(column='ApplicantIncome')


# In[86]:


testdata.boxplot(column='LoanAmount')


# In[87]:


testdata.LoanAmount = testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[89]:


testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[90]:


testdata.isnull().sum()


# In[91]:


testdata['TotalIncome']=testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])


# In[92]:


testdata.head()


# In[94]:


test=testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[95]:


for i in range(0,5):
    test[:,i]=labelencoder_X.fit_transform(test[:,i])


# In[96]:


test[:,7]=labelencoder_X.fit_transform(test[:,7])


# In[97]:


test


# In[101]:


pred=NBClassifier.predict(test)


# In[102]:


pred


# In[ ]:




