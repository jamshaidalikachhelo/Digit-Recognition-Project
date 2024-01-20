#!/usr/bin/env python
# coding: utf-8

# In[20]:


from sklearn.datasets import load_digits


# In[21]:


digits = load_digits()


# In[22]:


dir(digits)


# In[23]:


digits.data[0]


# In[41]:


import matplotlib.pyplot as plt
plt.matshow(digits.images[1])


# In[55]:


for i in range (7):
    plt.matshow(digits.images[i])


# In[45]:


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()


# In[46]:


from sklearn.model_selection import train_test_split


# In[49]:


X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target, test_size= 0.2)


# In[50]:


reg.fit(X_train,y_train)


# In[51]:


len(X_train)


# In[52]:


len(X_test)


# In[53]:


reg.score(X_test,y_test)


# In[54]:


reg.predict(digits.data[0:7])


# In[65]:


y_predicted = reg.predict(X_test)


# In[67]:


from sklearn.metrics import confusion_matrix
#from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


# In[68]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[86]:


plt.matshow(digits.images[591])


# In[87]:


digits.target[591]


# In[84]:


reg.predict([digits.data[591]])


# In[ ]:




