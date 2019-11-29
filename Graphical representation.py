#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("F:\datasets\Earthq.csv")


# In[3]:


df


# In[4]:


new_df1=df.drop(["magType","nst","gap","dmin","rms","updated","net","id","type","horizontalError","depthError","magError","magNst","status","locationSource","magSource"],axis=1)


# In[5]:


new_df1


# In[6]:


df.hist()


# In[7]:


new_df1.hist()


# In[8]:


plt.matshow(df.corr())


# In[9]:


plt.matshow(new_df1.corr())


# In[10]:


import matplotlib.pyplot as plt 
# defining labels 
activities = ['latitude','longitude','depth','mag'] 
# portion covered by each label 
slices = [3, 7, 8,10] 
# color for each label 
colors = ['r', 'y', 'g', 'b'] 
# plotting the pie chart 
plt.pie(slices, labels = activities, colors=colors,startangle=90, shadow = True, explode = (0.1, 0.1, 0.1,0.1),radius = 1.2, autopct = '%1.1f%%') 
# plotting legend 
plt.legend() 
# showing the plot 
plt.show() 


# In[ ]:




