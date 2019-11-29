#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("F:\datasets\Earthq.csv")


# In[3]:


df


# In[4]:


new_df1=df.drop(["magType","nst","gap","dmin","rms","updated","type","horizontalError","depthError","magError","magNst","status","locationSource","magSource"],axis=1)


# In[5]:


new_df1


# In[10]:


new_df2=new_df1.drop(["net","id"],axis=1)


# In[9]:


import matplotlib.pyplot as plt
plt.matshow(new_df2.corr())


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df


# In[3]:


df=pd.read_csv("F:\datasets\Earthq.csv")


# In[4]:


df


# In[5]:


df.bar()


# In[6]:


df.hist()


# In[7]:


new_df1.mean(axis=0)


# In[8]:


plt.matshow(new_df2.corr())


# In[11]:


new_df1


# In[12]:


df


# In[13]:


new_df1=df.drop(["magType","nst","gap","dmin","rms","updated","type","horizontalError","depthError","magError","magNst","status","locationSource","magSource","net","id"],axis=1)


# In[14]:


new_df1


# In[15]:


import matplotlib.pyplot as plt
plt.matshow(new_df1.corr())


# In[16]:


plt.matshow(df.corr())


# In[17]:


new_df1


# In[18]:


import matplotlib.pyplot as plt 
# plotting points as a scatter plot 
plt.new_df1('latitude','longitude', label= "stars", color= "green",  
            marker= "*", s=30) 
  
# x-axis label 
plt.xlabel('latitude') 
# frequency label 
plt.ylabel('longitude') 
# plot title 
plt.title('My scatter plot!') 
# showing legend 
plt.legend() 
  
# function to show the plot 
plt.show() 


# In[26]:


import matplotlib.pyplot as plt
new_df1.plot.bar() 
  
# plot between 2 attributes 
plt.bar(new_df1['latitude'], new_df1['longitude']) 
plt.xlabel("latitude") 
plt.ylabel("longitude") 
plt.show() 


# In[23]:


new_df1


# In[24]:


df


# In[29]:


df


# In[30]:


new_df1


# In[ ]:




