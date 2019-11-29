#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 


# In[14]:


ds=pd.read_csv("Earthq.csv")
ds=ds[["time","latitude","longitude","depth","mag","magType","place"]]


# 

# In[16]:


ds.info()


# In[17]:


ds.describe()


# In[18]:


ds["time"]=[x.split("-")[1] for x in ds["time"]]


# In[34]:


x=ds[["latitude","longitude"]]
y=ds["depth"]
platitude=923.23220
plongitude=92.232232
# Xpre=np.array([platitude])
# Xpre=Xpre.reshape(-1, 1)


# In[41]:


regr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=243,test_size = 0.15) 
regr.fit(X_train, y_train)
# predepth=regr.predict(Xpre)
Xpre=X_train.head(1)
predepth=regr.predict(Xpre)


# In[43]:


x=ds[["latitude","longitude"]]
y=ds["mag"]
platitude=923.23220
plongitude=92.232232

# Xpre=np.array([[platitude],[plongitude]])
# Xpre.reshape(-1, 1)
# Xpre


# In[44]:


regr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=236,test_size = 0.15) 
regr.fit(X_train, y_train)
Xpre=X_test.head(1)
premag=regr.predict(Xpre)


# In[45]:


depth=max(ds["depth"])
depth


# In[46]:


mag=max(ds["mag"])
mag


# In[96]:


prob=(predepth/depth)*(premag/mag)
prob=prob*100;
print(prob)


# # End

# In[48]:


mag=max(ds["mag"])
mag


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 


# In[3]:


ds.head()


# In[4]:


ds=pd.read_csv("Earthq.csv")
ds=ds[["time","latitude","longitude","depth","mag","magType","place"]]


# In[5]:


ds.head()


# In[6]:


ds.info()


# In[7]:


ds.describe()


# In[8]:


ds["time"]=[x.split("-")[1] for x in ds["time"]]


# In[ ]:





# In[9]:


x=ds[["latitude","longitude"]]
y=ds["depth"]
platitude=923.23220
plongitude=92.232232
Xpre=np.array([[platitude],[plongitude]])
# Xpre.reshape(-1, 1)
Xpre


# In[10]:


regr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=243,test_size = 0.15) 
regr.fit(X_train, y_train)
predepth=regr.predict(Xpre)


# In[11]:


depth=max(ds["depth"])
depth


# In[12]:


mag=max(ds["mag"])
mag


# In[56]:


from sklearn.preprocessing import StandardScaler as ss


# In[57]:


ds=['latitude','longitude','mag','depth']


# In[58]:


scaler=StandardScaler()


# In[59]:


scaler = StandardScaler()


# In[60]:


print(StandardScaler.fit(data))


# In[61]:


scaler = ss


# In[ ]:





# In[62]:


print(ss.fit(data))


# In[63]:


print(ss.fit(ds))


# In[64]:


from sklearn.preprocessing import StandardScaler as ss
data=['latitude','longitude','mag','depth']
scaler = ss
print(ss.fit(data))


# In[65]:


from sklearn.preprocessing import StandardScaler as ss
data=['latitude','longitude','mag','depth']
scaler = ss
print(ss.fit('latitude','longitude'))


# In[66]:


from sklearn.preprocessing import StandardScaler as ss
data=['latitude','longitude','mag','depth']
scaler = ss
print(ss.fit('latitude','longitude'))


# In[67]:


dataset=pd.read("Earthquake.csv")


# In[68]:


import pandas as pd
import numpy as np


# In[69]:


ds


# In[70]:


ds.describe()


# In[71]:


ds.head()


# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 


# In[73]:


ds=pd.read_csv("Earthq.csv")
ds=ds[["time","latitude","longitude","depth","mag","magType","place"]]


# In[74]:


ds.head()


# In[75]:


X=ds.iloc[:,:-1].values
Y=ds.iloc[:,5].values


# In[76]:


X


# In[77]:


Y


# In[78]:


ds.head()


# In[79]:


from sklearn.preprocessing import train_test_split


# In[81]:


from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test = train_test_split( X,y,test_size=0.2 )


# In[82]:


from sklearn.preprocessing import StandardScaler
stdscaler=StandardScaler()


# In[ ]:


X_train = stdscaler.fit_transform()


# In[83]:


ds


# In[85]:


a=ds['latitude']
type(a)


# In[88]:


mi=min(a)
#print(mi)
ma=max(a)
c=ma-mi


# In[ ]:


lst=[]
for i in range(0,14698):
    lst = (a[i]-mi)/c


# In[ ]:


lst


# In[ ]:





# In[97]:


ds


# In[98]:


import matplotlib.pyplot as plt


# In[99]:


ds.hist()


# In[100]:


plt.show()


# In[101]:


ds.plot.bar() 
plt.bar(ds['latitude'], ds['longitude']) 
plt.xlabel("latitude") 
plt.ylabel("longitude") 
plt.show() 


# In[104]:


import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
  
ds = pd.DataFrame(np.random.rand(500, 4), columns =['latitude','longitude','depth','mag']) 
  
ds.plot.scatter(x ='latitude', y ='longitude') 
plt.show() 


# In[105]:


import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
  
series = pd.Series(3 * np.random.rand(4), 
  index =['latitude','longitude','depth','mag'], name ='series') 
  
series.plot.pie(figsize =(4, 4)) 
plt.show() 


# In[119]:


import matplotlib.pyplot as plt 
  
# defining labels 
activities = ['latitude','longitude','depth','mag'] 
  
# portion covered by each label 
slices = [3, 7, 8,10] 
  
# color for each label 
colors = ['r', 'y', 'g', 'b'] 
  
# plotting the pie chart 
plt.pie(slices, labels = activities, colors=colors,  
        startangle=90, shadow = True, explode = (0.1, 0.1, 0.1,0.1), 
        radius = 1.2, autopct = '%1.1f%%') 
  
# plotting legend 
plt.legend() 
  
# showing the plot 
plt.show() 


# In[ ]:




