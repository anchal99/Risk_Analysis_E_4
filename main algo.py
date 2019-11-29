#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tkinter import *
import tkinter.messagebox

def show():
    ds=pd.read_csv("Earthquake.csv")
    ds=ds[["time","latitude","longitude","depth","mag","magType","place"]]

    ds["time"]=[x.split("-")[1] for x in ds["time"]]
    maxlat=max(ds["latitude"])
    maxlong=max(ds["longitude"])
    minlat=min(ds["latitude"])
    minlong=min(ds["longitude"])
    ds["latitude"]=[((maxlat-x)/(maxlat-minlat)) for x in ds["latitude"]]
    ds["longitude"]=[((maxlong-x)/(maxlong-minlong)) for x in ds["longitude"]]


    x=ds[["time","latitude","longitude"]]
    y1=ds["depth"]
    month=int(Date.get())
    platitude=float(Lati.get())
    platitude=(abs(maxlat-platitude)/(maxlat-minlat))
    plongitude=float(Longi.get())
    plongitude=(abs(maxlong-plongitude)/(maxlong-minlong))

    Xpre=np.array([[month,platitude,plongitude]])

    regr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(x, y1,random_state=243,test_size = 0.15)
    regr.fit(X_train, y_train)
    predepth=regr.predict(Xpre)

    y2=ds["mag"]

    regr = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(x, y2,random_state=236,test_size = 0.15)
    regr.fit(X_train, y_train)
    Xpre=X_test.head(1)
    premag=regr.predict(Xpre)

    depth=max(ds["depth"])
    mag=max(ds["mag"])

    prob=(predepth/depth)*(premag/mag)
    prob=prob*100;
    text1="Probability of getting Earthquick : "+str(prob[0])+"%"
    Label(tk,text=text1).grid(row=8,column=3,padx=98)
    print(prob[0])
   
tk=Tk()
p1 = StringVar()
p2 = StringVar()
p3 = StringVar()

Label(tk).grid(row=0,column=0,padx=98,pady=20)
Label(tk).grid(row=0,column=1,padx=98)
Label(tk).grid(row=0,column=2,padx=98)
Label(tk).grid(row=0,column=3,padx=98)
label = Label ( tk, text="Month : ", font = 'Times 17 bold', height =1, width=0)
label.grid(row=1, column=1,sticky=W,ipadx=10,ipady=0)
Date = Entry(tk, textvariable=p1, bd=5)
Date.grid(row=1, column=2, sticky=W,ipadx=10)


Label(tk).grid(row=2,column=0,padx=98,pady=10)
Label(tk).grid(row=2,column=1,padx=98)
Label(tk).grid(row=2,column=2,padx=98)
Label(tk).grid(row=2,column=3,padx=98)
label = Label ( tk, text="Latitude : ", font = 'Times 17 bold', height =1, width=0)
label.grid(row=3, column=1,sticky=W,ipadx=10)
Lati = Entry(tk, textvariable=p2, bd=5)
Lati.grid(row=3, column=2, sticky=W,ipadx=10)


Label(tk).grid(row=4,column=0,padx=98,pady=10)
Label(tk).grid(row=4,column=1,padx=98)
Label(tk).grid(row=4,column=2,padx=98)
Label(tk).grid(row=4,column=3,padx=98)
label = Label ( tk, text="Longitute : ", font = 'Times 17 bold', height =1, width=0)
label.grid(row=5, column=1,sticky=W,ipadx=10)
Longi = Entry(tk, textvariable=p3, bd=5)
Longi.grid(row=5, column=2, sticky=W,ipadx=10)


Label(tk).grid(row=6,column=0,padx=98,pady=10)
Label(tk).grid(row=6,column=1,padx=98)
Label(tk).grid(row=6,column=2,padx=98)
Label(tk).grid(row=6,column=3,padx=98)
Button(tk,text="Result",command=show).grid(row=7,column=2,sticky=W,ipadx=10)

Label(tk).grid(row=9,column=0,padx=98,pady=20)
Label(tk).grid(row=9,column=1,padx=98)
Label(tk).grid(row=9,column=2,padx=98)
Label(tk).grid(row=9,column=3,padx=98)



tk.mainloop()

