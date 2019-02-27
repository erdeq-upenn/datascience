#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:26:43 2019

@author: Dequan Er
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

digits = load_digits()
digits.images.shape

fig,axes =  plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[],'yticks':\
                         []},gridspec_kw=dict(hspace = 0.1,wspace=0.1))
for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap='binary',interpolation='nearest')
    ax.text(0.05,0.05,str(digits.target[i]),transform=ax.transAxes,color='green')
    #
X = digits.data
y = digits.target

iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
data_projected.shape
plt.show()
plt.figure()
plt.scatter(data_projected[:,0],data_projected[:,1],c=digits.target,edgecolor='none',
            alpha=0.5, cmap=plt.cm.get_cmap('viridis',10))
plt.colorbar(label = 'digit label',ticks = range(10))
plt.clim(-0.5,9.5)
plt.show()
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state =0)
model = GaussianNB()
model.fit(Xtrain,ytrain)
y_model = model.predict(Xtest)
print(accuracy_score(ytest,y_model))
plt.figure()
mat = confusion_matrix(ytest,y_model)
sns.heatmap(mat,square = True, annot = True, cbar = False)
plt.xlabel('predict value')
plt.ylabel('true value')
plt.show()

#fig,axes = plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[],'yticks'\
#                        :[]},gridspec_kw=dict(hspace=0.1,wspace=0.1))
#for i, ax in enumerate(axes.flat):
#    ax.imshow(digits.images[i],cmap='binary',interpolation='nearest')
#    ax.text(0.05,0.5,str(y_model[i]),transform=ax.transAexs,color='green' \
#            if (ytest[i]==y_model[i]) else 'red')

fig,axes = plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[],'yticks':\
                         []},gridspec_kw=dict(hspace = 0.1,wspace=0.1))
for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap='binary',interpolation='nearest')
    ax.text(0.05,0.05,str(y_model[i]),transform=ax.transAxes,color='green' if (ytest[i]==y_model[i]) else 'red')