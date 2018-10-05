# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 03:00:24 2018

@author: Parkheonjun
"""


import L1pca
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1,1],[2,2],[3,3],[4,4],[0,0],[-1,-1],[-2,-2],[10,60]])
X = np.transpose(X)


P_l1 = L1pca.L1pca(X,1,center="median")[1]  # loading vector via L1 PCA
P_l1_mean = L1pca.L1pca(X,1,center="mean")[1] 
P_l1_error = L1pca.L1pca(X,1,center= "abc")
P_l2 = L1pca.L2pca(X,1)[1]
mean = np.mean(X, axis = 1)
mean.shape = (2,1)
median = np.median(X,axis=1)
median.shape = (2,1)

plt.scatter(X[0,:],X[1,:])
for i in  np.linspace(0,3,1000):
    temp = median + P_l1*i    
    plt.scatter(temp[0,],temp[1,],s=1, c = "black")
    
    temp = mean + P_l1_mean*i    
    plt.scatter(temp[0,],temp[1,],s=1, c = "blue")
    
    temp = mean + P_l2*i    
    plt.scatter(temp[0,],temp[1,],s=1, c = "red")
    
