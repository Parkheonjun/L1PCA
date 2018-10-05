# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 00:43:29 2018

@author: Parkheonjun

@contents : Find the L1-norm Principal Components via exhaustive search
            Complexity = O(2^nk)
        
@input : numpy array type data matrix X : p times n, where n is the number of observations
                                          and p is the number of covariates.

@output : Loading matrix P : p times k orthogonal matrix, where k is the number of components.

@Reference : L1-norm Principal Component Analysis (Wikipedia)

"""

import numpy as np
import numpy.linalg as LA

def NextComb(l):    
    '''
    마지막 1인 친구가 끝일때는 -1로 바뀜
    마지막 1인 친구가 끝이 아닐때는 이 친구보다 오른쪽이 모두 -1이기때문에 -1 1 1 1 1 로 바꿈
    Example.
    l = np.array([1,1,1,1,1,1])
    k = 0
    while 1>0:
        if sum(l == np.array([-1,-1,-1,-1,-1,-1]))==6:
            k=k+1
            break;
        else:
            l = NextComb(l)
            k = k+1
    print(k)  ## return 2^6 = 64
    
    '''
    Last_one = np.where(l == 1)[0][-1]
    if Last_one == len(l)-1:
        l[-1] = -1
        return(l)
    else:
        l[Last_one:]=-1*l[Last_one:]
        return(l)
    

    
def BNM(X, k):
    '''
    @ input
        X : p times n data matrix
        k : the number of components
        
    @ output
        B : n times k Binary matrix which maximizes ||XB||_*^2.
    '''
    n = X.shape[1]
    tempB  = np.array([1]*(n*k))
    sum_sv = 0
    Last_B = np.array([-1]*(n*k))
    
    while sum(tempB == Last_B) != (n*k):
        tempB.shape = (n,k)
        temp_sum_sv = sum(LA.svd(np.dot(X,tempB))[1])
        if sum_sv < temp_sum_sv:
            sum_sv = temp_sum_sv
            B = tempB.copy()
        tempB.shape= n*k
        tempB = NextComb(tempB)
    
    return(B)


def L1pca(X,k,center = "median"):
    '''
    @ input
        X : p times n data matrix
        k : the number of components
        center : "median" or "mean"
                 centering with median or mean
    @ output [Y, P]
        Y : k times n principal components
        P : p times k orthogonal loading matirx 
    '''
    p = X.shape[0]
   
    if center =="median":
        median = np.median(X,axis=1)
        median.shape = (p,1)
        X = X- median # centering
    elif center == "mean":
        mean = np.mean(X,axis=1)
        mean.shape = (p,1)
        X = X- mean # centering
    else:
        print("center should be either 'median' or 'mean'.")
        return()

    B_BNM= BNM(X,k)
    U, D, Vt= LA.svd(np.dot(X, B_BNM))
    P = np.dot(U[:,:k], Vt)
    Y = np.dot(np.transpose(P), X)
    
    return(Y,P)
    


def L2pca(X,k):  # L2 PCA    
    p = X.shape[0]
    mean = np.mean(X,axis=1)
    mean.shape = (p,1)
    X = X-mean
    Vt= LA.svd(np.transpose(X))[2]
    P = np.transpose(Vt)[:,:k]
    Y = np.dot(np.transpose(P),X)
    
    return(Y,P)
    