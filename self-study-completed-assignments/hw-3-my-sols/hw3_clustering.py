#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy as sp
import sys

X = np.genfromtxt(sys.argv[1], delimiter = ",")

def KMeans(data):
	#perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively
    K = 5                           
    iterations = 10
    N = data.shape[0]
    d = data.shape[1]
    
    MU_0 = data[np.random.choice(N, K, replace=False), :] #Random initialise 5 cluster centroids (no random seed)
    
    MU_t = np.zeros((iterations, K, d))
    C_t = np.zeros((iterations, N))
    
    MU_t[iterations - 1, :, :] = MU_0 
    
    for t in range(iterations):          
        xi_distance_k = np.zeros((1, K)) 
        for i in range(N):               
            for k in range(K):           
                xi_distance_k[:, k] = np.linalg.norm(data[i, :] - MU_t[t - 1, k, :]) ** 2   
                C_t[t, i] = np.argmin(xi_distance_k)                                    
        for j in range(K):                   
            for i in range(d):
                MU_t[t, j, i] = np.mean(data[(C_t[t, :] == j)][:,i])                        
        
        filename = "centroids-" + str(t+1) + ".csv" #"i" would be each iteration
        np.savetxt(filename, MU_t[t, :, :], delimiter=",")
  
def EMGMM(data):
    
    iterations = 10
    K = 5
    N = data.shape[0]
    d = data.shape[1]
    
    pi_t = np.zeros((iterations, K))
    mu_t = np.zeros((iterations, K, d))
    covariance_t = np.zeros((iterations, K, d, d))
    posterior_t = np.zeros((iterations, N, K))
    
    pi_0 = np.zeros(K)
    for k in range(K):
        pi_0[k] = 1 / K

    mu_0 = data[np.random.choice(N, K, replace=False), :]
    
    covariance_0 = np.eye(d)
    covariance_k_0 = np.zeros((K, d, d))
    for k in range(K):
        covariance_k_0[k, :, :] = covariance_0
        
    posterior_0 = np.zeros((N, K))
    
    pi_t[iterations - 1, :] = pi_0
    mu_t[iterations - 1, :, :] = mu_0
    covariance_t[iterations - 1, :, :, :] = covariance_k_0
    
    for t in range(iterations):
        pd_product = np.zeros((N, K))
        for i in range(N):
            for k in range(K):
                a = np.sqrt(np.linalg.det(covariance_t[t - 1, k, :, :]))
                b = data.shape[1]
                norm = 1 / (((np.sqrt(2 * np.pi)) ** b) * a)
                centred_xi = data[i, :] - mu_t[t - 1, k, :]
                exp_quad = np.exp(-0.5 * (np.dot(np.dot(centred_xi.T, np.linalg.inv(covariance_t[t - 1, k, :, :])), centred_xi)))
                pd_product[i, k] = norm * exp_quad * pi_t[t - 1, k]
            posterior_t[t - 1, i, :] = pd_product[i, :] / np.sum(pd_product[i, :])

        n_k = np.sum(posterior_t[t - 1, :, :], axis=0)
        for k in range (K):
            pi_t[t, k] = n_k[k] / np.sum(n_k)

        for k in range(K):
            for i in range(N):
                mu_t[t, k, :] = mu_t[t, k, :] + ((posterior_t[t - 1, i, k] * data[i, :]) / n_k[k])
        
        for k in range(K):
            for i in range(N):
                centred_xi = data[i, :] - mu_t[t, k, :]
                covariance_t[t, k, :, :] = covariance_t[t, k, :, :] + ((posterior_t[t - 1, i, k] * (np.outer(centred_xi, centred_xi))) / n_k[k])
                
            filename = "Sigma-" + str(k+1) + "-" + str(t+1) + ".csv" #this must be done 5 times (or the number of clusters) for each iteration
            np.savetxt(filename, covariance_t[t, k, :, :], delimiter=",")
    
        filename = "pi-" + str(t+1) + ".csv" 
        np.savetxt(filename, pi_t[t, :], delimiter=",") 
        filename = "mu-" + str(t+1) + ".csv"
        np.savetxt(filename, mu_t[t, :, :], delimiter=",")

KMeans(X)
EMGMM(X)

