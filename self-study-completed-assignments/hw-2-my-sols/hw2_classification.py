#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

## can make more functions if required

def pluginClassifier(X_train, y_train, X_test):    
  # this function returns the required output                             
    
    N_observations = X_train.shape[0]       # # Dimensionality, classes, observation variables control
    K_classes = np.max(y_train).astype(int) + 1
    D_dimensions = X_train.shape[1]
    
                                            # Class observations/priors placeholders

    class_obs = np.zeros((1, K_classes))    # Generate class observations vector with all elements set to 0
    class_priors = np.zeros((1, K_classes)) # Generate class prior vector, with all elements set to 0
    
    
    for i in range(K_classes):                      # Maximum likelihood estimates of class prior probabilities and observ.
        class_obs[:, i] = np.count_nonzero(y_train == i)
        class_priors[:, i] = class_obs[:, i] / N_observations
    
    mu = np.zeros((K_classes, D_dimensions)) # (K x D) placeholder array to store mean vectors of shape (1, D) for each K class

    for j in range(K_classes):              # Maxmimum likelihood estimates of mean vectors for each class                   
        for i in range(D_dimensions):
            mu[j,i] = np.mean(X_train[(y_train == j)][:,i])
            
    covariance = np.zeros((K_classes, D_dimensions, D_dimensions)) 
    final_covariance = np.zeros((K_classes, D_dimensions, D_dimensions)) # (K x D x D) placeholder array to store covariance matrices 
                                                                         # of shape (D, D) for each K class
    for j in range(K_classes):                                           # Maximum likelihood estmiates of covariance matrices for each class
        for i in range(X_train[y_train == j].shape[0]): 
            centred_xi = X_train[y_train == j][i, :] - mu[j, :]
            covariance[j, :, :] = covariance[j, :, :] + np.outer(centred_xi, centred_xi)
        final_covariance[j, :, :] = covariance[j, :, :] / class_obs[:, j]
        
    pd_product = np.zeros((X_test.shape[0], K_classes))
    post_prob = np.zeros((X_test.shape[0], K_classes))                  # (M x K) placeholder array to store posterior probabilities of M test points for K classes
    for i in range(X_test.shape[0]):                                    # Posterior probabilities calculation
        for k in range(K_classes):
            a = np.sqrt(np.linalg.det(final_covariance[k, :, :]))
            b = X_test.shape[1]
            norm = 1 / (((np.sqrt(2 * np.pi)) ** b) * a)
            centred_xi = X_test[i, :] - mu[k, :]
            exp_quad = np.exp(-0.5 * (np.dot(np.dot(centred_xi.T, np.linalg.inv(final_covariance[k, :, :])), centred_xi)))
            pd_product[i, k] = norm * exp_quad * class_priors[0, k]
        post_prob[i, :] = pd_product[i, :] / np.sum(pd_product[i, :])
    
    return post_prob

final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file

