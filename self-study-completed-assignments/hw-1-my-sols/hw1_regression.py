#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1(X, y):
    
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    
    w_rr1 = np.linalg.inv((lambda_input * np.eye(X.shape[1])) + X.T.dot(X))
    w_rr2 = X.T.dot(y)
    
    w_rr = w_rr1.dot(w_rr2)
    
    return w_rr

wRR = part1(X_train, y_train)  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file

## Solution for Part 2
def part2(Xtrain, y_train, Xtest):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    indices = []
    iteration = 10
    X_test_copy = Xtest

    post_cov2 = Xtrain.T.dot(Xtrain)

    predictive_variance_t = np.zeros((iteration, X_test_copy.shape[0]))
    post_cov_t = np.zeros((iteration, Xtrain.shape[1], Xtrain.shape[1]))

    x_max = np.zeros(Xtest[0, :].shape)
    
    
    for t in range(iteration):
        
        post_cov1 = lambda_input * np.eye((Xtrain.shape[1]))
        post_cov2 = post_cov2 + np.outer(x_max,x_max)
        post_cov3 = (sigma2_input **  -1) * post_cov2
        post_cov_t[t, :, :] = np.linalg.inv(post_cov1 + post_cov3)
        
        for i in range(X_test_copy.shape[0]):
            predictive_variance_t[t, i] = sigma2_input + X_test_copy[i, :].T.dot(post_cov_t[t, :, :]).dot(X_test_copy[i, :])
        
        for i in range(X_test.shape[0]):
            if sigma2_input + Xtest[i, :].T.dot(post_cov_t[t, :, :]).dot(Xtest[i, :]) == max(predictive_variance_t[t, :]):
                index = i
                indices.append(index + 1)
                x_max = Xtest[index, :]
                X_test_copy = np.delete(X_test_copy, obj=np.argmax(predictive_variance_t[t, :]), axis=0)

    return indices

active = part2(X_train, y_train, X_test)  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file

