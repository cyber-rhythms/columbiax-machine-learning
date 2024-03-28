#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5

# Implement function here
def PMF(train_data):
    
    users = int(np.nanmax(train_data[:, 0]))
    objects = int(np.nanmax(train_data[:, 1]))
    iterations = 50
    
    L =  np.zeros((iterations, 1))
    U_t = np.zeros((iterations, users, d))
    V_t = np.zeros((iterations, objects, d))
    
    np.random.seed(21)
    mean = np.zeros(d)
    covariance = (1/lam) * np.eye(d)

    V_init = np.random.multivariate_normal(mean, covariance, objects)
    U_init = np.zeros((users, d))
    
    U_t[iterations - 1, :, :] = U_init
    V_t[iterations - 1, :, :] = V_init
    
    M = np.zeros((users, objects))
    
    for i in range(users):
        for j in range(objects):
            if train_data[(train_data[:, 0] == i + 1) & (train_data[:, 1] == j + 1)][:, 2].size > 0:
                M[i, j] = train_data[(train_data[:, 0] == i + 1) & (train_data[:, 1] == j + 1)][:, 2].item()
            else:
                M[i, j] = np.nan
    
    for t in range(iterations):
        for i in range(users):
            for j in range(objects):
                if np.isfinite(M[i, j]) == True:
                    init = lam * sigma2 * np.eye(d)
                    init2 = np.zeros(d)
                    ui_comp1 = init + np.outer(V_t[t - 1, j, :], V_t[t - 1, j, :])
                    ui_comp2 = init2 + (M[i, j] * V_t[t - 1, j, :])
                    U_t[t, i, :] = np.dot(np.linalg.inv(ui_comp1), ui_comp2)
            
        for j in range(objects):
            for i in range(users):
                if np.isfinite(M[i, j]) == True:
                    init = lam * sigma2 * np.eye(d)
                    init2 = np.zeros(d)
                    vj_comp1 = init + np.outer(U_t[t, i, :], U_t[t, i, :])
                    vj_comp2 = init2 + (M[i, j] * U_t[t, i, :])
                    V_t[t, j, :] = np.dot(np.linalg.inv(vj_comp1), vj_comp2)
        Lt_u = 0
        Lt_e = 0
        Lt_v = 0
    
        for i in range(users):
            Lt_u += (lam / 2) *(np.linalg.norm(U_t[t, i, :]) ** 2)
            for j in range(objects):
                if np.isfinite(M[i, j]) == True:
                    Lt_e += (1 / (2 * sigma2)) * ((M[i, j] - np.dot(U_t[t, i, :].T, V_t[t, j, :])) ** 2)

        for j in range(objects):
            Lt_v += (lam / 2) * (np.linalg.norm(V_t[t, j, :]) ** 2)
    
        L[t] = -(Lt_u + Lt_e + Lt_v)

    return L, U_t, V_t

# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")


# In[ ]:





# In[ ]:




