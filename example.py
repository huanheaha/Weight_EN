# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 09:49:34 2022

@author: hehuan
"""

import numpy as np 
import pandas as pd
from numpy.random import shuffle  
from weight_elasticnet.Weight_EN import LogisticRegElastic
from sklearn.metrics import roc_auc_score

#%%
#example
def Calc_prob(y):
    p = np.exp(5*y) / (1 + np.exp(5*y))
    return p

n_samples,n_features=500,1000 
m = 20  # network number
np.random.seed(80) 
X = np.random.normal(size=[n_samples,n_features])
w1 = np.zeros(int(n_features/m))
for i in range(10):
    w1[i] = 3
w2 = np.zeros(int(n_features/m))
for i in range(10):
    w2[i] = 2
w3 = np.zeros(int(n_features/m))
for i in range(10):
    w3[i] = 1       
siga = np.random.randn(n_samples)
y =  X[:,:50].dot(w1) + X[:,50:100].dot(w2) + X[:,100:150].dot(w3)+ 0.1*siga
w = np.hstack((np.hstack((np.hstack((w1,w2)),w3)),np.zeros(850)))
p = Calc_prob(y)

    
idx_pos = []
idx_neg = []
for i in range(len(p)):
    if p[i] > 0.5:
        idx_pos.append(i)
    else:
        idx_neg.append(i)
     
X1 = np.vstack((X[idx_pos,:],X[idx_neg,:])) 
np.random.seed(1)
shuffle(X1[:250]); shuffle(X1[250:])
Y = np.hstack((np.ones(250),np.zeros(250)))

X_train = X1[100:400]; Y_train = Y[100:400]
X_test = np.vstack((X1[:100,:],X1[400:,:]))# test set, 200 samples 
Y_test = np.hstack((Y[:100],Y[400:]))

matrix = np.zeros((X_train.shape[0], X_train.shape[1] + 2))
'''inpute training matrix'''
matrix[:,:-2] = X_train
matrix[:, -2] = np.ones(X_train.shape[0])
matrix[:, -1] = Y_train

# feature weight
Z = np.zeros((1000,20))
'''for the gene j in pathway m we set the value 
Zjm to be equal to one and zero otherwise'''
for i in range(20):
    Z[50*i:50*(i+1),i] = np.ones(50)

imp = np.ones((20,1000)) 
'''The importance of feature in each group, 
 and the value of each row in the matrix 
 is set to 1 to indicate that each information source is equally important'''
ii = np.ones((3,1000))+np.ones((3,1000))
imp[:3,:] = ii 
'''The value size of each row in the matrix is different, 
indicating that each information source is of different importance '''

Score = np.matrix(Z) * np.matrix(imp)

z_dj = [ ]
for i in range(len(Score)):
    z_dj.append(Score[i,i])
    
zz = pd.DataFrame(z_dj)
zz['rank'] = pd.DataFrame(z_dj).rank(ascending=True, method='dense')
zz['1/rank'] = 1/zz['rank']
wj = np.hstack((1,np.array(zz['1/rank'])))    

#%%
#training 

logit = LogisticRegElastic()
alpha = 0.5
lambd = 0.05
coef_path = logit.fit(matrix, alpha, wj, precision = 0.0001,
                          lambda_grid =[lambd], pyspark=False)
'''
matrix: inpute training matrix
alpha: α∈[0,1] which determines the relative weight of the l_1 and l_2 norm.
wj: feature weight
lambd: regularization parameter
'''
coef_ = logit.coef_
'''coef_: Estimation coefficient of weight elastic net.'''

y_pre = logit.predict(X_test, pyspark=False)
'''y_pre: prediction results of independent test set'''

Auc = roc_auc_score(Y_test,y_pre)
'''AUC: prediction AUC of independent test set'''












 
    
