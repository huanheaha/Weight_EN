# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 10:54:43 2022

@author: hehuan
"""

#
import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
zz = pd.read_csv('gene_weight.csv',sep=',')
'''
zz:feature weight matrix, i.e. [degree,rank,1/rank]
'''

thca = pd.read_csv('THCA_kegg_path.csv',sep =',',index_col = 'gene_name')
data_scaler = preprocessing.scale(np.array(thca).T) #标准化后的数据
y = np.hstack((np.ones(70),np.zeros(58))) #70 postive samples，58 negtive samples
np.random.seed(1)
np.random.shuffle(data_scaler[:70])  
np.random.shuffle(data_scaler[70:])

X_train = data_scaler[30:110,:]; Y_train = y[30:110]
X_test = np.vstack((data_scaler[:30,:],data_scaler[110:,:])) #independent test set
Y_test = np.hstack((y[:30],y[110:]))
matrix = np.zeros((X_train.shape[0], X_train.shape[1] + 2))
matrix[:,:-2] = X_train; matrix[:, -2] = np.ones(X_train.shape[0])
matrix[:, -1] = Y_train
'''
matrix: inpute training matrix
'''

model1 = joblib.load(filename="THCA_best.model")
'''model1:the model has been trained by THCA training data'''
coef_ = model1.coef_
'''coef_: Estimation coefficient of weight elastic net.'''
y_pre = model1.predict(X_test, pyspark=False)
'''y_pre: prediction results of independent test set'''

