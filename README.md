# Weight_EN
* Weight_elasticnet uses a statistic regularization method by integrating structure information and overlap of multiple pathways for analysis of high-dimensional gene
expression data.
## Installation
* It is required to install the following dependencies in order to be able to run the package: For the weighted elastic net method of the package named weight_elasticnet
```
python >= 3.5
numpy = 1.18.4
pandas = 0.23.4
joblib = 1.0.0
scipy >= 1.4.1
scikit-learn >= 0.23.2
```
You can also clone the repository and do a manual install.
```
git clone https://github.com/huanheaha/Weight_EN.git
python setup.py install
```
# Running the Code
* Acquire all the data and code in Weight_EN to the local address.
* Open Python editor,then dictory to Weight_EN folder which contains example.py. 
* Make sure the data and code are in one folder,or enter the exact data address when you run the code.
## **An example of applying the Stacked SGL is provided in example.py.** <br>
Running the example in example.py. Specific example including parameter description and parameter selection is provided in this file.
```
matrix: inpute training matrix
alpha: α∈[0,1] which determines the relative weight of the l_1 and l_2 norm.
wj: feature weight
lambd: regularization parameter
"""
from weight_elasticnet.Weight_EN import LogisticRegElastic
logit = LogisticRegElastic()
coef_path = logit.fit(matrix, alpha, wj, precision = 0.0001,
                          lambda_grid =[lambd], pyspark=False)
""" Data: P of the cross-validation set  probability matrix outputing 
           on the  base learner. n_train * T_base 
    Data_test: representing the independent test set probability matrix outputing 
              on the base learner. n_test * T_base
"""                     
y_pre = logit.predict(X_test, pyspark=False)
"""y_pre: prediction probability of independent test set"""  
```

