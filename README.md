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
from weight_elasticnet.Weight_EN import LogisticRegElastic
logit = LogisticRegElastic()
coef_path = logit.fit(matrix, alpha, wj, precision = 0.0001,
                          lambda_grid =[lambd], pyspark=False)
"""matrix: inpute training matrix
alpha: α∈[0,1] which determines the relative weight of the l_1 and l_2 norm.
wj: feature weight
lambd: regularization parameter
"""                     
y_pre = logit.predict(X_test, pyspark=False)
"""y_pre: prediction probability of independent test set
"""  
```
print(y_pre[95:105])
[0.8692451  0.68239113 0.72763984 0.68313643 0.51682172 0.16294713
 0.17701993 0.77651004 0.52424665 0.55346663]
```
