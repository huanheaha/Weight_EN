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

"""
y_pre, coef, Auc, Acc = S_S.Pre_MRL(Data, Data_test)  
""" y_pre : classification results of independent test set  
"""      
```
