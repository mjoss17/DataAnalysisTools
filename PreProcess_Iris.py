"""
PreProcess_Iris.py 
This is the PreProcessing File for the iris dataset. This 
Dataset can then be accessible through the methods 
get_split_dataset
get_all_dataset

The get_all_dataset() will return the data through two pandas
Dataframes, one with all of the feature variables, the tops of the
columns are the names of the feature, and the other DataFrame is a
single column labeled targets

The get_split_dataset() will return the data through four pandas
DataFrames after splitting the iris data into training and test samples.
The parameters for this split can be manipulated within the function if
needed. 

Commented out are a series of composite/manipulated datasets.
(yes they're ugly, no guarantee they'll work)
"""
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Single_Models as a
import random
random.seed(123)
from tensorflow import set_random_seed
set_random_seed(123)


from subprocess import check_output
from math import log
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

log = []

iris = pd.read_csv("Iris.csv")
iris.drop('Id', axis = 1, inplace = True)
iris.drop(0, axis = 0, inplace=True)
paramsFirst = iris.iloc[:, 0:-1]
labelsFirst = iris.iloc[:, -1]
df = pd.DataFrame(paramsFirst)
df.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
df['target'] = labelsFirst
params = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
labels = df[['target']]


#Creating PCA Variables
scaler = StandardScaler() 
scaler.fit(params)
st_data = scaler.transform(params)
pca_model = PCA(0.95)
pca_model.fit(st_data)
pca_data = pca_model.transform(st_data)

ex_variance = np.var(pca_data, axis=0)
ex_variance_ratio = ex_variance / np.sum(ex_variance)

pca_df = pd.DataFrame()
pca_df['V1'] = pca_data[:,0]
pca_df['V2'] = pca_data[:,1]
pca_df['Species'] = iris[['Species']]
params_pca = pca_df[['V1', 'V2']]

# params_log = np.log(params)
#Creating Log PCA variables
# scaler.fit(params_log)
# st_data_log = scaler.transform(params_log)
# pca_model_log = PCA(0.95)
# pca_model_log.fit(st_data)
# pca_data_log = pca_model.transform(st_data)
# ex_variance_log = np.var(pca_data, axis=0)
# ex_variance_ratio_log = ex_variance / np.sum(ex_variance)
# pca_df_log = pd.DataFrame()
# pca_df_log['V1'] = pca_data_log[:,0]
# pca_df_log['V2'] = pca_data_log[:,1]
# pca_df_log['Species'] = iris[['Species']]
# params_pca_log = pca_df_log[['V1', 'V2']]


def get_split_data():
    train_X, test_X, train_y, test_y = train_test_split(params, labels, test_size=0.33, stratify=labels)
    return (train_X, train_y, test_X, test_y)

def get_all_data():
    return df, params, labels