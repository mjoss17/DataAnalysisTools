"""
PreProcess_Glass.py 
This is the PreProcessing File for the glass dataset. This 
Dataset can then be accessible through the methods 
get_split_dataset
get_all_dataset

The get_all_dataset() will return the data through two pandas
Dataframes, one with all of the feature variables, the tops of the
columns are the names of the feature, and the other DataFrame is a
single column labeled targets

The get_split_dataset() will return the data through four pandas
DataFrames after splitting the glass data into training and test samples.
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
import random
import collections
import datetime
random.seed(123)
import tensorflow as tf
tf.compat.v1.set_random_seed(123)
import math

from subprocess import check_output
from math import log
from sklearn.utils import resample
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
from sklearn.decomposition import FastICA


log = []

db = pd.read_csv("glass.data", header=None)
db.drop(db.columns[0], axis = 1, inplace = True)
paramsFirst = db.iloc[:, 0:-1]
labelsFirst = db.iloc[:, -1]

names = np.array(["b_w_f_p", "b_w_N_f_p", "v_w_f_p", "Containers", "Table", "Head"])
namedict = {1:"b_w_f_p", 2:"b_w_N_f_p", 3:"v_w_f_p", 5:"Containers", 6:"Table", 7:"Head"}

feature_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
bin_names =  ["Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
ica_names = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9"]
pca_names = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6']
lda_names = ['LDA1', 'LDA2', 'LDA3', 'LDA4', 'LDA5', 'LDA6']

#Creating Originial Model 'df' with params and labels
df = pd.DataFrame(paramsFirst)
df.columns = feature_names
df['target'] = labelsFirst
params_og = df[["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]]
labels = df["target"]

# eda.multivariate("TESTING_EDA_glass_M.pdf", df, params_og, labels, "Glass")
# eda.bivariate("TESTING_EDA_glass_B.pdf", df, params_og, labels, "Glass")
# eda.univariate("TESTING_EDA_glass_U.pdf", df, params_og, labels, "Glass")

# #Creating PCA Model 'pca_df' with params_pca
# scaler = StandardScaler() 
# st_data = scaler.fit_transform(params_og)
# pca_model = PCA(.90)
# pca_model.fit(st_data)
# pca_data = pca_model.transform(st_data)
# ex_variance = np.var(pca_data, axis=0)
# ex_variance_ratio = ex_variance / np.sum(ex_variance)
# pca_df = pd.DataFrame()
# pca_df = pd.DataFrame(pca_data, columns=pca_names)
# pca_df['target'] = labels
# params_pca = pca_df.iloc[:, 0:-1]

# #Creating ICA Model 'ica_df' with params_ica
# ica = FastICA(random_state=123)
# S_ica_ = ica.fit(params_og).transform(params_og)  # Estimate the sources
# S_ica_ /= S_ica_.std(axis=0)
# ica_df = pd.DataFrame(S_ica_, columns=ica_names)
# ica_df['target'] = labels
# params_ica = ica_df.iloc[:, 0:-1]

# #Creating LDA Model 'lda_df' with params_lda
# lda = LinearDiscriminantAnalysis(n_components=6)
# lda_data = lda.fit(params_og, labels).transform(params_og) 
# lda_df = pd.DataFrame(lda_data, columns=lda_names)
# lda_df['target'] = labels
# params_lda = lda_df.iloc[:, 0:-1]

# #Creating Bin Model 'bin_df' with params_bin
# bin_df = df.copy()
# bin_Ba = []
# bin_Fe = []
# bin_Mg = []
# for val in df['Ba']:
#     if val == 0: bin_Ba.append(0)
#     else: bin_Ba.append(1)
# for val in df['Fe']:
#     if val == 0: bin_Fe.append(0)
#     else: bin_Fe.append(1)
# for val in df['Mg']:
#     if val == 0: bin_Mg.append(0)
#     elif val < 2: bin_Mg.append(1)
#     elif val < 3: bin_Mg.append(2)
#     else: bin_Mg.append(3)
# updates = pd.DataFrame()
# updates['Ba'] = bin_Ba
# updates['Fe'] = bin_Fe
# updates['Mg'] = bin_Mg
# bin_df.update(updates)
# del bin_df['RI']
# params_bin = bin_df.iloc[:, 0:-1]
# log.append("Binned: Ba and Fe are binned into 0 vs non-zero, MG is binned into <[1,2,3,5], RI is removed")

# #Creating Original Model with PCA1 PCA2 h1 h2
# mod1_df = pd.DataFrame(params_bin, columns=bin_names)
# mod1_df = pd.concat([mod1_df, lda_df[['LDA1','LDA2','LDA3','LDA4','LDA5','LDA6']]], axis=1)
# mod1_df['target'] = labels
# params_mod1 = mod1_df.iloc[:, 0:-1]
# log.append("MOD 1: Binned data with all 6 LDA features as well")

# #Creating Original Model with PCA1 PCA2 h1 h2
# mod2_df = pd.DataFrame(params_bin)
# mod2_df = pd.concat([mod2_df, lda_df[['LDA1', 'LDA2', 'LDA3']]], axis=1)
# mod2_df = pd.concat([mod2_df, pca_df[['PCA1', 'PCA2']]], axis=1)
# mod2_df['target'] = labels
# params_mod2 = mod2_df.iloc[:, 0:-1]
# log.append("MOD 2: Binned data as well as [LDA1, LDA2, LDA3] and [PCA1, PCA2]")

# #Creating model with PCA and LDA data
# mod3_df = pd.DataFrame(params_og, columns=feature_names)
# mod3_df = pd.concat([mod3_df, params_pca[['PCA1', 'PCA2']]], axis=1)
# mod3_df['target'] = labels
# params_mod3 = mod3_df.iloc[:, 0:-1]
# log.append("MOD 3: All the Original features plus [PCA1, PCA2]")

## Creating Subsets of Data
# testvalues1 = [1, 2, 6]
# testvalues2 = [1,2,3]
# testvalues3 = [7, 5, 1]
# l1 = df[df.target.isin(testvalues1)]
# l2 = df[df.target.isin(testvalues2)]
# l3 = df[df.target.isin(testvalues3)]
## Testing Models on Subsets
# print("1, 2, 6", a.decisionTree(l1[["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]], l1['target']))
# print("1, 2, and 3", a.decisionTree(l2[["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]], l2['target']))
# print("7, 5, 1", a.decisionTree(l3[["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]], l3['target']))

# # For creating and running models on subsets
# testvalues = [1,2,3]
# modData = df[df.target.isin(testvalues)]
# modLabels = modData['target']
# modParams = modData[feature_names]
# # st_data2 = scaler.fit_transform(modParams)
# print("shape, " + str(modParams.shape))
# train_Xm, test_Xm, train_ym, test_ym = train_test_split(modParams, modLabels, test_size=0.3)
# log.append('_MOD Data_')
# log.append(a.neuralNetwork(train_Xm, train_ym, test_Xm, test_ym, pdf_pages, 3, title="Neural Net Confusion Matrix Modified Data"))
# print('\n'.join(log))

# log.append('\n')


def get_split_data():
    train_X, test_X, train_y, test_y = train_test_split(params_og, labels, test_size=0.3, stratify=l)
    return (train_X, train_y, test_X, test_y)

def get_all_data():
    return df, params_og, labels