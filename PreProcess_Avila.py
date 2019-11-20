"""
PreProcess_Avila.py 
This is the PreProcessing File for the avila dataset. This 
Dataset can then be accessible through the methods 
get_split_dataset()
get_all_dataset()

The get_all_dataset() will return the data through two pandas
Dataframes, one with all of the feature variables, the tops of the
columns are the names of the feature, and the other DataFrame is a
single column labeled targets

The get_split_dataset() will return the data through four pandas
DataFrames after splitting the avila data into training and test samples.
The parameters for this split can be manipulated within the function if
needed. 

Commented out are a series of composite/manipulated datasets.
(yes they're ugly, no guarantee they'll work)
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
import Single_Models as a
import random
random.seed(123)
import tensorflow as tf
tf.compat.v1.set_random_seed(123)
from scipy import stats

from subprocess import check_output
from math import log

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
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

log = []

avila_train = pd.read_csv("avila-tr.txt", header=None)
avila_test = pd.read_csv("avila-ts.txt", header=None)
avila = pd.concat([avila_test, avila_train], ignore_index=True)
paramsog = avila.iloc[:, 0:-1]
labelsog = avila.iloc[:, -1]

params_train = avila_train.iloc[:, 0:-1]
labels_train = avila_train.iloc[:, -1]
params_test = avila_test.iloc[:, 0:-1]
labels_test = avila_test.iloc[:, -1]


names = np.array(np.unique(labelsog))
feature_names = ["ic_dist", "up_mg", "lw_mg", "expl", "row_n", "mod_rat", "il_space", "weight", "peak_n", "mod_rat/il_space"]
ica_names = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"]


df = pd.DataFrame(paramsog)
df.columns = feature_names
df["target"] = labelsog
df_train = pd.DataFrame(params_train)
df_train.columns = feature_names
df_train["target"] = labels_train
df_test = pd.DataFrame(params_test)
df_test.columns = feature_names
df_test['target'] = labels_test

b = df[(df['target'] == 'B')]
df = df[(df['target'] != 'B')]
b_train = df_train[(df_train['target'] == 'B')]
df_train = df_train[(df_train['target'] != 'B')]
for feat in feature_names:
    q = df[feat].quantile(0.99)
    p = df[feat].quantile(0.01)
    df = df[(df[feat] < q)]
    df = df[(df[feat] > p)]
    
    q = df_train[feat].quantile(0.99)
    p = df_train[feat].quantile(0.01)    
    df_train = df_train[(df_train[feat] < q)]
    df_train = df_train[(df_train[feat] > p)]

df = pd.concat([b,df], ignore_index=True)
df_train = pd.concat([b_train, df_train], ignore_index=True)


params_og = df[["ic_dist", "up_mg", "lw_mg", "expl", "row_n", "mod_rat", "il_space", "weight", "peak_n", "mod_rat/il_space"]]
params_train = df_train[["ic_dist", "up_mg", "lw_mg", "expl", "row_n", "mod_rat", "il_space", "weight", "peak_n", "mod_rat/il_space"]]
params_test = df_test[["ic_dist", "up_mg", "lw_mg", "expl", "row_n", "mod_rat", "il_space", "weight", "peak_n", "mod_rat/il_space"]]
labels = df["target"]
labels_train = df_train['target']
labels_test = df_test['target']

# eda.multivariate("TESTING_EDA_avila_M.pdf", df_train, params_train, labels_train, "Avila")
# eda.bivariate("TESTING_EDA_avila_B.pdf", df_train, params_train, labels_train, "Avila")
# eda.univariate("TESTING_EDA_avila_U.pdf", df_train, params_train, labels_train, "Avila")


# #Creatng PCA Model "pca_df" with params_pca
# scaler = StandardScaler() 
# st_data = scaler.fit_transform(params_train)
# st_data_test = scaler.fit_transform(params_test)
# pca_model = PCA(.90)
# pca_model_test =  PCA(.90)
# pca_model.fit(st_data)
# pca_model_test.fit(st_data_test)
# # print("pca variance explination: ", pca_model.explained_variance_ratio_)
# pca_data = pca_model.transform(st_data)
# pca_data_test = pca_model_test.transform(st_data_test)
# # print("PCA shape ", pca_data.shape)
# ex_variance = np.var(pca_data, axis=0)
# ex_variance_ratio = ex_variance / np.sum(ex_variance)
# # pca_df = pd.DataFrame()
# # pca_df_test = pd.DataFrame()
# pca_names = ["PCA"+str(i) for i in range(0,len(pca_data[0,:]))]
# pca_df = pd.DataFrame(pca_data, columns=pca_names)
# pca_df_test = pd.DataFrame(pca_data_test, columns=pca_names)
# pca_df['target'] = labels_train
# pca_df_test['target'] = labels_test
# params_pca_train = pca_df.iloc[:, 0:-1]
# params_pca_test = pca_df_test.iloc[:, 0:-1]

# #Creating ICA Model "ica_df" with params_ica
# ica = FastICA(random_state=123)
# ica_test = FastICA(random_state=123)
# S_ica = ica.fit_transform(params_train)  # Estimate the sources
# S_ica_test = ica.fit_transform(params_train)
# S_ica /= S_ica.std(axis=0)
# S_ica_test /= S_ica_test.std(axis=0)
# ica_df = pd.DataFrame(S_ica, columns=ica_names)
# ica_df_test = pd.DataFrame(S_ica, columns=ica_names)
# ica_df['target'] = labels_train
# ica_df_test['target'] = labels_test
# params_ica = ica_df.iloc[:, 0:-1]
# params_ica_test = ica_df_test.iloc[:, 0:-1]


# #Creating LDA Model "lda_df" with params_lda
# lda = LinearDiscriminantAnalysis(n_components=6)
# lda_test = LinearDiscriminantAnalysis(n_components=6)
# lda_data = lda.fit(params_train, labels_train).transform(params_train) 
# lda_data_test = lda_test.fit(params_test, labels_test).transform(params_test)
# lda_names = ["LDA"+str(i) for i in range(0,len(lda_data[0,:]))]
# lda_df = pd.DataFrame(lda_data, columns=lda_names)
# lda_df_test = pd.DataFrame(lda_data_test, columns=lda_names)
# lda_df['target'] = labels_train
# lda_df_test['target'] = labels_test
# params_lda_train = lda_df.iloc[:, 0:-1]
# params_lda_test = lda_df_test.iloc[:, 0:-1]






def models(output, param_dfs, label_dfs, test_params=None, test_labels=None):
    with PdfPages(output) as pdf_pages:
        for t in param_dfs:
            if(test_params == None):
                p = param_dfs[t]
                l = label_dfs[t]
                train_X, test_X, train_y, test_y = train_test_split(p, l, test_size=0.3, stratify=labels, random_state=0)
            else:
                train_X = param_dfs[t]
                train_y = label_dfs[t]
                test_X = test_params[t]
                test_y = test_labels[t]
            title = t
            log.append(t)
            log.append(a.decisionTree(train_X, train_y, test_X, test_y, pdf_pages, title=(title + ", Decision Tree")))
            log.append(a.randomForest(train_X, train_y, test_X, test_y, pdf_pages, title=(title + ", Random Forest")))
            log.append(a.logisticRegression(train_X, train_y, test_X, test_y, pdf_pages, title=(title + ", Log Regression")))
            log.append(a.kNN(train_X, train_y, test_X, test_y, pdf_pages, title=(title + ", KNN")))
            log.append(a.sVM(train_X, train_y, test_X, test_y, pdf_pages, title=(title + ", SVN")))

        # plt.show()
        fig = plt.figure(figsize =(8,5))
        ax = fig.add_subplot(111)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        txt = ('\n'.join(log))
        plt.text(.05,0.05,txt, size=6)
        pdf_pages.savefig(fig)
        print('\n'.join(log))

def modelsCrossValidation(output, param_dfs, label_dfs):                
    with PdfPages(output) as pdf_pages:
        for title in param_dfs:
            p = param_dfs[title]
            l = label_dfs[title]
            log.append((title + " CV"))
            log.append(a.decisionTreeCV(p, l))
            log.append(a.randomForestCV(p, l))
            log.append(a.logisticRegressionCV(p, l))
            log.append(a.kNNCV(p, l))
            log.append(a.sVMCV(p, l))

        # plt.show()
        fig = plt.figure(figsize =(8,5))
        ax = fig.add_subplot(111)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        txt = ('\n'.join(log))
        plt.text(.05,0.05,txt, size=6)
        pdf_pages.savefig(fig)
        print('\n'.join(log))



def neuralNets(output, param_dfs, label_dfs, test_params=None, test_labels=None):
    with PdfPages(output) as pdf_pages:
        if(test_params == None):
            for t in param_dfs:
                p = param_dfs[t]
                l = label_dfs[t]
                train_X, test_X, train_y, test_y = train_test_split(p, l, test_size=0.3, stratify=l, random_state=13)
                title = t
                log.append(t)
                log.append(a.neuralNetwork(train_X, train_y, test_X, test_y, pdf_pages, 12, title = title))
                print('\n'.join(log))
        else:
            for t in param_dfs:
                train_X = param_dfs[t]
                train_y = label_dfs[t]
                test_X = test_params[t]
                test_y = test_labels[t]
                title = t
                log.append(t)
                log.append(a.neuralNetwork(train_X, train_y, test_X, test_y, pdf_pages, 12, title = title))
                print('\n'.join(log))

        # plt.show()
        fig = plt.figure(figsize =(8,5))
        ax = fig.add_subplot(111)
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        txt = ('\n'.join(log))
        plt.text(.05,0.05,txt, size=6)
        pdf_pages.savefig(fig)

# modelPDF = 'avila_Modeling_PRAC_CuredWithB.pdf'
# bivariatePDF = 'avila_EDA_bivariate_TESTTRAINSPLIT.pdf'
# univariatePDF = 'avila_EDA_univariate_curedWB.pdf'
# multivariatePDF = 'avila_EDA_multivariate_PR.pdf'

# p = {"ORIGINAL DATA":params_train}
# l = {"ORIGINAL DATA":labels_train}

# p_test = {"ORIGINAL DATA": params_test}
# l_test = {"ORIGINAL DATA": labels_test}

# univariate(univariatePDF, df_train, params_train, labels_train)
# bivariate(bivariatePDF, df_train, params_train, labels_train)
# multivariate(multivariatePDF)
# modelsCrossValidation(modelPDF, p, l)
# models(modelPDF, p, l, test_params=p_test, test_labels=l_test)
# neuralNets(modelPDF, p, l)
# neuralNets(modelPDF, p, l, test_params=p_test, test_labels=l_test)


def get_split_data():
    return (params_train, labels_train, params_test, labels_test)

def get_all_data():
    return df_train, params_train, labels_train