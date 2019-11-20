"""
CrossValidation.py
Author: Matt Joss

This file contains a series of runctions to run cross validation tests
on different ML algorithms. 

These functions can be used induvidually or all of them can be compared
in conjunction with the compareAllModels() function. 
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import PreProcess_Avila as av
import PreProcess_Glass as gl
import PreProcess_Iris as ir
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

log = []
crossValSplits = 5

def decisionTreeCV(params, labels, boot=False):
    model = DecisionTreeClassifier()
    splits = crossValSplits
    if(boot):
        splits = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(model, params, labels, cv= splits)
    return ("DT: Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

def randomForestCV(params, labels, boot=False):
    model = RandomForestClassifier(n_estimators=100, max_depth=2, class_weight='balanced')
    splits = crossValSplits
    if(boot):
        splits = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(model, params, labels, cv= splits)
    return ("RF: Accuracy: %0.5f (+/- %0.5f) " % (scores.mean(), scores.std() * 2))

def kNNCV(params, labels, boot=False):
    model = KNeighborsClassifier(n_neighbors =3)
    splits = crossValSplits
    if(boot):
        splits = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(model, params, labels, cv= splits)
    return ("KN: Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

def logisticRegressionCV(params, labels, boot=False):
    model = LogisticRegression()
    splits = crossValSplits
    if(boot):
        splits = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(model, params, labels, cv= splits)
    return ("LR: Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

def sVMCV(params, labels, boot=False):
    model = SVC(kernel='linear', C= 1, class_weight='balanced')
    splits = crossValSplits;
    if(boot):
        splits = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(model, params, labels, cv= splits)
    return ("SV: Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

def neuralNetworkCV(params, labels, boot=False):
    #One-Hot encode the labels
    targets = pd.get_dummies(labels)
    splits = crossValSplits
    if(boot):
        splits = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    neural_network = KerasClassifier(build_fn=create_nn, 
                                epochs=100, 
                                batch_size=50, 
                                verbose=0)
    scores = cross_val_score(neural_network, params.values, targets, cv= splits)
    return("NN: Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))


def create_nn(num_outputs):
    model = Sequential()
    model.add(Dense(1000, activation= 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(500, activation= 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(200, activation= 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_outputs, activation= 'softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def compareAllModels(output, params, labels, name_of_dataset):
    with PdfPages(output) as pdf_pages:
        log.append((name_of_dataset + " CV"))
        log.append(decisionTreeCV(params, labels))
        log.append(randomForestCV(params, labels))
        log.append(logisticRegressionCV(params, labels))
        log.append(kNNCV(params, labels))
        log.append(sVMCV(params, labels))
        # log.append(neuralNetworkCV(params, labels))

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

def run():

    df, params, labels = av.get_all_data()
    compareAllModels("CV_Models_avila.pdf", params, labels, "Avila")

    df, params, labels = gl.get_all_data()
    compareAllModels("CV_Models_glass.pdf", params, labels, "Glass")

    df, params, labels = ir.get_all_data()
    compareAllModels("CV_Models_iris.pdf", params, labels, "Iris")

run()