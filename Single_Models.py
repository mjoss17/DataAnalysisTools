"""
Single_Models.py
Author: Matt Joss

This file contains a series of functions that will run ML Algorithms
and return accuracies + loss for the run. 
"""
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import Utils as u
from keras import models
from keras import layers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import jarque_bera, kstest, shapiro

from tensorflow import set_random_seed
set_random_seed(123)
random.seed(123)

def decisionTree(train_X, train_y, test_X, test_y, pdf_pages, title=""):
    model = DecisionTreeClassifier()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    accuracy = accuracy_score(test_y, pred_y)
    loss = log_loss(test_y, model.predict_proba(test_X))
    u.plot_confusion_matrix(pdf_pages, test_y, pred_y, classes=np.unique(train_y), normalize=True, title=title)
    return accuracy, loss

def randomForest(train_X, train_y, test_X, test_y, pdf_pages, title=""):
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    accuracy = accuracy_score(test_y, pred_y)
    loss = log_loss(test_y, model.predict_proba(test_X))
    u.plot_confusion_matrix(pdf_pages, test_y, pred_y, classes=np.unique(train_y), normalize=True, title=title)
    return accuracy, loss

def kNN(train_X, train_y, test_X, test_y, pdf_pages, title=""):
    model = KNeighborsClassifier(n_neighbors =3)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    accuracy = accuracy_score(test_y, pred_y)
    loss = log_loss(test_y, model.predict_proba(test_X))
    u.plot_confusion_matrix(pdf_pages, test_y, pred_y, classes=np.unique(train_y), normalize=True, title=title)
    return accuracy, loss

def logisticRegression(train_X, train_y, test_X, test_y, pdf_pages, title=""):
    model = LogisticRegression()
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    accuracy = accuracy_score(test_y, pred_y)
    loss = log_loss(test_y, model.predict_proba(test_X))
    u.plot_confusion_matrix(pdf_pages, test_y, pred_y, classes=np.unique(train_y), normalize=True, title=title)
    return accuracy, loss

def sVM(train_X, train_y, test_X, test_y, pdf_pages, title=""):
    model = svm.SVC(kernel='linear', C= 1, class_weight='balanced', probability=True)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    accuracy = accuracy_score(test_y, pred_y)
    loss = log_loss(test_y, model.predict_proba(test_X))
    u.plot_confusion_matrix(pdf_pages, test_y, pred_y, classes=np.unique(train_y), normalize=True, title=title)
    return accuracy, loss

def neuralNetwork(train_X, train_y, test_X, test_y, pdf_pages, num_outputs, title=""):
    # test_y = modTestY(test_y)
    classes = np.unique(pd.concat([train_y, test_y]))
    model = create_nn(num_outputs)
    scaler = StandardScaler() 
    scaler_test = StandardScaler()
    st_data = scaler.fit_transform(train_X)
    st_data_test = scaler_test.fit_transform(test_X)
    train_values = train_y.values
    print(np.unique(train_values))
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(train_values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # targets = pd.get_dummies(train_y).values
    print(onehot_encoded.shape)

    history = model.fit(st_data, onehot_encoded, batch_size=30, epochs=100, validation_data=(st_data, onehot_encoded))
    pred_y = model.predict(st_data_test)
    inverted = []
    for i in pred_y:
        inverted.append(label_encoder.inverse_transform([np.argmax(i)])[0])
    print(len(history.history['loss']))
    print(len([x for x in range(1,101)]))
    erange = range(1, len(history.history['loss']) + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    # ax = sns.lineplot(history.history['loss'], [x for x in range(1,101)])
    ax = plt.plot(erange, history.history['loss'], 'r-')
    ax = plt.plot(erange, history.history['val_loss'], 'b-')
    plt.title('NN Model Loss for ' + title)
    plt.legend(['Training Loss', 'Test Loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    pdf_pages.savefig(fig)
    u.plot_confusion_matrix(pdf_pages, test_y,
                        inverted,
                        normalize=True, classes=classes, title=title + "NN")
    return ("NN: Accuracy: %0.5f, Log Loss: %0.5f" % (accuracy_score(test_y, inverted), log_loss(test_y, model.predict_proba(test_X))))

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