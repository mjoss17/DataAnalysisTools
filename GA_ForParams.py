"""
GA_ForParams.py
Author: Matt Joss

This file uses genetic algorithms to modify the parameters required
to build a variety of ML algorithms in order to optimize them on a
given dataset.

The Avalible ML algorithms are:
Neural Networks
Random Forests
Decision Trees
K Nearest Neighbors
Support Vector Machines
Logistic Regressions

The parameters for each type of ML algorithm are stored in a Genome 
class. Upon creation, a genome (ML alg) takes up a series of random 
values from the dictionary of parameters held globally. Two genomes 
can be 'bread' with eachother in order to create a new genome. 

To run a GA algorithm to optimize a ML algorithm, use the bigBang()
function and update the 'struct' tag to choose the function that you 
want: ('nn' for Neural Net, 'rf' for Random Forest, 'dt  for Decision
Tree, 'lr' for Log Regression, 'sv' for SVM, 'kn' for K Nearest
Neighbors)

The get_data() function will allow you to import your preprocessed and 
split data. IMPORTANT: Modify the global variables to reflect the dataset
before running

"""
import Single_Models as a
import PreProcess_Avila as av
import PreProcess_Glass as gl
import PreProcess_Iris as ir
import Utils as u
import random
import math
import statistics as s
import numpy as np 
import pandas as pd
import seaborn as sns
import multiprocessing as mp
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
log = []

# modify these based on dataset 
num_classes = 3
num_features = 4
n_pop = 36
n_iter = 20

#get formatted data
def get_data():
    return ir.get_split_data()

nn_params = {
    "epochs": [x for x in range(50, 100)],
    "batch_size": [x for x in range(10,40)],
    "n_layers": [1, 2, 3, 4, 5],
    "n_start_neurons": [x for x in range(600, 1000)],
    "n_neurons": [x for x in range(100, 600)],
    "dropout": [0.4, 0.45, 0.5],
    "optimizers": ["nadam", "adam"],
    "activations": ["relu", "sigmoid", "tanh"],
    "last_layer_activations": ["sigmoid", "tanh"],
    "losses": ["binary_crossentropy"]
}

dt_params = {
    "max_depth": [x for x in range(2, 100)],
    "min_samples_split": [x for x in range(2, 100)],
    "min_samples_leaf": [x for x in range(1, 100)],
    "min_weight_fraction_leaf": [x for x in np.arange(0,0.5,.01)],
    "max_features":[x for x in range(1, num_features)],
}

rf_params = {
    "n_estimators": [x for x in range(40, 300)],
    "max_depth": [x for x in range(2, 100)],
    "min_samples_split": [x for x in range(2, 100)],
    "min_samples_leaf": [x for x in range(1, 100)],
    "min_weight_fraction_leaf": [x for x in np.arange(0,0.5,.01)],
    "max_features":[x for x in range(1, num_features)],
}

lr_params = {
    "fit_intercept": [True, False],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

kn_params = {
    "n_neighbors": [x for x in range(2, 50)],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "leaf_size": [x for x in range(5, 100)]
}

sv_params = {
    "C": [x for x in range(1,3)],
    "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    "degree": [x for x in range(2, 5)],
    "coef0": [x for x in np.arange(0,2,0.1)],
}


class NN_Genome:
    def __init__(self, id_in,):
        self.id = id_in
        self.n_classes = num_classes
        self.attributes = {'epochs' :random.choice(nn_params['epochs']), 
            'batch_size' : random.choice(nn_params['batch_size']),
            'n_layers' : random.choice(nn_params['n_layers']),
            'n_start_neurons' : random.choice(nn_params['n_start_neurons']),
            'n_neurons' : random.choice(nn_params['n_neurons']),
            'dropout' : random.choice(nn_params['dropout']),
            'optimizers' : random.choice(nn_params['optimizers']),
            'activations' : random.choice(nn_params['activations']),
            'last_layer_activations' : random.choice(nn_params['last_layer_activations']),
            'losses' : random.choice(nn_params['losses'])}
  
    def set_a(self, name, x):
        self.attributes[name] = x
    def get_a(self, name):
        return self.attributes[name]
    def get_all(self):
        return self.attributes

    def breed(self, mate, child_id, percent_done):
        child = NN_Genome(child_id)
        n_swaps = random.randint(0, len(nn_params))
        n_mutations = n_mute(percent_done)
        swaps = random.sample(nn_params.keys(), n_swaps)
        stays = [x for x in nn_params if x not in swaps]
        for gene in swaps:
            child.set_a(gene, self.attributes[gene])
        for gene in stays:
            child.set_a(gene, mate.get_a(gene))
        mutations = random.sample(nn_params.keys(), n_mutations)
        for gene in mutations:
            child.set_a(gene, random.choice(nn_params[gene]))
        return child

class DT_Genome:
    def __init__(self, id_in,):
        self.id = id_in
        self.n_classes = num_classes
        self.attributes = {"max_depth": random.choice(dt_params["max_depth"]),
                            "min_samples_split": random.choice(dt_params["min_samples_split"]),
                            "min_samples_leaf": random.choice(dt_params["min_samples_leaf"]),
                            "min_weight_fraction_leaf": random.choice(dt_params["min_weight_fraction_leaf"]),
                            "max_features": random.choice(dt_params["max_features"])}

    def set_a(self, name, x):
        self.attributes[name] = x
    def get_a(self, name):
        return self.attributes[name]
    def get_all(self):
        return self.attributes

    def breed(self, mate, child_id, percent_done):
        child = DT_Genome(child_id)
        n_swaps = random.randint(0, len(dt_params))
        n_mutations = n_mute(percent_done)
        swaps = random.sample(dt_params.keys(), n_swaps)
        stays = [x for x in dt_params if x not in swaps]
        for gene in swaps:
            child.set_a(gene, self.attributes[gene])
        for gene in stays:
            child.set_a(gene, mate.get_a(gene))
        mutations = random.sample(dt_params.keys(), n_mutations)
        for gene in mutations:
            child.set_a(gene, random.choice(dt_params[gene]))
        return child

class RF_Genome:
    def __init__(self, id_in,):
        self.id = id_in
        self.n_classes = num_classes
        self.attributes = {"n_estimators": random.choice(rf_params["n_estimators"]),
                            "max_depth": random.choice(dt_params["max_depth"]),
                            "min_samples_split": random.choice(dt_params["min_samples_split"]),
                            "min_samples_leaf": random.choice(dt_params["min_samples_leaf"]),
                            "min_weight_fraction_leaf": random.choice(dt_params["min_weight_fraction_leaf"]),
                            "max_features": random.choice(dt_params["max_features"])} 

    def set_a(self, name, x):
        self.attributes[name] = x
    def get_a(self, name):
        return self.attributes[name]
    def get_all(self):
        return self.attributes

    def breed(self, mate, child_id, percent_done):
        child = RF_Genome(child_id)
        n_swaps = random.randint(0, len(rf_params))
        n_mutations = n_mute(percent_done)
        swaps = random.sample(rf_params.keys(), n_swaps)
        stays = [x for x in rf_params if x not in swaps]
        for gene in swaps:
            child.set_a(gene, self.attributes[gene])
        for gene in stays:
            child.set_a(gene, mate.get_a(gene))
        mutations = random.sample(rf_params.keys(), n_mutations)
        for gene in mutations:
            child.set_a(gene, random.choice(rf_params[gene]))
        return child

class LR_Genome:
    def __init__(self, id_in,):
        self.id = id_in
        self.n_classes = num_classes
        self.attributes = {"fit_intercept": random.choice(lr_params["fit_intercept"]),
                            "solver": random.choice(lr_params["solver"])} 

    def set_a(self, name, x):
        self.attributes[name] = x
    def get_a(self, name):
        return self.attributes[name]
    def get_all(self):
        return self.attributes

    def breed(self, mate, child_id, percent_done):
        child = LR_Genome(child_id)
        n_swaps = random.randint(0, len(lr_params))
        n_mutations = 1
        swaps = random.sample(lr_params.keys(), n_swaps)
        stays = [x for x in lr_params if x not in swaps]
        for gene in swaps:
            child.set_a(gene, self.attributes[gene])
        for gene in stays:
            child.set_a(gene, mate.get_a(gene))
        mutations = random.sample(lr_params.keys(), n_mutations)
        for gene in mutations:
            child.set_a(gene, random.choice(lr_params[gene]))
        return child

class KN_Genome:
    def __init__(self, id_in,):
        self.id = id_in
        self.n_classes = num_classes
        self.attributes = {'n_neighbors' :random.choice(kn_params['n_neighbors']),
                            'algorithm' :random.choice(kn_params['algorithm']),
                            'leaf_size' :random.choice(kn_params['leaf_size'])} 

    def set_a(self, name, x):
        self.attributes[name] = x
    def get_a(self, name):
        return self.attributes[name]
    def get_all(self):
        return self.attributes

    def breed(self, mate, child_id, percent_done):
        child = KN_Genome(child_id)
        n_swaps = random.randint(0, len(kn_params))
        # n_mutations = random.randint(0, len(params))
        n_mutations = 1
        swaps = random.sample(kn_params.keys(), n_swaps)
        stays = [x for x in kn_params if x not in swaps]
        for gene in swaps:
            child.set_a(gene, self.attributes[gene])
        for gene in stays:
            child.set_a(gene, mate.get_a(gene))
        mutations = random.sample(kn_params.keys(), n_mutations)
        for gene in mutations:
            child.set_a(gene, random.choice(kn_params[gene]))
        return child

class SV_Genome:
    def __init__(self, id_in,):
        self.id = id_in
        self.n_classes = num_classes
        self.attributes = {'C' :random.choice(sv_params['C']),
                            'kernel' :random.choice(sv_params['kernel']),
                            'degree' :random.choice(sv_params['degree']),
                            'coef0' :random.choice(sv_params['coef0'])} 

    def set_a(self, name, x):
        self.attributes[name] = x
    def get_a(self, name):
        return self.attributes[name]
    def get_all(self):
        return self.attributes

    def breed(self, mate, child_id):
        child = SV_Genome(child_id)
        n_swaps = random.randint(0, len(sv_params))
        # n_mutations = random.randint(0, len(params))
        n_mutations = 1
        swaps = random.sample(sv_params.keys(), n_swaps)
        stays = [x for x in sv_params if x not in swaps]
        for gene in swaps:
            child.set_a(gene, self.attributes[gene])
        for gene in stays:
            child.set_a(gene, mate.get_a(gene))
        mutations = random.sample(sv_params.keys(), n_mutations)
        for gene in mutations:
            child.set_a(gene, random.choice(sv_params[gene]))
        return child

def create_nn(genome):
    model = Sequential()
    model.add(Dense(genome.get_a('n_start_neurons'), activation=genome.get_a('activations')))
    model.add(Dropout(genome.get_a('dropout')))
    for layer in range(0, genome.get_a('n_layers')):
        model.add(Dense(genome.get_a('n_neurons'), activation=genome.get_a('activations')))
        model.add(Dropout(genome.get_a('dropout')))
    model.add(Dense(genome.n_classes, activation=genome.get_a('last_layer_activations')))
    model.compile(optimizer=genome.get_a('optimizers'), loss=genome.get_a('losses'), metrics=['accuracy'])
    return model

def create_dt(genome):
    model = DecisionTreeClassifier(max_depth=genome.get_a("max_depth"),
                                    min_samples_split=genome.get_a("min_samples_split"),
                                    min_samples_leaf=genome.get_a("min_samples_leaf"),
                                    min_weight_fraction_leaf=genome.get_a("min_weight_fraction_leaf"),
                                    max_features=genome.get_a("max_features"))
    return model

def create_rf(genome):
    model = RandomForestClassifier(n_estimators=genome.get_a("n_estimators"),
                                    max_depth=genome.get_a("max_depth"),
                                    min_samples_split=genome.get_a("min_samples_split"),
                                    min_samples_leaf=genome.get_a("min_samples_leaf"),
                                    min_weight_fraction_leaf=genome.get_a("min_weight_fraction_leaf"),
                                    max_features=genome.get_a("max_features"))
    return model

def create_lr(genome):
    model = LogisticRegression(fit_intercept=genome.get_a("fit_intercept"),
                                solver=genome.get_a("solver"))
    return model

def create_kn(genome):
    model = KNeighborsClassifier(n_neighbors=genome.get_a("n_neighbors"),
                                algorithm=genome.get_a("algorithm"),
                                leaf_size=genome.get_a("leaf_size"))
    return model

def create_sv(genome):
    model = svm.SVC(kernel=genome.get_a("kernel"), 
                    C=genome.get_a("C"), 
                    degree=genome.get_a("degree"),
                    class_weight='balanced', 
                    probability=True)
    return model




def bigBang(struct, n_pop, n_iterations, pdf_pages=None):
        pop = []
        for i in range(0, n_pop):
            if (struct == 'dt'):
                pop.append(DT_Genome('0_' + str(i)))
            elif (struct == 'nn'):
                pop.append(NN_Genome('0_' + str(i)))
            elif(struct == 'rf'):
                pop.append(RF_Genome('0_' + str(i)))
            elif(struct == 'lr'):
                pop.append(LR_Genome('0_' + str(i)))
            elif(struct == 'kn'):
                pop.append(KN_Genome('0_' + str(i)))
            elif(struct == 'sv'):
                pop.append(SV_Genome('0_' + str(i)))
        if (pdf_pages == None): # local instance 
            with PdfPages("GA_RUNZ_TEST.pdf") as pdf_pages:
                accuracy, loss = iterate(struct, n_iterations, pop, pdf_pages)
                print('\n'.join(log))
                return accuracy, loss
        else: #called from a meta model or has specific pdf
            accuracy, loss = iterate(struct, n_iterations, pop, pdf_pages)
            print('\n'.join(log))
            return accuracy, loss
        

def iterate(struct, n_iterations, pop, pdf_pages, verbose=True):
    i_max = []
    i_mean = []
    i_median = []
    i_min = []

    n_pop = len(pop)
    work_X, work_y, test_X, test_y = get_data() #work = train and validate data
    for n in range(1, n_iterations):
        train_X, ver_X, train_y, ver_y = train_test_split(work_X, work_y, test_size=0.5, stratify=work_y)
        scores = {}
        # pool = mp.Pool(mp.cpu_count())
        pool = mp.Pool(8)
        out = pool.starmap(evaluate, [(struct, train_X, train_y, ver_X, ver_y, genome) for genome in pop])
        pool.close()
        results = [x[0] for x in out] 
        for i in range(0, len(pop)):
            score = results[i]
            log.append("Iteration: " + str(n) + "  Genome: " + str(i))
            log.append(str(score))
            scores[pop[i]] = score
        i_mean.append(s.mean(results))
        i_max.append(max(results))
        i_median.append(s.median(results))
        i_min.append(min(results))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:int(math.sqrt(n_pop))]
        new_pop = []
        count = 0 #just for id's
        for x in top:
            dad = x[0]
            for y in top:
                mom = y[0]
                new_pop.append(mom.breed(dad, str(n) + "_" + str(count), n/float(n_iterations)))
                count += 1
        log.append("TOP SCORES ITERATION " + str(n) )
        log.append(str([x[1] for x in top]))
        log.append('\n')
        log.append(str(new_pop[0].get_all()))
        if(verbose):
            print('\n'.join(log))   
        pop = new_pop

    #Use the test data on the best result from the final iteration
    final_score, loss = evaluate(struct, work_X, work_y, test_X, test_y, pop[0], visualize=True, pdf_pages=pdf_pages)
    log.append(("FINAL SCORE: " + str(final_score)))
    print('\n'.join(log))
    
    #create GA Evolution plot
    stats = pd.DataFrame()
    stats['mean'] = i_mean
    stats['median'] = i_median
    stats['max'] = i_max
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111)
    ax.plot(i_mean, label='Mean')
    ax.plot(i_median, label='Median')
    ax.plot(i_max, label='Max')
    ax.plot(i_min, label='Min')
    plt.title("Accuracy vs. Generations for Genetic Modification of " + struct + " Parameters")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left', labels=['Mean', 'Median', 'Max', 'Min'])
    pdf_pages.savefig(fig)

    return final_score, loss



def evaluate(struct, train_X, train_y, test_X, test_y, genome, visualize=False, pdf_pages=None):

    if(struct == 'nn'):
        model = create_nn(genome)
        title = "Neural Net"
    elif(struct == 'dt'):
        model = create_dt(genome)
        title = "Decision Tree"
    elif(struct == 'rf'):
        model = create_rf(genome)
        title = "Random Forest"
    elif(struct == 'lr'):
        model = create_lr(genome)
        title = "Log Regression"
    elif(struct == 'kn'):
        model = create_kn(genome)
        title = "K Nearest Neighbors"
    elif(struct == 'sv'):
        model = create_sv(genome)
        title = "SMV"

    scaler = StandardScaler() 
    st_data = scaler.fit_transform(train_X)

    scaler_test = StandardScaler()
    st_data_test = scaler_test.fit_transform(test_X)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(train_y.values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    if (struct == 'nn'):
        history = model.fit(st_data, onehot_encoded, batch_size=genome.get_a('batch_size'), epochs=genome.get_a('epochs'), validation_data=(st_data, onehot_encoded), verbose=False)
        # history = model.fit(train_X, onehot_encoded, batch_size=genome.get_a('batch_size'), epochs=genome.get_a('epochs'), validation_data=(st_data, onehot_encoded), verbose=False) 
    else:
        history = model.fit(train_X, onehot_encoded)

    pred_y_ohe = model.predict(test_X)
    pred_y = []
    for i in pred_y_ohe:
        pred_y.append(label_encoder.inverse_transform([np.argmax(i)])[0])
    loss = sum(i != j for i, j in zip(test_y, pred_y ))
    # loss = log_loss(test_y, model.predict_proba(test_X))   

    if (visualize): 
        # visualize(history, test_y, pred_y)
        classes = np.unique(test_y)
        u.plot_confusion_matrix(pdf_pages, test_y, pred_y,
                        normalize=True, classes=classes, title="Confusion Matrix for " + title + " After GA")

    return accuracy_score(test_y, pred_y), loss


def n_mute(percent_done):
    if(percent_done < .92):
        return 1
    elif(percent_done < 1):
        return 0

bigBang('rf', n_pop, n_iter) 