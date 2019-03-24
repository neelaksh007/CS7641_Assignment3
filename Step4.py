#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection as RP
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import time
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
np.random.seed(0)
out = 'step4/'


nn_arch= [(10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(82,82,82)]

nn_iter = [500,1000,1500,2000,2500]

for dataset in ['biodeg.csv']:
    skip = False
    if not skip:
        print("\nWorking on",dataset,"data set...")
    
        data_df = pd.read_csv(dataset)        
        dataX = data_df.iloc[:,:41]
        dataY = data_df.iloc[:,41]
        dataX = StandardScaler().fit_transform(dataX.astype('float64'))
    
        timing = {}
        components = {}
        
        # Fit/transform with PCA
        print("Running PCA...")
        #sys.exit(1)
        pca = PCA(n_components = 10, random_state=5)
        components['PCA'] = 10
        dataX_PCA = pca.fit_transform(dataX)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(dataX_PCA,dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN PCA.csv')
    
        # Time PCA with best parameters
        start = time.time()
        clf = MLPClassifier(activation='relu',early_stopping=True,random_state=5,hidden_layer_sizes=(30,30),max_iter=500)
        clf.fit(dataX_PCA,dataY)
        end = time.time()
        timing['PCA']=end-start
    
        # Fit/transform with FastICA
        print("Running FastICA...")
        ica = FastICA(n_components = 10, random_state=5)
        components['ICA'] = 10
        dataX_ICA = ica.fit_transform(dataX)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(dataX_ICA,dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN ICA.csv')
    
        # Time ICA with best parameters
        start = time.time()
        clf = MLPClassifier(activation='relu',early_stopping=True,random_state=5,hidden_layer_sizes=(10,10),max_iter=500)
        clf.fit(dataX_ICA,dataY)
        end = time.time()
        timing['ICA']=end-start
    
        # Fit/transform with RP
        print("Running RP...")
        rp = RP(n_components = 45, random_state=5)
        components['RP'] = 45
        dataX_RP = rp.fit_transform(dataX)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(dataX_RP,dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN RP.csv')
    
        # Time RP with best parameters
        start = time.time()
        clf = MLPClassifier(activation='relu',early_stopping=True,random_state=5,hidden_layer_sizes=(40,40),max_iter=500)
        clf.fit(dataX_RP,dataY)
        end = time.time()
        timing['RP']=end-start
    
        # Fit/transform with SVD
        print("Running SVD...")
        svd = SVD(n_components = 3, random_state=5)
        components['SVD'] = 3
        dataX_SVD = svd.fit_transform(dataX)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(dataX_SVD,dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN SVD.csv')
        
        # Time SVD with best parameters
        start = time.time()
        clf = MLPClassifier(activation='relu',early_stopping=True,random_state=5,hidden_layer_sizes=(50,50),max_iter=500)
        clf.fit(dataX_SVD,dataY)
        end = time.time()
        timing['SVD']=end-start
    
        # Run benchmark grid search
        print("Benchmark NN...")
        components['Benchmark'] = dataX.shape[1]
        
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(dataX,dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN bmk.csv')
    
        # Time benchmark with best parameters
        start = time.time()
        clf = MLPClassifier(activation='relu',early_stopping=True,random_state=5,hidden_layer_sizes=(50,50),max_iter=500)
        clf.fit(dataX,dataY)
        end = time.time()
        timing['Benchmark']=end-start
    
        print(timing)
        plt.bar(range(len(timing)), list(timing.values()), align='center')
        plt.xticks(range(len(timing)), list(timing.keys()))
        plt.title("Training Time")
        plt.ylabel("seconds")
        plt.savefig(out+'timing.png')
    
        plt.close()
        plt.figure()
        plt.bar(range(len(components)), list(components.values()), align='center')
        plt.xticks(range(len(components)), list(components.keys()))
        plt.title("Number of Components")
        plt.savefig(out+'components.png')

    #####################################################
    #####################################################
    print ("TESTING ANOTHER PIECE")
    
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),type=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("step4/"+'{}_components.png'.format(type))

data = pd.read_csv('biodeg.csv')
X = data.iloc[:,:41]
y = data.iloc[:,41]
scaler = MinMaxScaler(feature_range=[0,100])
from sklearn.preprocessing import StandardScaler
X_norm = StandardScaler().fit_transform(X)
###
pca = PCA(n_components=10, random_state=10)
X_r = pca.fit(X).transform(X)
X_pca = X_r
####
ica = FastICA(n_components=10, random_state=10)
X_r = ica.fit(X).transform(X)
X_ica = X_r
####
rca = GaussianRandomProjection(n_components=10, random_state=10)
X_r = rca.fit_transform(X_norm)
X_rca = X_r
####
svd = SVD(n_components=10)
X_r = svd.fit_transform(X_norm)
X_svd = X_r
#'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]
clf = MLPClassifier(hidden_layer_sizes=(82,82,82),alpha=0.316227766, learning_rate_init=0.016, random_state=0, solver="lbfgs")
plot_learning_curve(clf, "MLP using Benchmark transformed features", X_norm, y, ylim=[0,1],type = "BMK")
plot_learning_curve(clf, "MLP using PCA transformed features", X_pca, y, ylim=[0,1],type = "PCA")
plot_learning_curve(clf, "MLP using ICA transformed features", X_ica, y, ylim=[0,1],type = "ICA")
plot_learning_curve(clf, "MLP using RP transformed features", X_rca, y, ylim=[0,1],type = "RP")
plot_learning_curve(clf, "MLP using SVD transformed features", X_svd, y, ylim=[0,1],type = "SVD")



    ########################
    ##trying EM
    #print ("DOING EM stuff")
    #k_em = 10
    #model = GaussianMixture(k_em , covariance_type='full', random_state=0)
    #X_R = model.fit(X)
    