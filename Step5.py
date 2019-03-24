#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection as RP
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
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
out = 'step5/ '

nn_arch= [(10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(82,82,82)]
nn_iter = [500,1000,1500,2000,2500]

for dataset in ['biodeg.csv']:
    skip = False
    if not skip:
        print("\nWorking on",dataset,"data set...")
    
        best_km = pd.DataFrame(columns=['Layers','Iterations','Score'])
        best_em = pd.DataFrame(columns=['Layers','Iterations','Score'])
    
        data_df = pd.read_csv(dataset)        
        dataX = data_df.iloc[:,:41]
        dataY = data_df.iloc[:,41]
        dataX = StandardScaler().fit_transform(dataX.astype('float64'))
    
        
        # Fit/transform with PCA
        print("Running PCA...")
        pca = PCA(n_components = 10, random_state=5)
        dataX_PCA = pca.fit_transform(dataX)
    
        # Run KM
        print("Running k-means...")
        if dataset == 'biodeg.csv':
            km = 10
        else:
            km = 2
        model = KMeans(n_clusters=km)
        labels_KM_PCA = model.fit_predict(dataX_PCA)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(labels_KM_PCA.reshape(-1,1),dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN KM PCA.csv')
        best_indices = tmp.index[tmp['rank_test_score'] == 1].tolist()
        best_km = best_km.append({'Layers': str(tmp.iloc[best_indices[0],4]), 'Iterations': tmp.iloc[best_indices[0],5],'Score': tmp.iloc[best_indices[0],12]},ignore_index=True)
    
        # Run EM
        print("Running E-M...")
        if dataset == 'biodeg.csv':
            em = 10
        else:
            em = 2
    
        model = GaussianMixture(em, covariance_type='full', random_state=0)
        labels_EM_PCA = model.fit_predict(dataX_PCA)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(labels_EM_PCA.reshape(-1,1),dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN EM PCA.csv')
        best_indices = tmp.index[tmp['rank_test_score'] == 1].tolist()
        best_em = best_em.append({'Layers': str(tmp.iloc[best_indices[0],4]), 'Iterations': tmp.iloc[best_indices[0],5],'Score': tmp.iloc[best_indices[0],12]},ignore_index=True)
    
        # Fit/transform with FastICA
        print("Running FastICA...")
        ica = FastICA(n_components = 10, random_state=5)
        dataX_ICA = ica.fit_transform(dataX)
    
        # Run KM
        print("Running k-means...")
        model = KMeans(n_clusters=km)
        labels_KM_PCA = model.fit_predict(dataX_ICA)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(labels_KM_PCA.reshape(-1,1),dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'Adult NN KM ICA.csv')
        best_indices = tmp.index[tmp['rank_test_score'] == 1].tolist()
        best_km = best_km.append({'Layers': str(tmp.iloc[best_indices[0],4]), 'Iterations': tmp.iloc[best_indices[0],5],'Score': tmp.iloc[best_indices[0],12]},ignore_index=True)
    
        # Run EM
        print("Running E-M...")
        model = GaussianMixture(em, covariance_type='full', random_state=0)
        labels_EM_PCA = model.fit_predict(dataX_ICA)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(labels_EM_PCA.reshape(-1,1),dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN EM ICA.csv')
        best_indices = tmp.index[tmp['rank_test_score'] == 1].tolist()
        best_em = best_em.append({'Layers': str(tmp.iloc[best_indices[0],4]), 'Iterations': tmp.iloc[best_indices[0],5],'Score': tmp.iloc[best_indices[0],12]},ignore_index=True)
    
        # Fit/transform with RP
        print("Running RP...")
        rp = RP(n_components = 45, random_state=5)
        dataX_RP = rp.fit_transform(dataX)
    
        # Run KM
        print("Running k-means...")
        model = KMeans(n_clusters=km)
        labels_KM_RP = model.fit_predict(dataX_RP)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(labels_KM_RP.reshape(-1,1),dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN KM RP.csv')
        best_indices = tmp.index[tmp['rank_test_score'] == 1].tolist()
        best_km = best_km.append({'Layers': str(tmp.iloc[best_indices[0],4]), 'Iterations': tmp.iloc[best_indices[0],5],'Score': tmp.iloc[best_indices[0],12]},ignore_index=True)
    
        # Run EM
        print("Running E-M...")
        model = GaussianMixture(em, covariance_type='full', random_state=0)
        labels_EM_RP = model.fit_predict(dataX_RP)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(labels_EM_RP.reshape(-1,1),dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN EM RP.csv')
        best_indices = tmp.index[tmp['rank_test_score'] == 1].tolist()
        best_em = best_em.append({'Layers': str(tmp.iloc[best_indices[0],4]), 'Iterations': tmp.iloc[best_indices[0],5],'Score': tmp.iloc[best_indices[0],12]},ignore_index=True)
    
        # Fit/transform with SVD
        print("Running SVD...")
        svd = SVD(n_components = 3, random_state=5)
        dataX_SVD = svd.fit_transform(dataX)
    
        # Run KM
        print("Running k-means...")
        model = KMeans(n_clusters=km)
        labels_KM_SVD = model.fit_predict(dataX_SVD)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(labels_KM_SVD.reshape(-1,1),dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN KM SVD.csv')
        best_indices = tmp.index[tmp['rank_test_score'] == 1].tolist()
        best_km = best_km.append({'Layers': str(tmp.iloc[best_indices[0],4]), 'Iterations': tmp.iloc[best_indices[0],5],'Score': tmp.iloc[best_indices[0],12]},ignore_index=True)
    
        # Run EM
        print("Running E-M...")
        model = GaussianMixture(em, covariance_type='full', random_state=0)
        labels_EM_SVD = model.fit_predict(dataX_SVD)
    
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(labels_EM_SVD.reshape(-1,1),dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN EM SVD.csv')
        best_indices = tmp.index[tmp['rank_test_score'] == 1].tolist()
        best_em = best_em.append({'Layers': str(tmp.iloc[best_indices[0],4]), 'Iterations': tmp.iloc[best_indices[0],5],'Score': tmp.iloc[best_indices[0],12]},ignore_index=True)
    
        # Run NN grid search
    
        print("Benchmark NN...")
        grid ={'NN__hidden_layer_sizes':nn_arch, 'NN__max_iter':nn_iter, 'NN__learning_rate_init': [0.016], 'NN__alpha': [0.316227766]}
        mlp = MLPClassifier(activation='relu',early_stopping=True,random_state=5)
        pipe = Pipeline([('NN',mlp)])
        gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)
    
        gs.fit(dataX,dataY)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(out+'QSAR NN bmk.csv')
    
        print("KM:",best_km)
        print("EM:",best_em)

    #########################################################################
    ############################################################################
    #Rerun ANN on transformed features with clusters new feature
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
        plt.savefig("step5/"+'{}_components.png'.format(type))




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
    svd = SVD(n_components=2)
    X_r = svd.fit_transform(X_norm)
    X_svd = X_r


    clf = MLPClassifier(hidden_layer_sizes=(82,82,82),alpha=0.316227766, learning_rate_init=0.016, random_state=0, solver="lbfgs")
    
    clusterer = KMeans(n_clusters=10, random_state=10).fit(X_pca)
    y_kmeans = clusterer.labels_
    X_df = pd.DataFrame(X_pca)
    print(X_df)
    X_df[11] = y_kmeans
    print(X_df)
    plot_learning_curve(clf, "MLP using PCA - Kmeans transformed features", X_df, y, ylim=[0,1],type="PCA")
    
    clusterer = KMeans(n_clusters=10, random_state=10).fit(X_ica)
    y_kmeans = clusterer.labels_
    X_df = pd.DataFrame(X_ica)
    X_df[11] = y_kmeans
    plot_learning_curve(clf, "MLP using ICA - Kmeans transformed features", X_df, y, ylim=[0,1],type="ICA")
    
    clusterer = KMeans(n_clusters=10, random_state=10).fit(X_rca)
    y_kmeans = clusterer.labels_
    X_df = pd.DataFrame(X_rca)
    X_df[11] = y_kmeans
    plot_learning_curve(clf, "MLP using RP - Kmeans transformed features", X_df, y, ylim=[0,1],type="RP")
    
    clusterer = KMeans(n_clusters=10, random_state=10).fit(X_svd)
    y_kmeans = clusterer.labels_
    X_df = pd.DataFrame(X_svd)
    X_df[11] = y_kmeans
    plot_learning_curve(clf, "MLP using SVD - Kmeans transformed features", X_df, y, ylim=[0,1],type="SVD")

    #############################################
    ##EM
    ##############################################
    k_em = 10
    y_kmeans = GaussianMixture(k_em , covariance_type='full', random_state=0).fit_predict(X_pca)
    X_df = pd.DataFrame(X_pca)
    X_df[11] = y_kmeans
    plot_learning_curve(clf, "MLP using PCA - EM transformed features", X_df, y, ylim=[0,1],type="PCA_EM")

    y_kmeans = GaussianMixture(k_em , covariance_type='full', random_state=0).fit_predict(X_ica)
    X_df = pd.DataFrame(X_ica)
    X_df[11] = y_kmeans
    plot_learning_curve(clf, "MLP using ICA - EM transformed features", X_df, y, ylim=[0,1],type="ICA_EM")
    
    y_kmeans = GaussianMixture(k_em , covariance_type='full', random_state=0).fit_predict(X_rca)
    X_df = pd.DataFrame(X_rca)
    X_df[11] = y_kmeans
    plot_learning_curve(clf, "MLP using RP - EM transformed features", X_df, y, ylim=[0,1],type="RP_EM")
    
    y_kmeans = GaussianMixture(k_em , covariance_type='full', random_state=0).fit_predict(X_svd)
    X_df = pd.DataFrame(X_svd)
    X_df[11] = y_kmeans
    plot_learning_curve(clf, "MLP using SVD - EM transformed features", X_df, y, ylim=[0,1],type="SVD_EM")

    