#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD as SVD
from itertools import product
import scipy.sparse as sps
from scipy.linalg import pinv
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
from sklearn.svm import LinearSVC
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

np.random.seed(0)
out = 'step2/'


######
##PCA
######
for dataset in ['biodeg.csv_header','voice.csv']:
    print("Working on",dataset,"data set...")

    data_df = pd.read_csv(dataset) 
    if  dataset == "biodeg.csv_header":      
        dataX = data_df.iloc[:,:41]
        dataY = data_df.iloc[:,41]
        dataset = "QSAR"
        comps = np.int32(np.linspace(2, 40,20))
    else:
        dataX = data_df.iloc[:,:20]
        dataY = data_df.iloc[:,20] 
        dataset = "VOICE"  
        comps = np.int32(np.linspace(2, 20,20))
    dataX = StandardScaler().fit_transform(dataX.astype('float64'))
    #######################################################
    split = train_test_split(dataX, dataY, test_size = 0.3,
    random_state = 42)
    (trainData, testData, trainTarget, testTarget) = split
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    print("Running RP...")
    accuracies = []
    for comp in comps:
        # create the random projection
        #sp = SparseRandomProjection(n_components = comp)
        #X = sp.fit_transform(trainData)
        sp = PCA(n_components = comp, random_state=5)
        X = sp.fit_transform(trainData)
        # train a classifier on the sparse random projection
        model = LinearSVC()
        model.fit(X, trainTarget)
    
        # evaluate the model and update the list of accuracies
        test = sp.transform(testData)
        accuracies.append(metrics.accuracy_score(model.predict(test), testTarget))

    plt.figure()
    plt.suptitle("Accuracy of Sparse Projection on {}".format(dataset))
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    if dataset =="QSAR":
        plt.xlim([2, 40])
    else:
        plt.xlim([2, 20])    
    plt.ylim([0, 1.0])
     
    # plot the baseline and random projection accuracies
    plt.plot(comps, [baseline] * len(accuracies), color = "r")
    plt.plot(comps, accuracies)
    plt.grid()
    plt.savefig(out+"{}_PCA_Accuracy.png".format(dataset))
    #######################################################
    
    # Run PCA
    print("Running PCA...")
    pca = PCA(random_state=5)
    pca.fit(dataX)
    PCA_EV = pd.DataFrame(pca.explained_variance_,columns=['PCA EV'])
    PCA_EV.to_csv(out+'{} explained_variance.csv'.format(dataset))

    print("Prepping plot...")
    plt.close()
    plt.figure()
    plt.grid()
    plt.title('Eigenvalues per Component'.format(dataset))
    plt.bar(PCA_EV.index.values,PCA_EV.iloc[:, 0],label=dataset)
    plt.xlabel('Components')
    plt.ylabel('Explained Variance')
    plt.legend(loc='best')
    plt.savefig(out+"{}_PCA_EV.png".format(dataset))

    # PCA:  Look at loss in reconstruction versus number of components
    print("Plotting reconstruction loss...")
    loss = []
    for num_components in range(len(data_df.columns)-1):
        pca = PCA(n_components=num_components)
        pca.fit(dataX)
        dataX_pca = pca.transform(dataX)
        dataX_projected = pca.inverse_transform(dataX_pca)
        loss.append(((dataX - dataX_projected) ** 2).mean())
    plt.close()
    plt.figure()
    plt.grid()
    plt.title('Loss in Reconstruction ({})'.format(dataset))
    plt.plot(range(1,len(loss)+1),loss)
    plt.xlabel('Components')
    plt.ylabel('MSE')
    plt.savefig(out+"{}_PCA_Loss.png".format(dataset))

    # Visualize PCA with 2 components (https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
    print("Visualizing two component PCA...")
    pca_2D = PCA(n_components=2)
    PC = pca_2D.fit_transform(dataX)
    principalDf = pd.DataFrame(data = PC, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataY], axis = 1)
    
    plt.close()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('Two Component PCA ({})'.format(dataset), fontsize = 20)
    targets = list(set(data_df['class'].copy().values))
    print (data_df)
    for label in [0,1]:
        indicesToKeep = finalDf['class']==label
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],finalDf.loc[indicesToKeep, 'principal component 2'])
    ax.legend(targets)
    ax.grid()
    plt.savefig(out+"{}_PCA_2D.png".format(dataset))

            
####
#ICA
####
for dataset in ['biodeg.csv_header','voice.csv']:
    print("Working on",dataset,"data set...")

    data_df = pd.read_csv(dataset) 
    if  dataset == "biodeg.csv_header":      
        dataX = data_df.iloc[:,:41]
        dataY = data_df.iloc[:,41]
        dataset = "QSAR"
        comps = np.int32(np.linspace(2, 40,20))
    else:
        dataX = data_df.iloc[:,:20]
        dataY = data_df.iloc[:,20]
        dataset = "VOICE" 
        comps = np.int32(np.linspace(2, 20,20))  
    dataX = StandardScaler().fit_transform(dataX.astype('float64'))
    
    #######################################################
    split = train_test_split(dataX, dataY, test_size = 0.3,
    random_state = 42)
    (trainData, testData, trainTarget, testTarget) = split
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    print("Running RP...")
    accuracies = []
    for comp in comps:
        # create the random projection
        #sp = SparseRandomProjection(n_components = comp)
        #X = sp.fit_transform(trainData)
        #sp = PCA(n_components = comp, random_state=5)
        #X = sp.fit_transform(trainData)
        sp = FastICA(random_state=5, n_components=comp)
        X = sp.fit_transform(trainData)
        # train a classifier on the sparse random projection
        model = LinearSVC()
        model.fit(X, trainTarget)
    
        # evaluate the model and update the list of accuracies
        test = sp.transform(testData)
        accuracies.append(metrics.accuracy_score(model.predict(test), testTarget))

    plt.figure()
    plt.suptitle("Accuracy of Sparse Projection on {}".format(dataset))
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    if dataset =="QSAR":
        plt.xlim([2, 40])
    else:
        plt.xlim([2, 20])    
    plt.ylim([0, 1.0])
     
    # plot the baseline and random projection accuracies
    plt.plot(comps, [baseline] * len(accuracies), color = "r")
    plt.plot(comps, accuracies)
    plt.grid()
    plt.savefig(out+"{}_ICA_Accuracy.png".format(dataset))
    #######################################################

    print("Running ICA...")
    loss = []
    kurtosis = []
    for num_components in range(1,len(data_df.columns)):
        ICA = FastICA(random_state=5, n_components=num_components)
        dataX_ICA = ICA.fit_transform(dataX)
        dataX_ICA_df = pd.DataFrame(dataX_ICA)
        dataX_ICA_df = dataX_ICA_df.kurt(axis=0)
        kurtosis.append(dataX_ICA_df.abs().mean())
        dataX_projected = ICA.inverse_transform(dataX_ICA)
        loss.append(((dataX - dataX_projected) ** 2).mean())
    
    print("Plotting reconstruction loss...")
    plt.close()
    plt.figure()
    plt.grid()
    plt.title('Loss in Reconstruction ({})'.format(dataset))
    plt.plot(range(1,len(loss)+1),loss)
    plt.xlabel('Components')
    plt.ylabel('MSE')
    plt.savefig(out+"{}_ICA_Loss.png".format(dataset))

    print("Plotting kurtosis...")
    plt.close()
    plt.figure()
    plt.grid()
    plt.title('Kurtosis ({})'.format(dataset))
    plt.plot(range(1,len(kurtosis)+1),kurtosis)
    plt.xlabel('Components')
    plt.ylabel('Kurtosis')
    plt.savefig(out+"{}_ICA_Kurtosis.png".format(dataset))

    # Visualize ICA with 2 components (https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
    print("Visualizing two component ICA...")
    ICA_2D = FastICA(n_components=2)
    PC = ICA_2D.fit_transform(dataX)
    principalDf = pd.DataFrame(data = PC, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataY], axis = 1)
    
    plt.close()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('Two Component ICA ({})'.format(dataset), fontsize = 20)
    targets = list(set(data_df['class'].copy().values))
    for label in targets:
        indicesToKeep = finalDf['class']==label
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],finalDf.loc[indicesToKeep, 'principal component 2'])
    ax.legend(targets)
    ax.grid()
    plt.savefig(out+"{}_ICA_2D.png".format(dataset))


#####
##RP
#####

def reconstructionError(projections,X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W)@(X.T)).T # Unproject projected data
    errors = np.square(X-reconstructed)
    return np.nanmean(errors)

for dataset in ['biodeg.csv_header','voice.csv']:
    print("\nWorking on",dataset,"data set...")

    data_df = pd.read_csv(dataset) 
    if  dataset == "biodeg.csv_header":      
        dataX = data_df.iloc[:,:41]
        dataY = data_df.iloc[:,41]
        dataset = "QSAR"
        comps = np.int32(np.linspace(2, 40,20))
    else:
        dataX = data_df.iloc[:,:20]
        dataY = data_df.iloc[:,20]
        dataset = "VOICE" 
        comps = np.int32(np.linspace(2,20,20))  
    dataX = StandardScaler().fit_transform(dataX.astype('float64'))
    split = train_test_split(dataX, dataY, test_size = 0.3,
    random_state = 42)
    (trainData, testData, trainTarget, testTarget) = split
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    print("Running RP...")
    accuracies = []
    for comp in comps:
        # create the random projection
        sp = SparseRandomProjection(n_components = comp)
        X = sp.fit_transform(trainData)
 
        # train a classifier on the sparse random projection
        model = LinearSVC()
        model.fit(X, trainTarget)
    
        # evaluate the model and update the list of accuracies
        test = sp.transform(testData)
        accuracies.append(metrics.accuracy_score(model.predict(test), testTarget))

    plt.figure()
    plt.suptitle("Accuracy of Sparse Projection on {}".format(dataset))
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    if dataset =="QSAR":
        plt.xlim([2, 40])
    else:
        plt.xlim([2, 20])    
    plt.ylim([0, 1.0])
     
    # plot the baseline and random projection accuracies
    plt.plot(comps, [baseline] * len(accuracies), color = "r")
    plt.plot(comps, accuracies)
    plt.grid()
    plt.savefig(out+"{}_RP_Accuracy.png".format(dataset))    

    # Plot loss
    print("Plotting reconstruction loss...")
    loss = pd.DataFrame(index=range(2,len(data_df.columns)),columns=set(range(10)))
    for i,num_components in product(range(10),range(2,len(data_df.columns))):
        rp = SparseRandomProjection(random_state=i, n_components=num_components)
        dataX_RP = rp.fit(dataX)
        loss.loc[num_components,i] = reconstructionError(rp,dataX)
    loss['average']=loss.mean(axis=1)
    plt.close()
    plt.figure()
    plt.grid()
    plt.plot(range(2,len(data_df.columns)),loss['average'])
    plt.xlabel('Components')
    plt.ylabel('Loss')
    plt.title("Performance of Random Projection")
    plt.savefig(out+"{}_RP_Loss.png".format(dataset))
#
    print("Plotting average distance...")
    # From https://scikit-learn.org/stable/auto_examples/plot_johnson_lindenstrauss_bound.html
    plt.close()
    plt.figure()
    if LooseVersion(matplotlib.__version__) >= '2.1':
        density_param = {'density': True}
    else:
        density_param = {'normed': True}
    dists = euclidean_distances(dataX, squared=True).ravel()
    nonzero = dists != 0
    dists = dists[nonzero]
#
    n_components_range = range(2,len(data_df.columns),3)
    mean_distance = pd.DataFrame(index=n_components_range,columns=set(range(10)))
    for i,n_components in product(range(10),n_components_range):
        rp = SparseRandomProjection(n_components=n_components)
        projected_data = rp.fit_transform(dataX)
        projected_dists = euclidean_distances(projected_data, squared=True).ravel()[nonzero]
        rates = projected_dists / dists
        mean_distance.loc[n_components,i] = np.mean(rates)
    mean_distance['average']=mean_distance.mean(axis=1)
    mean_distance['average']=(1-mean_distance['average'])**2
    plt.xlabel("Number of Components")
    plt.ylabel("Distance Distortion")
    plt.grid()
    plt.title("Distance Distortion")
    plt.plot(n_components_range,mean_distance['average'])
    plt.savefig(out+"{}_RP_Distance.png".format(dataset))
#
    print("Distance plots for optimal components...")
    if  dataset == "QSAR":
        n_components = 12
    else:
        n_components = 8
    projected_dists = pd.DataFrame(columns=range(10))
    rates = pd.DataFrame(columns=range(10))
    for i in range(10):
        rp = SparseRandomProjection(n_components=n_components)
        projected_data = rp.fit_transform(dataX)
        projected_dists[i] = euclidean_distances(projected_data, squared=True).ravel()[nonzero]
        rates[i] = projected_dists[i] / dists
#
    projected_dists['average'] = projected_dists.mean(axis=1)
    rates['average'] = rates.mean(axis=1)
    
    plt.close()
    plt.figure()
    plt.hexbin(dists, projected_dists['average'], gridsize=100, cmap=plt.cm.PuBu)
    plt.xlabel("Pairwise squared distances in original space")
    plt.ylabel("Pairwise squared distances in projected space")
    plt.title("Pairwise distances distribution for n_components=%d" % n_components)
    cb = plt.colorbar()
    cb.set_label('Sample pairs counts')
    plt.savefig(out+"{}_RP_OptDistance.png".format(dataset))
#
    plt.close()
    plt.figure()
    plt.hist(rates['average'], bins=50, range=(0., 2.), edgecolor='k', **density_param)
    plt.xlabel("Squared distances rate: projected / original")
    plt.ylabel("Distribution of samples pairs")
    plt.title("Histogram of pairwise distance rates for n_components=%d" % n_components)
    plt.savefig(out+"{}_RP_DistHist.png".format(dataset))
#
    print("Visualizing two component RP...")
    RP_2D = SparseRandomProjection(n_components=2)
    RP = RP_2D.fit_transform(dataX)
    principalDf = pd.DataFrame(data = RP, columns = ['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataY], axis = 1)
    
    plt.close()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('Two Component RP ({})'.format(dataset), fontsize = 20)
    targets = list(set(data_df['class'].copy().values))
    for label in targets:
        indicesToKeep = finalDf['class']==label
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],finalDf.loc[indicesToKeep, 'principal component 2'])
    ax.legend(targets)
    ax.grid()
    plt.savefig(out+"{}_RP_2D.png".format(dataset))

###
##SVD
###

for dataset in ['biodeg.csv_header','voice.csv']:
    print("Working on",dataset,"data set...")

    data_df = pd.read_csv(dataset) 
    if  dataset == "biodeg.csv_header":      
        dataX = data_df.iloc[:,:41]
        dataY = data_df.iloc[:,41]
        n_components_range = [2,5,10,15,20,25,30,35,40]
        dataset = "QSAR"
        comps = np.int32(np.linspace(2, 39,20))
    else:
        dataX = data_df.iloc[:,:20]
        dataY = data_df.iloc[:,20]
        n_components_range = [2,5,10,15] 
        dataset = "VOICE"  
        comps = np.int32(np.linspace(2, 19,20))
    dataX = StandardScaler().fit_transform(dataX.astype('float64'))
    #######################################################
    split = train_test_split(dataX, dataY, test_size = 0.3,
    random_state = 42)
    (trainData, testData, trainTarget, testTarget) = split
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    model = LinearSVC()
    model.fit(trainData, trainTarget)
    baseline = metrics.accuracy_score(model.predict(testData), testTarget)
    print("Running RP...")
    accuracies = []
    for comp in comps:
        # create the random projection
        #sp = SparseRandomProjection(n_components = comp)
        #X = sp.fit_transform(trainData)
        #sp = PCA(n_components = comp, random_state=5)
        #X = sp.fit_transform(trainData)
        sp = SVD(n_components=comp)
        X = sp.fit_transform(trainData)
        # train a classifier on the sparse random projection
        model = LinearSVC()
        model.fit(X, trainTarget)
    
        # evaluate the model and update the list of accuracies
        test = sp.transform(testData)
        accuracies.append(metrics.accuracy_score(model.predict(test), testTarget))

    plt.figure()
    plt.suptitle("Accuracy of Sparse Projection on {}".format(dataset))
    plt.xlabel("# of Components")
    plt.ylabel("Accuracy")
    if dataset =="QSAR":
        plt.xlim([2, 40])
    else:
        plt.xlim([2, 20])    
    plt.ylim([0, 1.0])
     
    # plot the baseline and random projection accuracies
    plt.plot(comps, [baseline] * len(accuracies), color = "r")
    plt.plot(comps, accuracies)
    plt.grid()
    plt.savefig(out+"{}_SVD_Accuracy.png".format(dataset))
    #######################################################    

    print("Running SVD...")
    loss = []
    for num_components in n_components_range:
        svd = SVD(n_components=num_components)
        dataX_SVD = svd.fit_transform(dataX)
        dataX_projected = svd.inverse_transform(dataX_SVD)
        loss.append(((dataX - dataX_projected) ** 2).mean())
    plt.close()
    plt.figure()
    plt.grid()
    plt.title('Loss in Reconstruction ({})'.format(dataset))
    plt.plot(n_components_range,loss,'-o')
    plt.xlabel('Components')
    plt.ylabel('MSE')
    plt.savefig(out+"{}_SVD_Loss.png".format(dataset))

    print("Visualizing components...")
    svd_2D = SVD(n_components=2)
    SV = svd_2D.fit_transform(dataX)
    principalDf = pd.DataFrame(data = SV, columns = ['component 1', 'component 2'])
    finalDf = pd.concat([principalDf, dataY], axis = 1)
    
    plt.close()
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('Two Component SVD ({})'.format(dataset), fontsize = 20)
    targets = list(set(data_df['class'].copy().values))
    for label in targets:
        indicesToKeep = finalDf['class']==label
        ax.scatter(finalDf.loc[indicesToKeep, 'component 1'],finalDf.loc[indicesToKeep, 'component 2'])
    ax.legend(targets)
    ax.grid()
    plt.savefig(out+"{}_SVD_2D.png".format(dataset))

