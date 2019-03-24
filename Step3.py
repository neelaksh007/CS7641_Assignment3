#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
import sys
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from itertools import product
from sklearn.metrics.pairwise import pairwise_distances
from cycler import cycler
from sklearn.metrics import accuracy_score as acc
from collections import Counter
from sklearn.decomposition import TruncatedSVD as SVD

def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target   
    return acc(Y,pred)

######
##PCA
######
print ('''
######
##PCA
######''')
for datst in ['biodeg.csv','voice.csv']:
	data = pd.read_csv(datst)
	if datst == 'biodeg.csv':
		dataX = data.iloc[:,:41]
		dataY = data.iloc[:,41]
		features = list(dataX.columns.values)
		dataset = "QSAR"
		k = 10
		k_em = 10
	else:
		dataX = data.iloc[:,:20]
		dataY = data.iloc[:,20]
		features = list(dataX.columns.values)	
		dataset = "Voice"
		k = 10
		k_em = 15
	print("Running PCA for {}...".format(datst))
	pca = PCA(n_components = 20, random_state=5)
	dataX_PCA = pca.fit_transform(dataX)
	
	#kmeans
	
	model = KMeans(n_clusters=k)
	labels_KM_PCA = model.fit_predict(dataX_PCA)
	model = KMeans(n_clusters=k)
	labels_KM = model.fit_predict(dataX)
	
	    
	accuracy = cluster_acc(dataY,labels_KM_PCA)
	print("\nAccuracy for k-means on",dataset,"is",accuracy)
	accuracy_clusters = cluster_acc(labels_KM,labels_KM_PCA)
	print("\nCluster alignment for k-means is",accuracy_clusters)
	
	#EM
	model = GaussianMixture(k_em, covariance_type='full', random_state=0)
	labels_EM_PCA = model.fit_predict(dataX_PCA)
	model = GaussianMixture(k_em, covariance_type='full', random_state=0)
	labels_EM = model.fit_predict(dataX)
	
	accuracy = cluster_acc(dataY,labels_EM_PCA)
	print("\nAccuracy for E-M on",dataset,"is",accuracy)
	accuracy_clusters = cluster_acc(labels_EM,labels_EM_PCA)
	print("\nCluster alignment for E-M is",accuracy_clusters)
	
	print("Ended PCA for {}...".format(datst))

######
##ICA
######
print ('''
######
##ICA
######''')
for datst in ['biodeg.csv','voice.csv']:
	data = pd.read_csv(datst)
	if datst == 'biodeg.csv':
		dataX = data.iloc[:,:41]
		dataY = data.iloc[:,41]
		features = list(dataX.columns.values)
		dataset = "QSAR"
		k = 10
		k_em = 10
		comp = 40
	else:
		dataX = data.iloc[:,:20]
		dataY = data.iloc[:,20]
		features = list(dataX.columns.values)	
		dataset = "Voice"
		k = 10
		k_em = 15
		comp = 20
	print("Running ICA for {}...".format(datst))
	ica = FastICA(n_components = comp, random_state=5)
	dataX_ICA = ica.fit_transform(dataX)
	
	#kmeans
	
	model = KMeans(n_clusters=k)
	labels = model.fit_predict(dataX_ICA)
	model = KMeans(n_clusters=k)
	labels_KM = model.fit_predict(dataX)

	accuracy = cluster_acc(dataY,labels)
	print("\nAccuracy for k-means on",dataset,"is",accuracy)
	accuracy_clusters = cluster_acc(labels_KM,labels)
	print("\nCluster alignment for k-means is",accuracy_clusters)
	
	#EM
	model = GaussianMixture(k_em, covariance_type='full', random_state=0)
	labels = model.fit_predict(dataX_ICA)
	model = GaussianMixture(k_em, covariance_type='full', random_state=0)
	labels_EM = model.fit_predict(dataX)

	accuracy = cluster_acc(dataY,labels)
	print("\nAccuracy for E-M on",dataset,"is",accuracy)
	accuracy_clusters = cluster_acc(labels_EM,labels)
	print("\nCluster alignment for E-M is",accuracy_clusters)
	
	print("Ended ICA for {}...".format(datst))

######
##RP
######
print ('''
######
##RP
######''')
for datst in ['biodeg.csv','voice.csv']:
	data = pd.read_csv(datst)
	if datst == 'biodeg.csv':
		dataX = data.iloc[:,:41]
		dataY = data.iloc[:,41]
		features = list(dataX.columns.values)
		dataset = "QSAR"
		k = 10
		k_em = 10
		comp = 10
	else:
		dataX = data.iloc[:,:20]
		dataY = data.iloc[:,20]
		features = list(dataX.columns.values)	
		dataset = "Voice"
		k = 10
		k_em = 15
		comp = 5
	print("Running RP for {}...".format(datst))
	rp = SparseRandomProjection(n_components = comp, random_state=5)
	dataX_RP = rp.fit_transform(dataX)
	
	#kmeans
	
	model = KMeans(n_clusters=k)
	labels = model.fit_predict(dataX_RP)
	model = KMeans(n_clusters=k)
	labels_KM = model.fit_predict(dataX)

	accuracy = cluster_acc(dataY,labels)
	print("\nAccuracy for k-means on",dataset,"is",accuracy)
	accuracy_clusters = cluster_acc(labels_KM,labels)
	print("\nCluster alignment for k-means is",accuracy_clusters)
	
	#EM
	model = GaussianMixture(k_em, covariance_type='full', random_state=0)
	labels = model.fit_predict(dataX_RP)
	model = GaussianMixture(k_em, covariance_type='full', random_state=0)
	labels_EM = model.fit_predict(dataX)

	accuracy = cluster_acc(dataY,labels)
	print("\nAccuracy for E-M on",dataset,"is",accuracy)
	accuracy_clusters = cluster_acc(labels_EM,labels)
	print("\nCluster alignment for E-M is",accuracy_clusters)
	
	print("Ended RP for {}...".format(datst))

######
##RP
######
print ('''
######
##SVD
######''')
for datst in ['biodeg.csv','voice.csv']:
	data = pd.read_csv(datst)
	if datst == 'biodeg.csv':
		dataX = data.iloc[:,:41]
		dataY = data.iloc[:,41]
		features = list(dataX.columns.values)
		dataset = "QSAR"
		k = 10
		k_em = 10
		comp = 39
	else:
		dataX = data.iloc[:,:20]
		dataY = data.iloc[:,20]
		features = list(dataX.columns.values)	
		dataset = "Voice"
		k = 10
		k_em =15
		comp = 19
	print("Running SVD for {}...".format(datst))
	svd = SVD(n_components = comp, random_state=5)
	dataX_SVD = svd.fit_transform(dataX)
	
	#kmeans
	
	model = KMeans(n_clusters=k)
	labels = model.fit_predict(dataX_SVD)
	model = KMeans(n_clusters=k)
	labels_KM = model.fit_predict(dataX)

	accuracy = cluster_acc(dataY,labels)
	print("\nAccuracy for k-means on",dataset,"is",accuracy)
	accuracy_clusters = cluster_acc(labels_KM,labels)
	print("\nCluster alignment for k-means is",accuracy_clusters)
	
	#EM
	if datst == 'biodeg.csv':
	    k = 10
	elif dataset == 'voice.csv':
	    k = 10

	model = GaussianMixture(k_em, covariance_type='full', random_state=0)
	labels = model.fit_predict(dataX_SVD)
	model = GaussianMixture(k_em, covariance_type='full', random_state=0)
	labels_EM = model.fit_predict(dataX)

	accuracy = cluster_acc(dataY,labels)
	print("\nAccuracy for E-M on",dataset,"is",accuracy)
	accuracy_clusters = cluster_acc(labels_EM,labels)
	print("\nCluster alignment for E-M is",accuracy_clusters)
	
	print("Ended SVD for {}...".format(datst))	









