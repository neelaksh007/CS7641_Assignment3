#!/usr/bin/env python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.preprocessing import StandardScaler

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

def cluster_acc(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target   
    return acc(Y,pred)

# Load data
out = 'step1/'

# Set up k-means and produce plots
for dataset in ['biodeg.csv','voice.csv']:
    print("\nWorking on",dataset,"data set...")
    data_df = pd.read_csv(dataset)
    if  dataset == "biodeg.csv":      
        dataX = data_df.iloc[:,:41]
        dataY = data_df.iloc[:,41]
        dataset = "QSAR"
    else:
        dataX = data_df.iloc[:,:20]
        dataY = data_df.iloc[:,20]
        dataset = "Voice"
    #dataX = StandardScaler().fit_transform(dataX.astype('float64'))
    dataY[dataY == -1] = 0
    y = np.bincount(dataY.astype(int))

    # Plot benchmark cluster distribution
    plt.close()
    plt.figure()
    plt.bar(range(y.size), y, align='center')
    plt.xticks(range(y.size), range(y.size))
    plt.title("Class Distribution ({})".format(dataset))
    plt.ylabel("Number of Samples")
    plt.xlabel("Class")
    plt.savefig(out+'{}_bmk_clusters.png'.format(dataset))

    # k-means clustering and associated plots
    print("Starting k-means...")
    model = KMeans(random_state=5)

    print("Prepping silhoutte plot...")
    plt.close()
    plt.figure()
    if dataset =="QSAR":
        visualizer = KElbowVisualizer(model, metric='silhouette', k=[2,5,10,15,20,25,30,35,40])
    else:
        visualizer = KElbowVisualizer(model, metric='silhouette', k=[2,5,10,15,20])   
    visualizer.fit(dataX)    # Fit the data to the visualizer
    visualizer.poof(outpath=out+"{}_kmeans_sil.png".format(dataset))    # Draw/show/poof the data

    print("Prepping distortion plot...")
    plt.close()
    plt.figure()
    if dataset =="QSAR":
        visualizer = KElbowVisualizer(model, metric='distortion', k=[2,5,10,15,20,25,30,35,40])
    else:
        visualizer = KElbowVisualizer(model, metric='distortion', k=[2,5,10,15,20])
    visualizer.fit(dataX)    # Fit the data to the visualizer
    visualizer.poof(outpath=out+"{}_kmeans_distortion.png".format(dataset))    # Draw/show/poof the data

    print("Prepping CH plot...")
    plt.close()
    plt.figure()
    if dataset =="QSAR":
        visualizer = KElbowVisualizer(model, metric='calinski_harabaz', k=[2,5,10,15,20,25,30,35,40])
    else:
        visualizer = KElbowVisualizer(model, metric='calinski_harabaz', k=[2,5,10,15,20])
    visualizer.fit(dataX)    # Fit the data to the visualizer
    visualizer.poof(outpath=out+"{}_kmeans_CH.png".format(dataset))    # Draw/show/poof the data

    print("Validating k-means labels....")
    k=10
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(dataX)
    accuracy = cluster_acc(dataY,labels)
    print("\nAccuracy for k-means on",dataset,"is",accuracy)
    y = np.bincount(labels)
    print("Clusters:",[(i,y[i]) for i in range(y.size)])

    # Plot kmeans cluster distribution
    plt.close()
    plt.figure()
    plt.bar(range(y.size), y, align='center')
    plt.xticks(range(y.size), range(y.size))
    plt.title("K-Means Cluster Distribution ({})".format(dataset))
    plt.ylabel("Number of Samples")
    plt.xlabel("Cluster")
    plt.savefig(out+'{}_kmeans_clusters.png'.format(dataset))
    plt.close()
    ##SSE'
    if dataset == "QSAR":
        range_n_clusters = [2,5,10,15,20,25,30,35,40]
    else:
        range_n_clusters = [2,5,10,15,20]   
    Sum_of_squared_distances = []
    for n_clusters in range_n_clusters:
        print ("Kmeans_QSAR: working on cluster {}".format(n_clusters))
        X_norm = StandardScaler().fit_transform(dataX)
        kmeans = KMeans(n_clusters=n_clusters).fit(X_norm)
        centroids = kmeans.cluster_centers_
        Sum_of_squared_distances.append(kmeans.inertia_)
        #plt.scatter(X_norm[:, 0], X_norm[:, 1], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
        #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
        #plt.savefig("Kmeans_graphs/Fig_QSAR_kmeans_clust_{}".format(n_clusters),format='png', bbox_inches='tight', dpi=150)
    fig_81 = plt.figure()
    plt.plot(range_n_clusters, Sum_of_squared_distances)
    #plt.bar(range(y.size), y, align='center')
    #plt.xticks(range(y.size), range(y.size))
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    print ("I AM DATASET - {}".format(dataset))
    if dataset == "QSAR":
        plt.title('Elbow Method For Optimal k - QSAR')
    else:
        plt.title('Elbow Method For Optimal k - Voice')    
    if dataset == "QSAR":
        fig_81.savefig("step1/SSE_Kmeans_QSAR.png",format='png', bbox_inches='tight', dpi=150)
    else:
        fig_81.savefig("step1/SSE_Kmeans_VOICE.png",format='png', bbox_inches='tight', dpi=150)    
    #SSE


    # EM clustering and associated plots
    print("\nStarting EM...")
    if dataset == "QSAR":
        clusters = [2,5,10,15,20,25,30,35,40]
    else:
        clusters = [2,5,10,15,20]   
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(dataX)
          for n in clusters]

    print("Plotting BIC/LL...")
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Clusters')
    ax1.set_ylabel('BIC', color=color)
    ax1.plot(clusters, [m.bic(dataX) for m in models], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Log Likelihood', color=color)  # we already handled the x-label with ax1
    ax2.plot(clusters, [m.score(dataX) for m in models], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title("Evaluation of E-M Metrics - {}".format(dataset))
    plt.savefig(out+"{}_EM_BIC_LL.png".format(dataset))

    print("Plotting adjusted mutual info...")
    plt.close()
    plt.figure()
    plot_ami = [ami(dataY,m.predict(dataX),average_method='arithmetic') for m in models]
    plt.plot(clusters, plot_ami)
    plt.xlabel('Clusters')
    plt.ylabel('Adjusted Mutual Information')
    plt.title("Performance of E-M {}".format(dataset))
    plt.savefig(out+"{}_EM_AMI.png".format(dataset))

    print("Validating EM labels....")
    if dataset == 'QSAR':
        k = 5
    else:
        k = 10
    model = GaussianMixture(k, covariance_type='full', random_state=0)
    labels = model.fit_predict(dataX)
    accuracy = cluster_acc(dataY,labels)
    print("\nAccuracy for E-M on",dataset,"is",accuracy)
    
    y = np.bincount(labels)
    print("Clusters:",[(i,y[i]) for i in range(y.size)])

    # Plot EM cluster distribution
    plt.close()
    plt.figure()
    plt.bar(range(y.size), y, align='center')
    plt.xticks(range(y.size), range(y.size))
    plt.title("E-M Cluster Distribution ({})".format(dataset))
    plt.ylabel("Number of Samples")
    plt.xlabel("Cluster")
    plt.savefig(out+'{}_EM_clusters.png'.format(dataset))
