import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import cluster



def plot_clusters(clusterAlgo, dataSet, dimReducer):
    X = dimReducer.reduce(dataSet)
    
    t0 = time.time()
    k_means_cluster_centers = clusterAlgo.cluster(X)
    t_batch = time.time() - t0

    colors = ['#EA6E50', '#B7D655', '#28F5FC', '#029423',
        '#3D5DDE', '#7F46CA', '#ECA11B', '#E197C6', '#68E0AA',
        '#04C4D4', '#CA9562', '#973000', '#B8036C', '#E5B3AC',
        '#0F0EB0']

    k_means_labels = metrics.pairwise.pairwise_distances_argmin(X, k_means_cluster_centers)
    # KMeans
    completedIdxs = []
    for i in range(np.size(X,1)):
        for j in range(np.size(X,1)):
            fig, ax = plt.subplots(1, 1)
            if i == j:
                continue
            if j in completedIdxs:
                continue
            completedIdxs.append(i)
            for k, col in zip(range(clusterAlgo.nClusters), colors):
                my_members = k_means_labels == k
                ax.plot(X[my_members, i], X[my_members, j], 'w',
                    markerfacecolor=col, marker='.', linestyle='None')
                        
            for k, col in zip(range(clusterAlgo.nClusters), colors):
                cluster_center = k_means_cluster_centers[k]
                ax.plot(cluster_center[i], cluster_center[j], 'o', markerfacecolor=col,
                        markeredgecolor='k', markersize=6)
            
            if dimReducer.fileName == 'NoDR':
                ax.set_title(clusterAlgo.plotTitle + ' Cluster Visualization for\n' + dimReducer.columns[i] + ' & ' + dimReducer.columns[j])
                ax.set_xlabel(dimReducer.columns[i])
                ax.set_ylabel(dimReducer.columns[j])
            else:
                ax.set_title(clusterAlgo.plotTitle + ' Cluster Visualization for\n' + str(i) + ' & ' + str(j))
                ax.set_xlabel('component ' + str(i))
                ax.set_ylabel('component ' + str(j))

            plt.savefig(f"plots/clustering/{dataSet.plotFileName}/{clusterAlgo.fileName}/{dimReducer.fileName}/{dataSet.plotFileName}_{clusterAlgo.fileName}_{i}_{j}.png", format='png')
            plt.close('all')