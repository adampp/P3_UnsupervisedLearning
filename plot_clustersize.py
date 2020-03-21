import numpy as np
import time
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import cluster

def plot_clustersize(clusterAlgo, dataSet, dimReducer, nClustersUB, randomState):
    X, Y = dataSet.load()
    X = dimReducer.reduce(dataSet)
    
    fig, ax = plt.subplots(1, 2, figsize = (12, 5))
    
    scores = []
    sses = []
    for i in range(2,nClustersUB):
        tester = deepcopy(clusterAlgo)
        tester.nClusters = i
        tester.cluster(X)
        scores.append(metrics.silhouette_score(X, tester.getLabels(), random_state=randomState))
        sses.append(tester.getClusterSSEs())
    ax[0].plot(range(2,nClustersUB), scores)
    ax[0].set_title(clusterAlgo.plotTitle + ' Silhouette Score on ' + dataSet.plotFileName + ' with ' + dimReducer.plotTitle)
    ax[0].set_xlabel('number of clusters')
    ax[0].set_ylabel('silhouette score')
    ax[0].grid()
    
    ax[1].plot(range(2,nClustersUB), sses)
    ax[1].set_title(clusterAlgo.plotTitle + ' SSE on ' + dataSet.plotFileName + ' with ' + dimReducer.plotTitle)
    ax[1].set_xlabel('number of clusters')
    ax[1].set_ylabel('SSE')
    ax[1].grid()
    
    plt.savefig(f"plots/{dataSet.plotFileName}_{clusterAlgo.fileName}_{dimReducer.fileName}_ClusterSize.png", format='png')

    return plt