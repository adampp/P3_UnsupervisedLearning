import numpy as np
import time
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import cluster

def plot_silhouette(clusterAlgo, dataSet, nClustersUB, randomState):
    X, Y = dataSet.load()
    fig, ax = plt.subplots(1, 1)
    
    scores = []
    for i in range(2,nClustersUB):
        tester = deepcopy(clusterAlgo)
        tester.nClusters = i
        tester.cluster(dataSet)
        scores.append(metrics.silhouette_score(X, tester.getLabels(), random_state=randomState))
    ax.plot(range(2,nClustersUB), scores)
    ax.set_title(clusterAlgo.plotTitle + ' Silhouette Score on ' + dataSet.plotFileName)
    plt.savefig(f"plots/{dataSet.plotFileName}_{clusterAlgo.fileName}_Silhouette.png", format='png')

    return plt