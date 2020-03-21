import numpy as np
import time
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import cluster

def plot_elbow(clusterAlgo, dataSet, nClustersUB, randomState):
    fig, ax = plt.subplots(1, 1)
    
    scores = []
    for i in range(2,nClustersUB):
        tester = deepcopy(clusterAlgo)
        tester.nClusters = i
        tester.cluster(dataSet)
        scores.append(tester.model.inertia_)
    ax.plot(range(2,nClustersUB), scores)
    ax.set_title(clusterAlgo.plotTitle + ' SSE on ' + dataSet.plotFileName)
    plt.savefig(f"plots/{dataSet.plotFileName}_{clusterAlgo.fileName}_Elbow.png", format='png')

    return plt