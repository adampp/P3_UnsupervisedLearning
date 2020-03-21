import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import cluster

class KMeansClustering():
    plotTitle = "K-Means"
    fileName = "KMeans"
    def __init__(self, nClusters=8, nInit=10, maxIter=300, tol=1e-4, randomState=None):
        self.nClusters = nClusters
        self.nInit = nInit
        self.maxIter = maxIter
        self.tol = tol
        self.randomState = randomState
        self.model = None
        self.dataSet = None
        
    def cluster(self, dataSet):
        if hasattr(dataSet, 'plotTitle'):
            self.dataSet = dataSet
            X, Y = dataSet.load()
            X = X.values
        else:
            X = dataSet
            
        self.model = cluster.KMeans(n_clusters=self.nClusters, init='random',
            n_init=self.nInit, max_iter=self.maxIter, tol=self.tol, 
            precompute_distances='auto', verbose=0, random_state = self.randomState,
            copy_x=True, n_jobs=None, algorithm='auto')
        
        self.model.fit(X)
        self.X = X
        
        return self.model.cluster_centers_
        
    def getLabels(self):
        if self.model is None:
            return None
        
        return self.model.labels_
        
    def getClusterSSEs(self):
        if self.model is None:
            return None
        
        return self.model.inertia_