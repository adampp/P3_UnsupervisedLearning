import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import mixture

class EMClustering():
    plotTitle = "Expectation Maximization"
    fileName = "EM"
    def __init__(self, nClusters=1, covarianceType='full', tol=0.001, regCovar=1e-6, maxIter=100,
        nInit=1, initParams='kmeans', randomState=None):
        self.nClusters = nClusters
        self.covarianceType = covarianceType
        self.tol = tol
        self.regCovar = regCovar
        self.maxIter = maxIter
        self.nInit = nInit
        self.initParams = initParams
        self.randomState = randomState
        self.model = None
        self.dataSet = None
        self.X = None
        
    def cluster(self, dataSet):
        if hasattr(dataSet, 'plotTitle'):
            self.dataSet = dataSet
            X, Y = dataSet.load()
            X = X.values
        else:
            X = dataSet
            
        self.model = mixture.GaussianMixture(n_components=self.nClusters,
            covariance_type=self.covarianceType, tol=self.tol, reg_covar=self.regCovar,
            max_iter=self.maxIter, n_init=self.nInit, init_params=self.initParams,
            random_state=self.randomState)
        
        self.model.fit(X)
        self.X = X
        
        return self.model.means_
        
    def getLabels(self):
        if self.model is None:
            return None
        
        probs = self.model.predict_proba(self.X)
        
        labels = np.zeros((len(probs)))
        
        for i in range(len(probs)):
            labels[i] = np.argmax(probs[i,:])
            
        return labels.astype('int')
        
    def getClusterSSEs(self):
        if self.model is None:
            return None
        
        labels = self.getLabels()
        
        sses = np.zeros((self.nClusters))
        for i in range(len(self.X)):
            sses[labels[i]] += np.sqrt(np.sum(np.square(self.model.means_[labels[i],:] - self.X[i,:])))
            
        return np.sum(sses)