import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import decomposition
from sklearn import random_projection

class RPDR():
    plotTitle = "RP"
    fileName = "RP"
    def __init__(self, nComponents='auto', eps=0.1, randomState=None):
        self.nComponents = nComponents
        self.eps = eps
        self.randomState = randomState
        self.model = []
        
    def reduce(self, dataSet, randomState=None):
        if randomState is None:
            randomState = self.randomState
        self.dataSet = dataSet
        X, Y = dataSet.load()
        self.X_orig = X.values
        self.nFeatures = len(X.columns)
        if self.nComponents == 'auto':
            self.nComponents = len(X.columns)
        self.model.append(random_projection.GaussianRandomProjection(n_components=self.nComponents,
            eps=self.eps, random_state=randomState))
        
        X = self.model[-1].fit_transform(X)
        
        return X
        
    def createPlots(self):
        fig, ax = plt.subplots(1, 1)
        if self.nComponents is None:
            nComponents = self.nFeatures
        else:
            nComponents = self.nComponents
            
        avgReconstructionError = np.zeros(len(self.X_orig[0,:]))
        for i in range(len(self.X_orig[0,:])):
            for j in range(10):
                self.nComponents = i+1
                X = self.reduce(self.dataSet, randomState=j)
                A = np.linalg.pinv(self.model[-1].components_)
                X_reconstructed = np.dot(X,A.T)
                # avgReconstructionError[i] += np.sum(np.sqrt(np.sum(np.square(X_reconstructed - self.X_orig), 0))) / len(self.X_orig[:,0])
                avgReconstructionError[i] += np.sum(np.square(X_reconstructed - self.X_orig))
            
        ax.plot(range(1,len(self.X_orig[0,:])+1), avgReconstructionError)
        ax.set_title('Reconstruction Error for ' + self.dataSet.plotTitle)
        ax.set_xlabel('number of components')
        ax.set_ylabel('SSE')
        

        plt.savefig(f"plots/{self.dataSet.plotFileName}_RP_{self.nComponents}_Plots.png", format='png')