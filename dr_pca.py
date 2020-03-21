import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import decomposition

class PCADR():
    plotTitle = "PCA"
    fileName = "PCA"
    def __init__(self, nComponents=None, copy=True, whiten=False, svdSolver='auto', tol=0.0, iteratedPower='auto', randomState=None):
        self.nComponents = nComponents
        self.copy = copy
        self.whiten = whiten
        self.svdSolver = svdSolver
        self.tol = tol
        self.iteratedPower = iteratedPower
        self.randomState = randomState
        
    def reduce(self, dataSet):
        self.dataSet = dataSet
        X, Y = dataSet.load()
        self.nFeatures = len(X.columns)
        self.model = decomposition.PCA(n_components=self.nComponents, copy=self.copy, whiten=self.whiten,
            svd_solver=self.svdSolver, tol=self.tol, iterated_power=self.iteratedPower, random_state=self.randomState)
        
        X = self.model.fit_transform(X)
        
        return X
        
    def createPlots(self):
        fig, ax = plt.subplots(1, 2, figsize = (12, 5))
        if self.nComponents is None:
            nComponents = self.nFeatures
        else:
            nComponents = self.nComponents
            
        pos = ax[0].imshow(self.model.components_)
        fig.colorbar(pos, ax=ax[0])
        ax[0].set_title('Component Matrix for ' + self.dataSet.plotTitle)
        ax[0].set_xlabel('input index')
        ax[0].set_ylabel('component index')
        
        ax[1].plot(self.model.explained_variance_, range(nComponents))
        ax[1].set_title('PCA Explained Variance for ' + self.dataSet.plotTitle)
        ax[1].set_xlabel('explained variance')
        ax[1].set_ylabel('component index')
        
        ax[1].invert_yaxis()

        plt.savefig(f"plots/{self.dataSet.plotFileName}_PCA_Plots.png", format='png')