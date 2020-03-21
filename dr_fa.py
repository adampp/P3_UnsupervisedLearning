import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import decomposition

class FADR():
    plotTitle = "FA"
    fileName = "FA"
    def __init__(self, nComponents=None, copy=True, tol=0.01, maxIter=1000, noiseVarianceInit=None,
        svdMethod='randomized', iteratedPower=3, randomState=None):
        self.nComponents = nComponents
        self.copy = copy
        self.tol = tol
        self.maxIter = maxIter
        self.noiseVarianceInit = noiseVarianceInit
        self.svdMethod = svdMethod
        self.iteratedPower = iteratedPower
        self.randomState = randomState
        
    def reduce(self, dataSet):
        self.dataSet = dataSet
        X, Y = dataSet.load()
        self.nFeatures = len(X.columns)
        self.model = decomposition.FactorAnalysis(n_components=self.nComponents, copy=self.copy, tol=self.tol,
            max_iter=self.maxIter, noise_variance_init=self.noiseVarianceInit, svd_method=self.svdMethod,
            iterated_power=self.iteratedPower, random_state=self.randomState)
        
        X = self.model.fit_transform(X)
        
        return X
        
    def createPlots(self):
        fig, ax = plt.subplots(1, 2, figsize = (12, 5))
        if self.nComponents is None:
            nComponents = self.nFeatures
        else:
            nComponents = self.nComponents
        ax[0].plot(range(self.nFeatures), self.model.noise_variance_)
        ax[0].set_title('FA Noise Variance for ' + self.dataSet.plotTitle)
        ax[0].set_xlabel('component index')
        ax[0].set_ylabel('noise variance')
        
        pos = ax[1].imshow(self.model.components_)
        fig.colorbar(pos, ax=ax[1])
        ax[1].set_title('Unmixing Matrix for ' + self.dataSet.plotTitle)
        ax[1].set_xlabel('input index')
        ax[1].set_ylabel('factor index')

        plt.savefig(f"plots/{self.dataSet.plotFileName}_FA.png", format='png')