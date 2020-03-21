import numpy as np
import time
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import decomposition

class ICADR():
    plotTitle = "ICA"
    fileName = "ICA"
    def __init__(self, nComponents=None, algorithm='parallel', whiten=True, fun='logcosh',
        funArgs=None, maxIter=10000, tol=0.0001, wInit=None, randomState=None):
        self.nComponents = nComponents
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.funArgs = funArgs
        self.maxIter = maxIter
        self.tol = tol
        self.wInit = wInit
        self.randomState = randomState
        
    def reduce(self, dataSet):
        self.dataSet = dataSet
        X, Y = dataSet.load()
        self.nFeatures = len(X.columns)
        self.model = decomposition.FastICA(n_components=self.nComponents, algorithm=self.algorithm, whiten=self.whiten,
            fun=self.fun, fun_args=self.funArgs, max_iter=self.maxIter, tol=self.tol, w_init=self.wInit, random_state=self.randomState)
        
        self.X = self.model.fit_transform(X)
        
        return self.X
        
    def createPlots(self):
        fig, ax = plt.subplots(1, 2, figsize = (12, 5))
        if self.nComponents is None:
            nComponents = self.nFeatures
        else:
            nComponents = self.nComponents
            
        pos = ax[0].imshow(self.model.components_)
        fig.colorbar(pos, ax=ax[0])
        ax[0].set_title('Unmixing Matrix for ' + self.dataSet.plotTitle)
        ax[0].set_xlabel('input index')
        ax[0].set_ylabel('independent component index')
        
        kurt = np.zeros(len(self.X[0,:]))
        for i in range(len(self.X[0,:])):
            kurt[i] = stats.kurtosis(self.X[:,i])
            
        ax[1].plot(kurt, range(nComponents))
        ax[1].plot([3,3], [0,nComponents-1], 'r')
        ax[1].set_title('Independent Component Kurtosis for ' + self.dataSet.plotTitle)
        ax[1].set_xlabel('kurtosis')
        ax[1].set_ylabel('independent component index')
        ax[1].invert_yaxis()
        

        plt.savefig(f"plots/{self.dataSet.plotFileName}_ICA_{self.nComponents}_Plots.png", format='png')