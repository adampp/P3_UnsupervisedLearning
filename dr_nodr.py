import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import decomposition

class NoDR():
    plotTitle = "No DR"
    fileName = "NoDR"
    def __init__(self, randomState=None):
        self.randomState = randomState
        
    def reduce(self, dataSet):
        self.dataSet = dataSet
        X, Y = dataSet.load()
        self.nFeatures = len(X.columns)
        
        self.columns=X.columns
        
        return X.values
        
    def createPlots(self):
        pass