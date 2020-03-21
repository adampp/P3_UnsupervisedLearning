from math import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import random

class AdultData:
    def __init__(self):
        self.filepath = "data/adult.data"
        self.delimiter = ', '
        self.outputMap = {"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1}
        self.labels=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
        self.numerics=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        self.plotTitle = "Adult Dataset"
        self.plotFileName = "Adult"
        
        X, Y = self.load()
        self.nFeatures = len(X.columns)
        
    def load(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.iloc[:,-1] = data.iloc[:,-1].map(self.outputMap)
        
        idxs = random.sample_without_replacement(n_population=len(data), n_samples=ceil(len(data)*0.15), random_state = 1)
        data = data.iloc[idxs, :]
        
        X = data.iloc[:,0:-1]
        
        # X = pd.get_dummies(X, columns = self.labels)

        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(14):
            X.iloc[:,i] = numericScaler.fit_transform(pd.DataFrame(X.iloc[:,i]))
        
        data.iloc[:,0:-1] = X
        
        X = data.iloc[:,0:-1]
        Y = data.iloc[:,-1]
        
        
        return X, Y