import pandas as pd
import numpy as np
from sklearn import preprocessing

class BabyData:
    def __init__(self):
        self.filepath = "data/baby.csv"
        self.delimiter = ','
        self.outputMap = {1: 1, 2: 2, 3: 3}
        self.labels=[]
        self.numerics=[]
        self.plotTitle = "Baby Dataset"
        self.plotFileName = "Baby"
        
        X, Y = self.load()
        self.nFeatures = len(X.columns)
        
    def load(self):
        fid = open(self.filepath)
        data = pd.read_csv(fid, header = 0, delimiter = self.delimiter, engine = 'python')
        data.columns = data.columns.str.strip()
        
        # data.iloc[:,-1] = data.iloc[:,-1].map(self.outputMap)
        
        X = data.iloc[:,0:-1]
        
        # X = pd.get_dummies(X, columns = self.labels)

        labelEncoder = preprocessing.LabelEncoder()
        for i in range(len(self.labels)):
            X[self.labels[i]] = labelEncoder.fit_transform(X[self.labels[i]])

        numericScaler = preprocessing.MinMaxScaler()
        for i in range(21):
            X.iloc[:,i] = numericScaler.fit_transform(pd.DataFrame(X.iloc[:,i]))
        
        Y = data.iloc[:,-1]
        
        # onehot = preprocessing.OneHotEncoder()
        # Y = onehot.fit_transform(Y.values.reshape(-1,1)).todense()
        
        return X, Y