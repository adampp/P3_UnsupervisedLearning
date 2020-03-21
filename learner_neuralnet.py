from sklearn import neural_network
import numpy as np
from learner_base import *
from math import *

class NeuralNetwork(BaseLearner):
    plotTitle = "Neural Network"
    plotFileName = "NeuralNet"
    paramGrid = [{'hidden_layer_sizes': [(5,), (10,), (20,), (40,)], 'activation': ['tanh', 'relu'], 'learning_rate_init': [0.01, 0.001, 0.0001]}]
    paramNames = ['hidden_layer_sizes', 'learning_rate_init']
    
    def __init__(self, hidden_layer_sizes, activation, solver, alpha, learning_rate, learning_rate_init, max_iter, random_state):
        self.learner = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, max_iter=max_iter, random_state=random_state)
            
        self.paramRanges = [np.ceil(np.linspace(hidden_layer_sizes[0]*self.paramLB, hidden_layer_sizes[0]*self.paramUB, self.numLinSpace)).astype(int),
            np.linspace(learning_rate_init*self.paramLB, learning_rate_init*self.paramUB, self.numLinSpace)]