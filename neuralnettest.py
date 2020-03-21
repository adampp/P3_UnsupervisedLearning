import mlrose_reborn as mlrose
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics

from plot_learning_curve import *
from plot_nn_learning_curve import *
from plot_model_complexity import *

from data_baby import *
from data_adult import *

from dr_pca import *
from dr_ica import *
from dr_rp import *
from dr_fa import *
from dr_nodr import *

from cluster_kmeans import *
from cluster_em import *

from learner_neuralnet import *

randomState = 1
testSize = 0.15
kfolds = 7

models = [0, 0, 0]

###########################
stepNum = 1
dr = 'pcakmeans'
run = 'test'
updateTxt = True
###########################

dataSet = BabyData()
X, Y = dataSet.load()

learner = NeuralNetwork(hidden_layer_sizes=(5,), activation='relu', solver='sgd', alpha=1e-4,
    learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
if dr == 'pca':
    nComponents = 9
    dimReducer = PCADR(nComponents = nComponents, randomState = randomState)
    learner = NeuralNetwork(hidden_layer_sizes=(20,), activation='relu', solver='sgd', alpha=1e-4,
        learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
    
elif dr == 'ica':
    nComponents = 7# same as above
    dimReducer = ICADR(nComponents = nComponents, randomState = randomState)
    learner = NeuralNetwork(hidden_layer_sizes=(20,), activation='tanh', solver='sgd', alpha=1e-4,
        learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
    X = dimReducer.reduce(dataSet)
    
elif dr == 'rp':
    nComponents = 10
    dimReducer = RPDR(nComponents = nComponents, randomState = stepNum)
    X = dimReducer.reduce(dataSet)
    learner = NeuralNetwork(hidden_layer_sizes=(5,), activation='relu', solver='sgd', alpha=1e-4,
        learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
    
elif dr == 'fa':
    nComponents = 8#one less than number of non-zero rows in Unmixing matrix
    dimReducer = FADR(nComponents = nComponents, randomState = randomState)
    learner = NeuralNetwork(hidden_layer_sizes=(20,), activation='relu', solver='sgd', alpha=1e-4,
        learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
    X = dimReducer.reduce(dataSet)
    
elif dr == 'kmeans':
    nClusters = 2#from baseline clustering assessment of baby
    clusterAlg = KMeansClustering(nClusters = nClusters, randomState = randomState)
    clusterAlg.cluster(dataSet)
    labels = clusterAlg.getLabels()
    X = np.concatenate((X, labels[:,None]), axis=1)
    dimReducer = NoDR(randomState = randomState)
    dimReducer.fileName = 'KMeans'
    learner = NeuralNetwork(hidden_layer_sizes=(5,), activation='relu', solver='sgd', alpha=1e-4,
        learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)

elif dr == 'em':
    nClusters = 2#from baseline clustering assessment of baby
    clusterAlg = EMClustering(nClusters = nClusters, randomState = randomState)
    clusterAlg.cluster(dataSet)
    labels = clusterAlg.getLabels()
    X = np.concatenate((X, labels[:,None]), axis=1)
    dimReducer = NoDR(randomState = randomState)
    dimReducer.fileName = 'EM'
    learner = NeuralNetwork(hidden_layer_sizes=(10,), activation='tanh', solver='sgd', alpha=1e-4,
        learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
        
elif dr == 'pcakmeans':
    nComponents = 9
    dimReducer = PCADR(nComponents = nComponents, randomState = randomState)
    nClusters = 2#from baseline clustering assessment of baby
    clusterAlg = KMeansClustering(nClusters = nClusters, randomState = randomState)
    clusterAlg.cluster(dataSet)
    labels = clusterAlg.getLabels()
    X = dimReducer.reduce(dataSet)
    X = np.concatenate((X, labels[:,None]), axis=1)
    learner = NeuralNetwork(hidden_layer_sizes=(10,), activation='relu', solver='sgd', alpha=1e-4,
        learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
    
elif dr == 'nodr':
    dimReducer = NoDR(randomState = randomState)
    learner = NeuralNetwork(hidden_layer_sizes=(40,), activation='relu', solver='sgd', alpha=1e-4,
        learning_rate='constant', learning_rate_init=0.01, max_iter=500, random_state=randomState)
    
print(np.shape(X))

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size = testSize,
    random_state = randomState)
    

if run == 'tune':
    samplePercent = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    title = f"{learner.plotTitle} Learning Curve on {dataSet.plotTitle}"
    plt, resultStr = plot_learning_curve(learner.learner, title, xTrain, yTrain, cv=kfolds,
        train_sizes=samplePercent, ylim=(0.60, 1.01), n_jobs = 4)
    plt.savefig(f"plots/{dataSet.plotFileName}_{learner.plotFileName}_{dimReducer.fileName}_{stepNum}_LearningCurve.png",
        format='png')

    if updateTxt:
        with open(f"plots/{dataSet.plotFileName}_{dimReducer.fileName}_{learner.plotFileName}.txt", "a") as file:
            file.write("====================================================\n")
            file.write(f"LearningCurve-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:"+"\n")
            file.write(resultStr+"\n")

    epochs = np.linspace(20, 500, 20)
    plot_nn_learning_curve(learner.learner, title, xTrain, yTrain, cv=kfolds, train_epochs=epochs,
        testSize=testSize, ylim=None)
    plt.savefig(f"plots/{dataSet.plotFileName}_{learner.plotFileName}_{dimReducer.fileName}_{stepNum}_NNLearningCurve.png",
        format='png')
        
        
    gridSearch = model_selection.GridSearchCV(learner.learner, param_grid = learner.paramGrid, cv=kfolds)
    gridSearch.fit(xTrain, yTrain)
    print("====================================================")
    print(f"GridSearch-{stepNum}-{dataSet.plotFileName}-{dimReducer.fileName}-{learner.plotFileName}:")
    print(f"Score={gridSearch.best_score_}, params={gridSearch.best_params_}")

    if updateTxt:
        with open(f"plots/{dataSet.plotFileName}_{dimReducer.fileName}_{learner.plotFileName}.txt", "a") as file:
            file.write("====================================================\n")
            file.write(f"GridSearch-{stepNum}-{dataSet.plotFileName}-{dimReducer.fileName}-{learner.plotFileName}:"+"\n")
            file.write(f"Score={gridSearch.best_score_}, params={gridSearch.best_params_}"+"\n")
            file.write(resultStr+"\n")
elif run == 'test':
    learner.learner.fit(xTrain, yTrain)
    
    train_results = learner.learner.score(xTrain,yTrain)
    test_results = learner.learner.score(xTest,yTest)
    print("====================================================")
    print(f"FinalTest-{stepNum}-{dataSet.plotFileName}-{learner.plotFileName}:")
    print(f"Train={train_results}, Test={test_results}")
    
    if updateTxt:
        with open(f"plots/{dataSet.plotFileName}_{dimReducer.fileName}_{learner.plotFileName}.txt", "a") as file:
            file.write("====================================================\n")
            file.write(f"FinalTest-{stepNum}-{dataSet.plotFileName}-{dimReducer.fileName}-{learner.plotFileName}:"+"\n")
            file.write(f"Train={train_results}, Test={test_results}"+"\n")
