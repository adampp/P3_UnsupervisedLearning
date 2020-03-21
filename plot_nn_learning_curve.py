import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from copy import deepcopy


def plot_nn_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        testSize=0.15, train_epochs=np.linspace(20, 500, 20)):
    # if axes is None:
        # _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=testSize, random_state=1)
    fig, axes = plt.subplots(1, 1)
    axes = [axes]

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Score")
    
    train_epochs = np.ceil(train_epochs).astype(int)
    
    train_scores = []
    test_scores = []
    for max_iter in train_epochs:
        tester = deepcopy(estimator)
        tester.set_params(max_iter = max_iter)
        tester = tester.fit(xTrain, yTrain)
        train_scores.append(tester.score(xTrain, yTrain))
        test_scores.append(tester.score(xTest, yTest))

    # Plot learning curve
    axes[0].grid()
    axes[0].plot(train_epochs, train_scores, '-', color="r",
                 label="Training score")
    axes[0].plot(train_epochs, test_scores, '-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # fig.text(0.5, 0.5, 'Adam Pierce',
     # fontsize=62, color='gray',
     # ha='center', va='center', alpha=0.8)

    return plt
