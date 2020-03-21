import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve


def plot_model_complexity(estimator, X, y, title, xlabel, param_name, param_range, cv=None, axes=None, ylim=None, n_jobs=None):

    # if axes is None:
        # _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    _, axes = plt.subplots(1, 2, figsize=(15, 5))
    # axes = [axes]
    resultStr = ""

    for i in range(2):
        axes[i].set_title(title)
        if ylim is not None:
            axes[i].set_ylim(*ylim)
        axes[i].set_xlabel(xlabel[i])
        axes[i].set_ylabel("Score")

        train_scores, test_scores = \
            validation_curve(estimator, X, y, param_name=param_name[i], param_range=param_range[i], cv=cv, n_jobs=n_jobs)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plot learning curve
        axes[i].grid()
        axes[i].fill_between(param_range[i], train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[i].fill_between(param_range[i], test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[i].plot(param_range[i], train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[i].plot(param_range[i], test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[i].legend(loc="best")

        resultStr = resultStr + f"{param_name[i]}: test score={np.max(test_scores_mean)}, index={np.argmax(test_scores_mean)}, param={param_range[i][np.argmax(test_scores_mean)]}"
        
        if i < 1:
            resultStr = resultStr + "\n"
        
    print(resultStr)
    
    return plt, resultStr
