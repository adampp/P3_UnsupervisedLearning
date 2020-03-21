import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import metrics
from sklearn import cluster

from data_baby import *
from data_adult import *

from cluster_kmeans import *
from cluster_em import *

from dr_pca import *
from dr_ica import *
from dr_rp import *
from dr_fa import *
from dr_nodr import *

from plot_clusters import *
from plot_clustersize import *

randomState = 1

###########################
stepNum = 1
dataset = 'adult'
cluster = 'em'
dr = 'ica'
run = 'cluster'
updateTxt = True
###########################

nClusters = 4
nComponents = 4

if dataset == 'adult':
    dataSet = AdultData()
elif dataset == 'baby':
    dataSet = BabyData()

X, Y = dataSet.load()
# for i in range(1, len(X.columns)):
    # nComponents = i
    
    
for i in range(2):
    if i == 0:
        dataset = 'adult'
    elif i == 1:
        dataset = 'baby'
    for j in range(2):
        if j == 0:
            cluster = 'em'
        elif j == 1:
            cluster = 'kmeans'
        for k in range(4,5):
            if k == 0:
                dr = 'nodr'
            elif k == 1:
                dr = 'pca'
            elif k == 2:
                dr = 'ica'
            elif k == 3:
                dr = 'fa'
            elif k == 4:
                dr = 'rp'
                
                
            if dr == 'pca':
                if dataset == 'adult':
                    nComponents = 6
                elif dataset == 'baby':
                    nComponents = 9
                dimReducer = PCADR(nComponents = nComponents, randomState = randomState)
                
            elif dr == 'ica':
                if dataset == 'adult':
                    nComponents = 7# because non-gaussian kurtosis
                elif dataset == 'baby':
                    nComponents = 7# same as above
                dimReducer = ICADR(nComponents = nComponents, randomState = randomState)
                
            elif dr == 'rp':
                if dataset == 'adult':
                    nComponents = 6
                elif dataset == 'baby':
                    nComponents = 10
                dimReducer = RPDR(nComponents = nComponents, randomState = randomState)
                
            elif dr == 'fa':
                if dataset == 'adult':
                    nComponents = 5#one less than the number of non-zero rows in Unmixing matrix
                elif dataset == 'baby':
                    nComponents = 8#one less than number of non-zero rows in Unmixing matrix
                dimReducer = FADR(nComponents = nComponents, randomState = randomState)
                
            elif dr == 'nodr':
                dimReducer = NoDR(randomState = randomState)
            
            print(f"{dataset}-{cluster}-{dr}")
            
            if dataset == 'baby':
                dataSet = BabyData()
                if cluster == 'em':
                    if dr == 'nodr':
                        nClusters = 2
                    elif dr == 'pca':
                        nClusters = 2#highest silhouette, no different from no dr
                    elif dr == 'ica':
                        nClusters = 5#peak silhouette, small trough SSE
                    elif dr == 'fa':
                        nClusters = 2#highest silhouette, no elbow in sse, no peak in silhouette
                    elif dr == 'rp':
                        nClusters = 5#local maxima silhouette
                    clusterAlg = EMClustering(nClusters = nClusters, randomState = randomState)
                elif cluster == 'kmeans':
                    if dr == 'nodr':
                        nClusters = 2
                    elif dr == 'pca':
                        nClusters = 2#same as no dr
                    elif dr == 'ica':
                        nClusters = 26# peak silhouette, SSE curve is smooth
                    elif dr == 'fa':
                        nClusters = 8# highest silhouette, SSE curve is smooth
                    elif dr == 'rp':
                        nClusters = 7# local maxima silhouette, 2nd highest, SSE curve is smooth
                    clusterAlg = KMeansClustering(nClusters = nClusters, randomState = randomState)
                        
            elif dataset == 'adult':
                dataSet = AdultData()
                if cluster == 'em':
                    if dr == 'nodr':
                        nClusters = 2#peak silhouette
                    elif dr == 'pca':
                        nClusters = 2#peak silhouette
                    elif dr == 'ica':
                        nClusters = 27# peak silhouette, 2nd lowest SSE
                    elif dr == 'fa':
                        nClusters = 3#peak silhouette
                    elif dr == 'rp':
                        nClusters = 2#*based on only 1 rp
                    clusterAlg = EMClustering(nClusters = nClusters, randomState = randomState)
                elif cluster == 'kmeans':
                    if dr == 'nodr':
                        nClusters = 2
                    elif dr == 'pca':
                        nClusters = 2#similar to graph with no dr
                    elif dr == 'ica':
                        nClusters = 15# peakish silhouette
                    elif dr == 'fa':
                        nClusters = 3# peak silhouette
                    elif dr == 'rp':
                        nClusters = 2#*based on only 1 rp
                    clusterAlg = KMeansClustering(nClusters = nClusters, randomState = randomState)
                    
            if run == 'cluster':
                plot_clusters(clusterAlg, dataSet, dimReducer)
            elif run == 'clustersize':
                plot_clustersize(clusterAlg, dataSet, dimReducer, 30, randomState)
                
                
# # dimReducer = RPDR(nComponents = nComponents, randomState = randomState)
# # dimReducer = FADR(randomState = randomState)
            # result = dimReducer.reduce(dataSet)
            # dimReducer.createPlots()

