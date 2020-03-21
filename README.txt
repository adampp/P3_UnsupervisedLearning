Install/Setup Steps:
1. Download and Extract https://github.com/adampp/P2_RandomizedOptimization
    https://github.com/adampp/P2_RandomizedOptimization.git

2. Make sure scikit-learn, numpy, pandas, matplotlib, scipy, and all required dependencies are installed
with Python 3.6+
    pip3 install scikit-learn, numpy, pandas, matplotlib, scipy

3. Two main executable files, test.py which runs tests related to the 3 fitness functions to
optimize, and neuralnettest.py which does the NN optimization. 
They are executed as 'python3 test.py'

4. test.py is written to be executed manually multiple times in succession, manipulating the
constants at the top of the main function in order to do the necessary experiments. Some are
self-explanatory

    a. run constant defines what test you are doing. Available options are
        'clustersize', 'cluster'
    b. dataset defines which dataset to analyze. Available options are
        'adult' and 'baby'
    c. stepNum constant defines what step in the tuning process you are on. Every execution of
    experiments.py will append results to a txt file titled [ProblemName]_[Algorithm].txt.
    Additionally, most tests you can run will generate a .png image of a results graph. This stepNum
    constant helps keep the order you did your operations in straight when looking back through
    results, and the graphs in chronological order when sorting files by name.
    d. updateTxt constant defines if that execution will write to a file or not - useful if debugging
    new code.
    e. cluster defines which clustering algorithm to run. Available options are
        'em' and 'kmeans'
    f. dr defines which dimensionality reduction algorithm to run. Available options are
        'pca', 'ica', 'rp', and 'fa'
    g. Some commenting and uncommenting will be required to do various tests. For example, to run just
    dimensionality reduction against a dataset, the whole nested loop sequence will need to be commented
    
    
4. neuralnettest.py is written to be executed manually multiple times in succession, manipulating the
constants at the top of the main function in order to do the necessary experiments. Some are
self-explanatory, and this file is a little more crudely written

    a. run constant defines what test you are doing. Available options are
        'tune', 'test'
    b. stepNum constant defines what step in the tuning process you are on. Every execution of
    experiments.py will append results to a txt file titled NN_[Algorithm].txt.
    Additionally, most tests you can run will generate a .png image of a results graph. This stepNum
    constant helps keep the order you did your operations in straight when looking back through
    results, and the graphs in chronological order when sorting files by name.
    c. updateTxt constant defines if that execution will write to a file or not - useful if debugging
    new code.
    d. dr defines what datareduction to run. In this instance, clustering was included in this set of options.
        'pca', 'ica', 'rp', 'fa', 'kmeans', 'em', and 'pcakmeans'
    
5. Seperate files were produced for every dataset (all prefixed by data_*.py) and each cluster or dim reduction 
algo (all prefixed by cluster_*.py or dr_*.py as appropriate). These contain the necessary configuration for tests
(like ranges for configuration parameter analysis, or filenames, or titles). To customize these
parameters, please see the respective file.

Other Information
2. All raw data files are in the data/ folder, including many files not used in the report. I have
kept them in for completeness, as well as their associated data_*.py folder should the reader wish to
try some of these out. They are all already setup with the required pre-processing steps.

2. All results used in the report are also contained in plots/ folder. You may find quite a few more
graphs than included in the report, as was mentioned.