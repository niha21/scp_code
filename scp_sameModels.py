"""
 Author: Niharika gauraha
 Synergy conformal prediction using linear SVM classifier
"""
import matplotlib
matplotlib.use('Agg')
import dataset_preprocessing as data
from collections import OrderedDict
from scp import synergyCP
import os
import perf_measure as pm


epsilon = 0.1
iteration = 10
nrSources = 3
methods = ['svm', 'linear_svm', 'RF']

dataset_names = ['SB', 'BC', 'Phishing', 'Cover', 'Adult', 'Tic',
                     'Aus', 'Monk-1', 'Monk-2', 'Bank']

load_functions = OrderedDict()
load_functions["SB"] = data.load_spambase_data
load_functions["BC"] = data.load_breast_cancer_data
load_functions["Phishing"] = data.load_Phishing_dataset
load_functions["Cover"] = data.load_covertype_dataset
load_functions["Adult"] = data.load_adult_dataset
load_functions["Tic"] = data.load_tic_tac_toe_dataset
load_functions["Aus"] = data.load_australian_dataset
load_functions["Monk-1"] = data.load_monks1_data
load_functions["Monk-2"] = data.load_monks2_data
load_functions["Bank"] = data.load_bank_dataset

dataset_names = ['BC']

for dataset_name in dataset_names:
    X, y = load_functions[dataset_name]()

    if not os.path.exists('json_sameModel_linear'):
        os.makedirs('json_sameModel_linear')

    file_name = "json_sameModel_linear/" + dataset_name + ".json"

    print(dataset_name)
    synergyCP(X, y, n_source=nrSources, methods=methods, path=file_name)


