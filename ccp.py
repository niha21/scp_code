"""
 Author: Niharika gauraha
 Synergy Conformal Prediction Using Random Forest Classifier
"""

from sklearn.model_selection import train_test_split
import icp
import perf_measure as pm
import numpy as np
import json
from prettytable import PrettyTable
from sklearn.model_selection import ShuffleSplit
import os
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def CCP(X, y, n_source = 3, method='linear_svm', path='filename'):
    listCCPOF = []  # empty list
    x = PrettyTable()
    x.field_names = ["Validity", "Efficiency", "ErrorRate", "Observed Fuzziness"]
    n_labels = len(np.unique(y))

    for i in range(10):
        XX, X_test, yy, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=i)

        scaler = MinMaxScaler()
        scaler.fit(XX)
        XX = scaler.transform(XX)
        X_test = scaler.transform(X_test)

        meanPValues = np.zeros((len(y_test), n_labels))

        #sss = StratifiedShuffleSplit(n_splits=n_source, test_size=1/n_source)
        sss = ShuffleSplit(n_splits=n_source, test_size=1/n_source, random_state=i)
        for train_index, test_index in sss.split(XX, yy):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_calib = XX[train_index], XX[test_index]
            y_train, y_calib = yy[train_index], yy[test_index]
            calibPredProb, testPredProb = icp.ICPClassification(X_train, y_train, X_calib, X_test,
                                                                method = method)
            srcMCListConfScores = icp.computeConformityScores(calibPredProb, y_calib)
            pValues = icp.computePValues(srcMCListConfScores, testPredProb)
            meanPValues = np.add(meanPValues, pValues)

        meanPValues = meanPValues/n_source
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(meanPValues, y_test)
        #pm.CalibrationPlot(meanPValues, y_test, color='b')
        listCCPOF.append(obsFuzz)

    if not os.path.exists('json_ccp'):
        os.makedirs('json_ccp')

    with open(path, 'w') as fh:
        fh.write(json.dumps(listCCPOF))

    return listCCPOF

if __name__ == '__main__':
    import dataset_preprocessing as data


    epsilon = 0.1
    iteration = 10
    nrSources = 3

    method = "linear_svm"

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

    #dataset_names = ['Cover']

    for dataset_name in dataset_names:
        X, y = load_functions[dataset_name]()

        if not os.path.exists('json_ccp'):
            os.makedirs('json_ccp')

        file_name = "json_ccp/" + dataset_name + method+".json"

        print(dataset_name)
        CCP(X, y, n_source=3, method=method, path=file_name)
