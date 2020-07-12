"""
 Author: Niharika gauraha
 Synergy Conformal Prediction using whole data and
 three different methods
"""
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
import numpy as np
import icp as icp
import perf_measure as pm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import OrderedDict


x = PrettyTable()
x.field_names = ["Dataset", "SVR-ICP", "RF-ICP", "RBF-SVR-ICP", 'SCP']
epsilon = 0.1
iterate = 10



def synergyCP(X, y, methods = None, path = None):
    n_source = len(methods)
    listIndSrcVal = []  # empty list
    listSCPVal = []  # empty list
    listIndSrcOF = []  # empty list
    listSCPOF = []  # empty list
    for i in range(n_source):
        listIndSrcOF.append([])

    for i in range(iterate):
        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=i)

        #scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train, X_calib, y_train, y_calib \
            = train_test_split(X_train, y_train, test_size=0.3, random_state=i)

        n_labels = len(np.unique(y_train))
        meanCalibPredProb = np.zeros((len(y_calib), n_labels))
        meanTestPredProb = np.zeros((len(y_test), n_labels))

        for indexSrc in range(n_source):
            sourceData = X_train
            sourceTarget = y_train
            calibPredProb, testPredProb = icp.ICPClassification(sourceData, sourceTarget, X_calib,\
                                                        X_test, method=methods[indexSrc])

            srcMCListConfScores = icp.computeConformityScores(calibPredProb, y_calib)
            pValues = icp.computePValues(srcMCListConfScores, testPredProb)
            errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
            listIndSrcVal.append(val)
            listIndSrcOF[indexSrc].append(obsFuzz)

            meanCalibPredProb = np.add(meanCalibPredProb, calibPredProb)
            meanTestPredProb = np.add(meanTestPredProb, testPredProb)

        meanCalibPredProb = meanCalibPredProb / n_source
        meanTestPredProb = meanTestPredProb / n_source
        srcMCListConfScores = icp.computeConformityScores(meanCalibPredProb, y_calib)
        pValues = icp.computePValues(srcMCListConfScores, meanTestPredProb)
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        listSCPVal.append(val)
        listSCPOF.append(obsFuzz)

    results = OrderedDict()
    results["source1"] = listIndSrcOF[0]
    results["source2"] = listIndSrcOF[1]
    results["source3"] = listIndSrcOF[2]
    results["SCP_OF"] = listSCPOF
    results["SCP_val"] = listSCPVal

    import json
    with open(path, 'w') as fh:
        fh.write(json.dumps(results))

    print(path, np.round(np.median(listIndSrcOF[0]), 3),
          np.round(np.median(listIndSrcOF[1]), 3),
          np.round(np.median(listIndSrcOF[2]), 3),
          np.round(np.median(listSCPOF), 3))


if __name__ == '__main__':
    import os
    import dataset_preprocessing as data

    np.random.seed(123)
    epsilon = 0.1
    iteration = 10
    nrSources = 3

    methods = ["linear_svm", "RF", "svm"]

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

    #dataset_names = ['BC']

    for dataset_name in dataset_names:
        X, y = load_functions[dataset_name]()

        if not os.path.exists('json_same_data'):
            os.makedirs('json_same_data')

        file_name = "json_same_data/" + dataset_name + ".json"

        print(dataset_name)
        synergyCP(X, y, methods=methods, path=file_name)