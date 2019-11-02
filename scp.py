"""
 Author: Niharika gauraha
 Synergy Conformal Prediction Using Random Forest Classifier
"""
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import random
import icp
import perf_measure as pm
from prettytable import PrettyTable
from dataset_preprocessing import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler


boxPlot = True
x = PrettyTable()
x.field_names = ["Dataset", "Individual", "SCP"]


def synergyCP(X, y, n_source = 3, methods = None, path = None):
    # initialize lists
    listIndSrcVal = []  # empty list
    listSCPVal = []  # empty list
    listIndSrcOF = []  # empty list
    listSCPOF = []  # empty list
    for i in range(n_source):
        listIndSrcOF.append([])

    if methods is None:
        methods = ['linear_svm'] * n_source

    for i in range(10):
        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2, random_state=i)

        X_train, X_calib, y_train, y_calib \
            = train_test_split(X_train, y_train, test_size=.3, random_state=i)

        # scale data to be in the same range
        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        nrTrainCases = len(y_train)
        n_labels = len(np.unique(y_train))
        randIndex = random.sample(list(range(0, nrTrainCases)), nrTrainCases)
        splitLen = int(nrTrainCases / n_source)
        # split training data into equal parts
        trainIndex = randIndex[0:splitLen]

        meanCalibPredProb = np.zeros((len(y_calib), n_labels))
        meanTestPredProb = np.zeros((len(y_test), n_labels))
        for indexSrc in range(0, n_source):
            sourceData = X_train[trainIndex, :]
            sourceTarget = y_train[trainIndex]
            calibPredProb, testPredProb = icp.ICPClassification(sourceData, sourceTarget, X_calib,\
                                                                X_test, method=methods[indexSrc])

            srcMCListConfScores = icp.computeConformityScores(calibPredProb, y_calib)
            pValues = icp.computePValues(srcMCListConfScores, testPredProb)
            errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
            listIndSrcVal.append(val)
            listIndSrcOF[indexSrc].append(obsFuzz)

            meanCalibPredProb = np.add(meanCalibPredProb, calibPredProb)
            meanTestPredProb = np.add(meanTestPredProb, testPredProb)
            trainIndex = randIndex[splitLen * (indexSrc + 1):splitLen * (indexSrc + 2)]

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

    #x.add_row([path, np.median(listIndSrcOF), np.median(listSCPOF)])
    #print(x)

#just to compute the partition size
def computeSize(X, y):
    X_train, X_test, y_train, y_test \
        = train_test_split(X, y, test_size=0.2)
    X_train, X_calib, y_train, y_calib \
        = train_test_split(X_train, y_train, test_size=.3)
    print(len(X_train), len(X_calib), len(X_test), X_train.shape[1])


if __name__=="__main__":
    # Unit test
    import dataset_preprocessing as data
    X, y = data.load_bank_dataset()
    import os
    if not os.path.exists('json'):
        os.makedirs('json')

    file_name = "json/" + "temp.json"
    synergyCP(X, y, n_source=3, path=file_name)

