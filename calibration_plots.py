"""
 Author: Niharika gauraha
 Synergy Conformal Prediction Using Random Forest Classifier
"""

from sklearn.model_selection import train_test_split
import random
import icp
import perf_measure as pm
import numpy as np
import matplotlib.pylab as plt
from prettytable import PrettyTable
from sklearn.datasets import load_breast_cancer
from dataset_preprocessing import *

def synergyCP(X, y, n_source = 3):
    listIndSrcErrRate = []  # empty list
    listSCPErrRate = []  # empty list
    listIndSrcEff = []  # empty list
    listSCPEff = []  # empty list
    listIndSrcVal = []  # empty list
    listSCPVal = []  # empty list
    listIndSrcOF = []  # empty list
    listSCPOF = []  # empty list

    for i in range(1):
        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2)

        X_train, X_calib, y_train, y_calib \
            = train_test_split(X_train, y_train, test_size=0.3)

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
                                                                X_test, method = "rf", nrTrees = 10)
            if indexSrc == 0:
                srcMCListConfScores = icp.computeConformityScores(calibPredProb, y_calib)
                pValues = icp.computePValues(srcMCListConfScores, testPredProb)
                errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
                listIndSrcVal.append(val)
                listIndSrcErrRate.append(errRate)
                listIndSrcEff.append(eff)
                listIndSrcOF.append(obsFuzz)
                pm.CalibrationPlot(pValues, y_test, color='orange')

            meanCalibPredProb = np.add(meanCalibPredProb, calibPredProb)
            meanTestPredProb = np.add(meanTestPredProb, testPredProb)
            trainIndex = randIndex[splitLen * (indexSrc + 1):splitLen * (indexSrc + 2)]

        meanCalibPredProb = meanCalibPredProb / n_source
        meanTestPredProb = meanTestPredProb / n_source
        srcMCListConfScores = icp.computeConformityScores(meanCalibPredProb, y_calib)
        pValues = icp.computePValues(srcMCListConfScores, meanTestPredProb)
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        pm.CalibrationPlot(pValues, y_test, color='b')
        print(val)


if 0:
    X = load_breast_cancer()
    y = X['target']
    X = X['data']
    X = preprocessing.scale(X)
    y[ y == -1] = 0


if 1:
    X, y = load_spambase_data() # Done
    #X, y = load_monks_data()

    y[ y == -1] = 0



def ACP(X, y, n_source = 3):
    listACPOF = []  # empty list
    x = PrettyTable()
    x.field_names = ["Validity", "Efficiency", "ErrorRate", "Observed Fuzziness"]
    n_labels = len(np.unique(y))

    for i in range(1):
        XX, X_test, yy, y_test \
            = train_test_split(X, y, test_size=0.2)

        meanPValues = np.zeros((len(y_test), n_labels))

        #sss = StratifiedShuffleSplit(n_splits=n_source, test_size=1/n_source)
        sss = ShuffleSplit(n_splits=n_source, test_size=1/n_source)
        for train_index, test_index in sss.split(XX, yy):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_calib = XX[train_index], XX[test_index]
            y_train, y_calib = yy[train_index], yy[test_index]
            calibPredProb, testPredProb = icp.ICPClassification(X_train, y_train, X_calib, X_test,
                                                                method = "rf", nrTrees = 10)
            srcMCListConfScores = icp.computeConformityScores(calibPredProb, y_calib)
            pValues = icp.computePValues(srcMCListConfScores, testPredProb)
            meanPValues = np.add(meanPValues, pValues)

        meanPValues = meanPValues/n_source
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(meanPValues, y_test)
        pm.CalibrationPlot(meanPValues, y_test, color='b')
        listACPOF.append(obsFuzz)

    print(val)
    return listACPOF


synergyCP(X, y, n_source=3)
#acp.ACP(X, y, n_source=3)
