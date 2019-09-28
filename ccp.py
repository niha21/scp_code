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
from sklearn.model_selection import ShuffleSplit


def CCP(X, y, n_source = 3):
    listCCPOF = []  # empty list
    x = PrettyTable()
    x.field_names = ["Validity", "Efficiency", "ErrorRate", "Observed Fuzziness"]
    n_labels = len(np.unique(y))

    for i in range(10):
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
                                                                method = "linear_svm")
            srcMCListConfScores = icp.computeConformityScores(calibPredProb, y_calib)
            pValues = icp.computePValues(srcMCListConfScores, testPredProb)
            meanPValues = np.add(meanPValues, pValues)

        meanPValues = meanPValues/n_source
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(meanPValues, y_test)
        #pm.CalibrationPlot(meanPValues, y_test, color='b')
        listCCPOF.append(obsFuzz)

    print(val)
    return listCCPOF


