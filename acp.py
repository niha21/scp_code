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


def ACP(X, y, n_source = 20, method="linear_svm"):
    listACPOF = []  # empty list
    x = PrettyTable()
    x.field_names = ["Validity", "Efficiency", "ErrorRate", "Observed Fuzziness"]
    n_labels = len(np.unique(y))

    for i in range(10):
        XX, X_test, yy, y_test \
            = train_test_split(X, y, test_size=0.2)

        meanPValues = np.zeros((len(y_test), n_labels))
        for j in range(n_source):
            X_train, X_calib, y_train, y_calib = train_test_split(XX,yy, test_size=1/3, stratify=yy)

            calibPredProb, testPredProb = icp.ICPClassification(X_train, y_train, X_calib, X_test,
                                                                method = method)
            srcMCListConfScores = icp.computeConformityScores(calibPredProb, y_calib)
            pValues = icp.computePValues(srcMCListConfScores, testPredProb)
            meanPValues = np.add(meanPValues, pValues)

        meanPValues = meanPValues/n_source
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(meanPValues, y_test)
        #pm.CalibrationPlot(meanPValues, y_test, color='b')
        listACPOF.append(obsFuzz)

    print(val)
    return listACPOF


