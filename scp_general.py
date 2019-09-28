"""
 Author: Niharika gauraha
 Synergy Conformal Prediction Using RF, RBF SVM, and Linear SVM
"""

from sklearn.model_selection import train_test_split
import random
import icp
import perf_measure as pm
import numpy as np
import matplotlib.pylab as plt
from prettytable import PrettyTable
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from dataset_preprocessing import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler


boxPlot = True
x = PrettyTable()
x.field_names = ["Dataset", "Individual", "SCP"]

def synergyCP(X, y, n_source = 3, file = "BC"):
    listIndSrcErrRate = []  # empty list
    listSCPErrRate = []  # empty list
    listIndSrcEff = []  # empty list
    listSCPEff = []  # empty list
    listIndSrcVal = []  # empty list
    listSCPVal = []  # empty list
    listIndSrcOF = []  # empty list
    listSCPOF = []  # empty list

    methods = ["linear_svm", "rf", "svm"]

    for i in range(10):
        X_train, X_test, y_train, y_test \
            = train_test_split(X, y, test_size=0.2)

        X_train, X_calib, y_train, y_calib \
            = train_test_split(X_train, y_train, test_size=1/3)

        scaler = MinMaxScaler()
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
            calibPredProb, testPredProb = icp.ICPClassification(sourceData, sourceTarget,
                                                                X_calib, X_test, method = methods[indexSrc])
            #if indexSrc == 0:
            srcMCListConfScores = icp.computeConformityScores(calibPredProb, y_calib)
            pValues = icp.computePValues(srcMCListConfScores, testPredProb)
            errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
            print(obsFuzz)
            if indexSrc == 0:
                listIndSrcVal.append(val)
                listIndSrcErrRate.append(errRate)
                listIndSrcEff.append(eff)
                listIndSrcOF.append(obsFuzz)
                #pm.CalibrationPlot(pValues, y_test, color='orange')

            meanCalibPredProb = np.add(meanCalibPredProb, calibPredProb)
            meanTestPredProb = np.add(meanTestPredProb, testPredProb)

            trainIndex = randIndex[splitLen * (indexSrc + 1):splitLen * (indexSrc + 2)]

        meanCalibPredProb = meanCalibPredProb / n_source
        meanTestPredProb = meanTestPredProb / n_source
        srcMCListConfScores = icp.computeConformityScores(meanCalibPredProb, y_calib)
        pValues = icp.computePValues(srcMCListConfScores, meanTestPredProb)
        errRate, eff, val, obsFuzz = pm.pValues2PerfMetrics(pValues, y_test)
        #pm.CalibrationPlot(pValues, y_test, color='b')
        listSCPVal.append(val)
        listSCPErrRate.append(errRate)
        listSCPEff.append(eff)
        listSCPOF.append(obsFuzz)
        print(obsFuzz)

    if boxPlot:
        boxPlotLabels = ["Individual", "Synergy"]
        data = np.column_stack((listIndSrcOF, listSCPOF))
        plt.boxplot(data, labels = boxPlotLabels, patch_artist=True)
        plt.ylabel("Observed Fuzziness")
        #plt.show()
        plt.savefig("ObservedFuzz"+file)
        plt.clf()

    x.add_row([file, np.mean(listIndSrcOF), np.mean(listSCPOF)])



if 1:
    X = load_breast_cancer()
    y = X['target']
    X = X['data']
    X = preprocessing.scale(X)
    y[ y == -1] = 0
    synergyCP(X, y, n_source=3, file="BC")

    X, y = load_spambase_data() # Done
    y[ y == -1] = 0
    synergyCP(X, y, n_source=3, file="Spambase")

    X, y = load_Phishing_dataset('data/phising.csv') # with delimiter ','
    y[ y == -1] = 0
    synergyCP(X, y, n_source=3, file="Phishing")

    X, y = load_dataset('data/australian.dat') # delimiter space ' '
    y[ y == -1] = 0
    synergyCP(X, y, n_source=3, file="Australian")

    X, y = load_categorical_dataset('data/tic-tac-toe.data') # done
    y[ y == -1] = 0
    synergyCP(X, y, n_source=3, file="Tic-tac-toe")

    X, y = load_bank_dataset('data/bank.csv')
    y[ y == -1] = 0
    synergyCP(X, y, n_source=3, file="Bank")


    X, y = load_adult_dataset('data/adult.data')
    y[ y == -1] = 0
    synergyCP(X, y, n_source=3, file="Adult")

    X,y = load_monks_data()
    y[ y == -1] = 0
    synergyCP(X, y, n_source=3, file="Monks-1")

    X, y = load_monks2_data()
    y[y == -1] = 0
    synergyCP(X, y, n_source=3, file="Monks-1")

    X, y = load_covertype_dataset('data/covtype.data')
    y[y == -1] = 0
    synergyCP(X, y, n_source=3, file="Monks-1")

    print(x)
