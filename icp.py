#############################################
# Author: Niharika Gauraha
# ICP: Inductive Conformal Prediction
#        for Classification using RF
#############################################

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

cv=6

def computeConformityScores(predProb, calibTarget):
    nrCases, nrLabels = predProb.shape
    category = np.unique(calibTarget.astype(np.int))
    calibLabels = calibTarget.astype(np.int)
    MCListConfScores = []  # Moderian Class wise List of conformity scores
    for i in range(0, nrLabels):
        clsIndex = np.where(calibLabels == category[i])
        classMembers = predProb[clsIndex, i]
        MCListConfScores.append(classMembers[0]) #MCListConfScores[i]+ classMembers.tolist()[0]

    return MCListConfScores


def computePValues(MCListConfScores, testConfScores):

    if (MCListConfScores is None) or (testConfScores is None):
        sys.exit("\n NULL model \n")

    nrTestCases, nrLabels = testConfScores.shape
    pValues = np.zeros((nrTestCases,  nrLabels))

    for k in range(0, nrTestCases):
        for l in range(0, nrLabels):
            alpha = testConfScores[k, l]
            classConfScores = np.ndarray.flatten(np.array(MCListConfScores[l]))
            pVal = len(classConfScores[np.where(classConfScores < alpha)]) + (np.random.uniform(0, 1, 1) * \
                len(classConfScores[np.where(classConfScores == alpha)]))
            tempLen = len(classConfScores)
            pValues[k, l] = pVal/(tempLen + 1)

    return(pValues)


def ICPClassification(trainData, trainTarget, calibData, testData, method="rf", nrTrees=100):
    if (trainData is None) or (calibData is None):
        sys.exit("\n 'training set' and 'calibration set' are required as input\n")

    if method =="linear_svm":
        print("Linear SVM")
        calibPredProb, testPredProb = linear_SVC(trainData, trainTarget, calibData, testData)
    elif method == "svm":
        print("SVM")
        calibPredProb, testPredProb = cv_SVC(trainData, trainTarget, calibData, testData)
    else:
        print("RF")
        calibPredProb, testPredProb = RF(trainData, trainTarget, calibData, testData, nrTrees)

    return calibPredProb, testPredProb


def cv_SVC(X_train, y_train, calibData, testData):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, .1, 1e-2, 1e-3, 1e-4],
                         'C': [1, 10, 100]}
                        ]
    clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=cv, iid=True)
    clf.fit(X_train, y_train)
    calibPredProb = clf.predict_proba(calibData)
    testPredProb = clf.predict_proba(testData)
    '''
    # Platt's scaling
    y_predict = clf.decision_function(trainData)
    lr = LR()
    lr.fit(y_predict.reshape(-1, 1), trainTarget.ravel())  # make it 2-dimensional
    y_predict = clf.decision_function(calibData)
    calibPredProb = lr.predict_proba(y_predict.reshape(-1, 1))[:, 1]
    calibPredProb = np.column_stack((calibPredProb, 1-calibPredProb))
    y_predict = clf.decision_function(testData)
    testPredProb = lr.predict_proba(y_predict.reshape(-1, 1))[:, 1]
    testPredProb = np.column_stack((testPredProb , 1-testPredProb))
    '''
    return calibPredProb, testPredProb

def linear_SVC(X_train, y_train, calibData, testData):
    tuned_parameters = [{'C': [1, 10, 100]}]
    clf = GridSearchCV(SVC(probability=True, kernel='linear'), tuned_parameters, cv=cv, iid=True)
    clf.fit(X_train, y_train)
    calibPredProb = clf.predict_proba(calibData)
    testPredProb = clf.predict_proba(testData)

    return calibPredProb, testPredProb

def RF(X_train, y_train, calibData, testData, nrTrees):
    model = RandomForestClassifier(n_estimators=nrTrees)
    model.fit(X_train, y_train)
    calibPredProb = model.predict_proba(calibData)
    testPredProb = model.predict_proba(testData)

    return calibPredProb, testPredProb
