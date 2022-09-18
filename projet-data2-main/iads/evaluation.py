# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 
def crossval(X, Y, n_iterations, iteration):
    #############
    # A COMPLETER
    #############
    Xm = X
    Ym = Y
    seq_it = len(X) // n_iterations
    Xapp = []
    Yapp = []
    Xtest = np.empty((0,0))
    Ytest = np.empty((0,0))
    for i in range(n_iterations):
        Xcurr = Xm[seq_it*i:seq_it*(i+1)].copy()
        Ycurr = Ym[seq_it*i:seq_it*(i+1)].copy()
        if i == iteration:
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
            Xtest = np.append(Xtest,Xcurr)
            Ytest = np.append(Ytest,Ycurr)
        else:
            Xapp = np.array(Xapp)
            Yapp = np.array(Yapp)
            Xapp = np.append(Xapp,Xcurr)
            Yapp = np.append(Yapp,Ycurr)
    Xapp = Xapp.reshape((len(Xapp)//len(X[0]),len(X[0])))
    Xtest = Xtest.reshape((len(Xtest)//len(X[0]),len(X[0])))
    return Xapp, Yapp, Xtest, Ytest

def crossval_strat(X, Y, n_iterations, iteration):
    #############
    # A COMPLETER
    #############
    index_neg = np.where(Y == -1)
    index_pos = np.where(Y == 1)
    index_neg = index_neg[0]
    index_pos = index_pos[0]
    X_pos = X[Y == 1]
    X_neg = X[Y == -1]
    Y_pos = Y[Y == 1]
    Y_neg = Y[Y == -1]
    seq_it = len(X) // (n_iterations*2)
    Xapp = []
    Yapp = []
    Xtest = np.empty((0,0))
    Ytest = np.empty((0,0))
    for i in range(n_iterations):
        index_curr_neg = index_neg[seq_it*i:seq_it*(i+1)]
        index_curr_pos = index_pos[seq_it*i:seq_it*(i+1)]
        index_curr = np.squeeze(np.sort(np.hstack((index_curr_neg,index_curr_pos))))
        index_bool = []
        Xcurr = np.empty((0,0))
        Ycurr = np.empty((0,0))
        for elt in index_curr:
            Xcurr = np.append(Xcurr,X[elt])
            Ycurr = np.append(Ycurr,Y[elt])
        if i == iteration:
            Xtest = np.array(Xtest)
            Ytest = np.array(Ytest)
            Xtest = np.append(Xtest,Xcurr)
            Ytest = np.append(Ytest,Ycurr)
        else:
            Xapp = np.array(Xapp)
            Yapp = np.array(Yapp)
            Xapp = np.append(Xapp,Xcurr)
            Yapp = np.append(Yapp,Ycurr)
            
    
    Xapp = Xapp.reshape((len(Xapp)//len(X[0]),len(X[0])))
    Xtest = Xtest.reshape((len(Xtest)//len(X[0]),len(X[0])))
    return Xapp, Yapp, Xtest, Ytest


def analyse_perfs(perf):
    """
    Retourne le tuple (mean,variance)
    """
    return np.mean(perf),np.var(perf)
