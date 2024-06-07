""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import numpy as np
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# def evaluate(labels, scores, metric='roc'):
#     if metric == 'roc':
#         return roc(labels, scores)
#     elif metric == 'auprc':
#         return auprc(labels, scores)
#     elif metric == 'f1_score':
#         threshold = 0.20
#         scores[scores >= threshold] = 1
#         scores[scores <  threshold] = 0
#         return f1_score(labels, scores)
#     else:
#         raise NotImplementedError("Check the evaluation metric.")

def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels.cpu(), scores.cpu())
    elif metric == 'auprc':
        return auprc(labels.cpu(), scores.cpu())
    elif metric == 'f1_score':
        return f1(labels.cpu(), scores.cpu())
    elif metric == 'accuracy':
        return accuracy(labels.cpu(), scores.cpu())
    elif metric == 'sens':
        return sens(labels.cpu(), scores.cpu())
    elif metric == 'spec':
        return spec(labels.cpu(), scores.cpu())
    else:
        raise NotImplementedError("Check the evaluation metric.")

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap

def f1(labels, scores):
    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    f1 = f1_score(labels, scores > optimal_threshold)

    return f1

def accuracy(labels, scores):
    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    accuracy = accuracy_score(labels, scores > optimal_threshold)

    return accuracy

def sens(labels, scores):
    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    sensitivity = tpr[optimal_idx]
    # specificity = 1 - fpr[optimal_idx]

    return sensitivity

def spec(labels, scores):
    fpr, tpr, threshold = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]

    return specificity

##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap