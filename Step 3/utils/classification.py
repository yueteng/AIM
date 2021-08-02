import pickle
import random
import numpy as np
import re
import csv
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_curve, auc, precision_score, recall_score,f1_score
from keras import backend as K

dict = {}
data ,label = list(), list()

digits=3
color = ["red", "blue", "orange", "green", "yellow"]

from keras import backend as K


def Precision(y_true, y_pred):
    """精确率"""
    tp = np.sum(y_true * y_pred)  # true positives
    pp = np.sum(y_pred)  # predicted positives
    precision = tp /pp
    return precision


def Recall(y_true, y_pred):
    """召回率"""
    tp = np.sum(y_true * y_pred) # true positives
    pp = np.sum(y_true) # possible positives
    recall = tp / (pp + K.epsilon())
    return recall


def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return f1

def report(test_labels, test_predict ):
    # p = Precision(test_labels, test_predict)
    # r = Recall(test_labels, test_predict)
    # f1 = F1(test_labels, test_predict)
    # print("p:{}, r:{}, f1:{}".format(p, r, f1))
    #
    # p1 = precision_score(test_labels, test_predict)
    # r1 = recall_score(test_labels, test_predict)
    # f11 = f1_score(test_labels, test_predict)
    # print("p1:{}, r1:{}, f11:{}".format(p1, r1, f11))
    print(classification_report_imbalanced(test_labels, np.round(test_predict), digits=5))

# def roc(y_test, test_predict):
#     report(y_test, test_predict)
#
#     fpr, tpr, threshold = roc_curve(y_test, test_predict)
#     roc_auc = auc(fpr, tpr)
#     print("AUC:", roc_auc)
#     plt.figure()
#     lw = 2
#     plt.figure(figsize=(10, 10))
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     # plt.title('Receiver operating characteristic example')
#     plt.title('ROC curve')
#     plt.legend(loc="lower right")
#     plt.savefig("ROC.png", dpi=600)
#     plt.show()

def roc_list(y_test, test_predict, model):
    model = model.replace("_", " ")
    plt.figure(figsize=(10, 10))
    for i in range(5):
        print("model " +  str(i))
        report(y_test[i], np.round(test_predict[i]))
        fpr, tpr, threshold = roc_curve(y_test[i], test_predict[i])
        roc_auc = auc(fpr, tpr)
        print("AUC:", roc_auc)
        lw = 2
        plt.plot(fpr, tpr, color=color[i],
                 lw=lw, label=model+"_"+str(i) + ' (AUC = %0.5f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Compare the ROC of the base model')
    plt.title(model)
    # plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(model + "_ROC.png", dpi=600)
    plt.show()


def fold_roc_list(y_test, test_predict, model):
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, len(y_test))

    i = 0
    for i in range(5):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test[i], test_predict[i])
        tprs.append(np.interp(mean_fpr, fpr, tpr))  # 根据每一折的fpr和tpr进行插值，得插值曲线后，对相同的mean_fpr求其对应插值曲线上的纵坐标
        # interp一维线性插值，fpr和tpr是插值结点横纵坐标，mean_fpr是已知中间节点横坐标(得到插值曲线后，求其纵坐标)
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html#numpy.interp
        tprs[-1][0] = 0.0  # tprs有6个元素，每个元素是一个长度为100的array
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)  # aucs有6个auc的值，是交叉验证每一折的auc
        plt.plot(fpr, tpr, alpha=0.3, label=model + 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)  # 将每一折交叉验证计算的tpr求和取平均(每个位置对应相同的fpr)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=model + r' Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def all_fold_roc_list(y_test, test_predict, model):
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(5):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test[i], test_predict[i])
        tprs.append(np.interp(mean_fpr, fpr, tpr))  # 根据每一折的fpr和tpr进行插值，得插值曲线后，对相同的mean_fpr求其对应插值曲线上的纵坐标
        # interp一维线性插值，fpr和tpr是插值结点横纵坐标，mean_fpr是已知中间节点横坐标(得到插值曲线后，求其纵坐标)
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html#numpy.interp
        tprs[-1][0] = 0.0  # tprs有6个元素，每个元素是一个长度为100的array
        # roc_auc = auc(fpr, tpr)
        # aucs.append(roc_auc)  # aucs有6个auc的值，是交叉验证每一折的auc
        # plt.plot(fpr, tpr, alpha=0.3, label=model + 'ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        # i += 1



    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)  # 将每一折交叉验证计算的tpr求和取平均(每个位置对应相同的fpr)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=model + r' Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")