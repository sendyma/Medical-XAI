from sklearn.metrics import roc_auc_score as AUROC
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np 
import pandas as pd
import os
import time
from tqdm import tqdm


def bootstrap_sampling(gt,prob,pred,n_times=2000):
    """Do bootstrap sampling to get confidence interval

    Args:
        gt (np_array):   gt label {0,1}
        prob (np_array): prediction in probability value [0,1]
        pred (np_array): prediction in actual label {0,1}
        n_times (int, optional): Number of resampling times. Defaults to 1000.

    Returns:
        ACC - mean, lower bound (5% quantile), upper bound (95% quantile)
        AUC, Precision, Recall, TPR, TNR, FPR, FNR 
    """
    gt = np.asarray(gt)
    prob = np.asarray(prob)
    pred = np.asarray(pred)


    #Initialise variables    
    sample_size = len(gt)
    index_array = np.arange(sample_size)
    LB_index = int(0.05*n_times)
    UB_index = int(0.95*n_times)
    ACC_list =  np.zeros([n_times])
    AUC_list =  np.zeros([n_times])
    Precision_list =  np.zeros([n_times])
    Recall_list    =  np.zeros([n_times])
    TNR_list       =  np.zeros([n_times])
    FPR_list       =  np.zeros([n_times])
    FNR_list       =  np.zeros([n_times])
    TPR_list       =  np.zeros([n_times])
    
    #Do sampling and calculate metrics
    for i in tqdm(range(n_times)):
        sampled_index = np.random.choice(index_array, sample_size, replace=True)
        gt_sampled = gt[sampled_index]
        prob_sampled = prob[sampled_index]
        pred_sampled = pred[sampled_index]
        ACC_score, AUC_score, Precision_score, Recall_score, TNR, FPR, FNR, TPR = calculate_metrics(gt_sampled,prob_sampled,pred_sampled)
        ACC_list[i]       = ACC_score
        AUC_list[i]       = AUC_score
        Precision_list[i] = Precision_score
        Recall_list[i]    = Recall_score
        TNR_list[i]       = TNR
        FPR_list[i]       = FPR
        FNR_list[i]       = FNR
        TPR_list[i]       = TPR
    #Return result dictionary
    ACC_result        = {'mean': ACC_list.mean(),       'LB': np.sort(ACC_list)[LB_index],       'UB': np.sort(ACC_list)[UB_index]}
    AUC_result        = {'mean': AUC_list.mean(),       'LB': np.sort(AUC_list)[LB_index],       'UB': np.sort(AUC_list)[UB_index]}
    Precision_result  = {'mean': Precision_list.mean(), 'LB': np.sort(Precision_list)[LB_index], 'UB': np.sort(Precision_list)[UB_index]}
    Recall_result     = {'mean': Recall_list.mean(),    'LB': np.sort(Recall_list)[LB_index],    'UB': np.sort(Recall_list)[UB_index]}
    TPR_result        = {'mean': TPR_list.mean(),       'LB': np.sort(TPR_list)[LB_index],       'UB': np.sort(TPR_list)[UB_index]}
    TNR_result        = {'mean': TNR_list.mean(),       'LB': np.sort(TNR_list)[LB_index],       'UB': np.sort(TNR_list)[UB_index]}
    FPR_result        = {'mean': FPR_list.mean(),       'LB': np.sort(FPR_list)[LB_index],       'UB': np.sort(FPR_list)[UB_index]}
    FNR_result        = {'mean': FNR_list.mean(),       'LB': np.sort(FNR_list)[LB_index],       'UB': np.sort(FNR_list)[UB_index]}

    AUC_VAR = np.var(AUC_list*100)
    AUC_STD = np.std(AUC_list*100)

    return ACC_result, AUC_result, Precision_result, Recall_result, TPR_result, TNR_result, FPR_result, FNR_result, AUC_VAR, AUC_STD


def calculate_metrics(gt,prob,pred):
    """Calculate common metrics

    Args:
        gt (np_array):   gt label {0,1}
        prob (np_array): prediction in probability value [0,1]
        pred (np_array): prediction in actual label {0,1}

    Returns:
        ACC, AUC, Precision, Recall, TNR, FPR, FNR, TPR
    """    
    AUC_score = AUROC(gt,prob)
    ACC_score = ACC(gt,pred)
    Precision_score = precision_score(y_true=gt, y_pred = pred)
    Recall_score    = recall_score(y_true=gt, y_pred = pred)
    TNR, FPR, FNR, TPR = confusion_matrix(y_true=gt,y_pred=pred,normalize='true').ravel()
    return ACC_score, AUC_score, Precision_score, Recall_score, TNR, FPR, FNR, TPR


def get_statistics(gt,prob,pred, n_times=2000):
    ACC_score, AUC_score, Precision_score, Recall_score, TPR_score, TNR_score, FPR_score, FNR_score, AUC_VAR, AUC_STD = bootstrap_sampling(gt,prob,pred,n_times=n_times)    
    print("_______TEST RESULTS for {} Bootstrapping_______".format(n_times))
    print("Number:________:{}".format(len(gt)))
    print("Accuracy_______:{:.3f},({:.3f},{:.3f})".format(ACC_score['mean'],ACC_score['LB'],ACC_score['UB']))
    print("AUROC__________:{:.3f},({:.3f},{:.3f}, Var: {:.3f}, Std: {:.3f})".format(AUC_score['mean'],AUC_score['LB'],AUC_score['UB'], AUC_VAR, AUC_STD))
    print("Precison_______:{:.3f},({:.3f},{:.3f})".format(Precision_score['mean'],Precision_score['LB'],Precision_score['UB']))
    print("Recall_________:{:.3f},({:.3f},{:.3f})".format(Recall_score['mean'],Recall_score['LB'],Recall_score['UB']))
    print("TNR____________:{:.3f},({:.3f},{:.3f})".format(TNR_score['mean'],TNR_score['LB'],TNR_score['UB']))
    print("FPR____________:{:.3f},({:.3f},{:.3f})".format(FPR_score['mean'],FPR_score['LB'],FPR_score['UB']))
    print("FNR____________:{:.3f},({:.3f},{:.3f})".format(FNR_score['mean'],FNR_score['LB'],FNR_score['UB']))
    print("TPR____________:{:.3f},({:.3f},{:.3f})".format(TPR_score['mean'],TPR_score['LB'],TPR_score['UB']))
    print("__________________________")