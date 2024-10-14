import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI


def clustering_metrix(true_label, pre_label):
    accuracy, per_class_correct_num = cluster_acc(y_true=true_label, y_pred=pre_label)
    nmi = NMI(labels_true=true_label, labels_pred=pre_label)
    ari = ARI(labels_true=true_label, labels_pred=pre_label)
    return accuracy, per_class_correct_num, nmi, ari


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = linear_sum_assignment(w.max() - w)
    calcul = 0
    CorrectNum = []
    for i in range(10):
        calcul += w[ind[0][i], ind[1][i]]
        CorrectNum.append(w[ind[0][i], ind[1][i]])
    calcul = calcul * 1.
    Acc = calcul / y_pred.size
    return Acc, CorrectNum


def label_fixer(true_label, pre_label):
    true_label = true_label.astype(np.int64)
    D = int(max(true_label.max(), true_label.max()) + 1)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pre_label.size):
        w[int(pre_label[i]), int(true_label[i])] += 1
    ind = linear_sum_assignment(w.max() - w)
    pre_ind = ind[0]
    tru_ind = ind[1]
    tru_label_fix = true_label + 1000
    for i in range(pre_ind.shape[0]):
        tt = tru_ind[i] + 1000
        idx = np.where(tru_label_fix == tt)[0]
        tru_label_fix[idx] = pre_ind[i]
    return tru_label_fix


def select_samples_detailed_info(class_num, select_idx, ground_truth_fixed, pseudo_labels):
    selected_ground_truth_fixed = ground_truth_fixed[select_idx]
    selected_pseudo_labels = pseudo_labels
    print('---------------- Selection Info -----------------')
    for i in range(class_num):
        idx_i = np.where(selected_pseudo_labels == i)[0]
        selected_pseudo_labels_i = selected_pseudo_labels[idx_i]
        selected_ground_truth_fixed_i = selected_ground_truth_fixed[idx_i]
        acc = np.mean(selected_ground_truth_fixed_i == selected_pseudo_labels_i)
        print('>>> class: {}, num:{}, acc: {}'.format(i, idx_i.shape[0], acc))
    print('--------------------------------------------------')

