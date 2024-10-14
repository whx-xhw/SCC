import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from rtds_train_mnist import *
from noise_transition_matrix_estimator_mnist import *


class classifier1(nn.Module):
    def __init__(self):
        super(classifier1, self).__init__()
        self.Linear1 = nn.Linear(784, 1000)
        self.Linear2 = nn.Linear(1000, 500)
        self.Linear3 = nn.Linear(500, 300)
        self.Linear4 = nn.Linear(300, 10)

    def forward(self, x):
        h = F.relu(self.Linear1(x))
        h = F.relu(self.Linear2(h))
        h = F.relu(self.Linear3(h))
        h = self.Linear4(h)
        return h


class classifier2(nn.Module):
    def __init__(self):
        super(classifier2, self).__init__()
        self.Linear1 = nn.Linear(784, 300)
        self.Linear2 = nn.Linear(300, 500)
        self.Linear3 = nn.Linear(500, 1000)
        self.Linear4 = nn.Linear(1000, 10)

    def forward(self, x):
        h = F.relu(self.Linear1(x))
        h = F.relu(self.Linear2(h))
        h = F.relu(self.Linear3(h))
        h = self.Linear4(h)
        return h


def rtds_mnist(param, select_idx, select_pseudo_label, device, epoch, aug_data, ground_truth_fixed, raw_data_tensor, membership, eta):
    select_ground_truth_fixed = ground_truth_fixed[select_idx]

    dual_model1 = classifier1().to(device)
    dual_model2 = classifier2().to(device)

    aligned_select_pseudo_labels = np.zeros(shape=(select_pseudo_label.shape[0] * param.aug_times, ), dtype=np.int32)
    aligned_select_idx = np.zeros(shape=(select_idx.shape[0] * param.aug_times), dtype=np.int32)
    for i in range(select_idx.shape[0]):
        aligned_index_start = select_idx[i] * param.aug_times
        aligned_select_idx[i * param.aug_times: (i + 1) * param.aug_times] = np.arange(aligned_index_start, aligned_index_start + param.aug_times)
        aligned_select_pseudo_labels[i * param.aug_times: (i + 1) * param.aug_times] = select_pseudo_label[i]

    select_augmented_data = aug_data[aligned_select_idx]
    select_augmented_data_tensor = torch.tensor(select_augmented_data)
    aligned_select_pseudo_labels_tensor = torch.tensor(aligned_select_pseudo_labels)

    dual_model_train_set = TensorDataset(select_augmented_data_tensor, aligned_select_pseudo_labels_tensor)
    dual_model_train_loader = DataLoader(dataset=dual_model_train_set, batch_size=4096, shuffle=True, drop_last=False, num_workers=16)
    dual_model_eval_loader = DataLoader(dataset=dual_model_train_set, batch_size=10000, shuffle=False, drop_last=False, num_workers=16)

    CE_Loss = nn.CrossEntropyLoss()
    CE = nn.CrossEntropyLoss(reduction='none')
    Conf_Penalty = NegEntropy()
    semi_loss = SemiL()

    optim1 = torch.optim.SGD(params=dual_model1.parameters(), lr=0.01, momentum=0.9)
    optim2 = torch.optim.SGD(params=dual_model2.parameters(), lr=0.01, momentum=0.9)

    print('Epoch: {}, warming up for sample selection...'.format(epoch))
    dual_model1 = warm_up(model=dual_model1, optim=optim1, epoch=5, loss1=CE_Loss, loss2=Conf_Penalty,
                          dataloader=dual_model_train_loader, device=device, crit=1.0)

    dual_model2 = warm_up(model=dual_model2, optim=optim2, epoch=5, loss1=CE_Loss, loss2=Conf_Penalty,
                          dataloader=dual_model_train_loader, device=device, crit=1.0)

    test(model1=dual_model1, model2=dual_model2, raw_data=raw_data_tensor, ground_truth=ground_truth_fixed,
         device=device, epoch=epoch)

    dual_model1.train()
    dual_model2.train()

    prob1, loss1 = eval(model=dual_model1, data_loader=dual_model_eval_loader, device=device, loss=CE,
                        eval_number=select_augmented_data.shape[0])

    labeled_idx1, unlabeled_idx1, w_b1 = ensemble_selector(prob=prob1, aug_times=param.aug_times, upsilon=0.3,
                                                           pseudo_label=select_pseudo_label,
                                                           ground_truth_fixed=select_ground_truth_fixed,
                                                           classifier_num=1, epoch=epoch)

    absolute_labeled_index1 = select_idx[labeled_idx1]
    absolute_unlabeled_index1 = select_idx[unlabeled_idx1]
    labeled_pseudo_label1 = select_pseudo_label[labeled_idx1]
    unlabeled_pseudo_label1 = select_pseudo_label[unlabeled_idx1]

    labeled_information1 = [absolute_labeled_index1, labeled_pseudo_label1]
    unlabeled_information1 = [absolute_unlabeled_index1, unlabeled_pseudo_label1]

    prob2, loss2 = eval(model=dual_model2, data_loader=dual_model_eval_loader, device=device, loss=CE, eval_number=select_augmented_data.shape[0])

    labeled_idx2, unlabeled_idx2, w_b2 = ensemble_selector(prob=prob2, aug_times=param.aug_times, upsilon=0.3,
                                                           pseudo_label=select_pseudo_label,
                                                           ground_truth_fixed=select_ground_truth_fixed,
                                                           classifier_num=2, epoch=epoch)

    absolute_labeled_index2 = select_idx[labeled_idx2]
    absolute_unlabeled_index2 = select_idx[unlabeled_idx2]
    labeled_pseudo_label2 = select_pseudo_label[labeled_idx2]
    unlabeled_pseudo_label2 = select_pseudo_label[unlabeled_idx2]

    labeled_information2 = [absolute_labeled_index2, labeled_pseudo_label2]
    unlabeled_information2 = [absolute_unlabeled_index2, unlabeled_pseudo_label2]

    accurate_ntm_estimate = False
    # If accurate_ntm_estimate is True, then another temp dual classifiers will be generated and trained with more epochs.
    # These two classifiers will only be used to estimate ntm.
    # By this means, the ntm can be estimated accurately.
    # However, if we do not use this strategy, the misassignment distribution can still be correctly estimated.
    # accurate_ntm_estimate = True will take a larger time-consuming with no contribution to clustering assignment
    # Not recommend.

    if accurate_ntm_estimate:
        temp_dual_model1_ntm = classifier1().to(device)
        temp_dual_model2_ntm = classifier2().to(device)

        dual_model1_ntm, dual_model2_ntm = ntm_process(model1=temp_dual_model1_ntm, model2=temp_dual_model2_ntm,
                                                       warm_up_epoch=20, dual_model_train_loader=dual_model_train_loader,
                                                       device=device)
        sigma = sigma_estimator_hand(membership=membership, eta=eta, aug_times=param.aug_times, aug_data=aug_data,
                                     device=device, model1=dual_model1_ntm, model2=dual_model2_ntm)

        del temp_dual_model1_ntm
        del temp_dual_model2_ntm
        del dual_model1_ntm
        del dual_model2_ntm
    else:
        sigma = sigma_estimator_hand(membership=membership, eta=eta, aug_times=param.aug_times, aug_data=aug_data,
                                     device=device, model1=dual_model1, model2=dual_model2)

    if sigma == 0:
        print('Epoch: {}, estimate as asymmetric noise.'.format(epoch))

    else:
        print('Epoch: {}, estimate as symmetric noise.'.format(epoch))

    optim1 = torch.optim.SGD(params=dual_model1.parameters(), lr=0.01, momentum=0.9)
    optim2 = torch.optim.SGD(params=dual_model2.parameters(), lr=0.01, momentum=0.9)

    labeled_loader1, unlabeled_loader1 = labeled_and_unlabeled_loader(absolute_labeled_information=labeled_information1,
                                                                      absolute_unlabeled_information=unlabeled_information1)

    labeled_loader2, unlabeled_loader2 = labeled_and_unlabeled_loader(absolute_labeled_information=labeled_information2,
                                                                      absolute_unlabeled_information=unlabeled_information2)

    for c_epo in range(60):
        dual_model2 = mixmatch(model1=dual_model2, model2=dual_model1, optim=optim2, labeled_loader=labeled_loader1,
                               unlabeled_loader=unlabeled_loader1, batch_size=5000, device=device, temp=0.5,
                               semi_loss=semi_loss, w_b=w_b1, aug_data=aug_data, tr_sel_idx=absolute_labeled_index1,
                               tr_unsel_idx=absolute_unlabeled_index1, ground_fixed=ground_truth_fixed, aug_times=param.aug_times,
                               sigma=sigma, model_number=2, rtds_current_epoch=c_epo, epoch=epoch, data_path='./data/')

        dual_model1 = mixmatch(model1=dual_model1, model2=dual_model2, optim=optim1, labeled_loader=labeled_loader2,
                               unlabeled_loader=unlabeled_loader2, batch_size=5000, device=device, temp=0.5,
                               semi_loss=semi_loss, w_b=w_b2, aug_data=aug_data, tr_sel_idx=absolute_labeled_index2,
                               tr_unsel_idx=absolute_unlabeled_index2, ground_fixed=ground_truth_fixed, aug_times=param.aug_times,
                               sigma=sigma, model_number=1, rtds_current_epoch=c_epo, epoch=epoch, data_path='./data/')

        raw_data_tensor_flatten = raw_data_tensor.to(device)
        raw_data_tensor_flatten = raw_data_tensor_flatten.view(-1, 784)
        raw_data_tensor_flatten = raw_data_tensor_flatten.float()
        out1 = dual_model1(raw_data_tensor_flatten)
        out2 = dual_model2(raw_data_tensor_flatten)
        out_soft1 = torch.softmax(out1, dim=1)
        out_soft2 = torch.softmax(out2, dim=1)
        pre_membership = (out_soft1 + out_soft2)
        target_distribution = pre_membership.detach().cpu().numpy()
        accuracy, per_class_correct, nmi, ari = tools_mnist.clustering_metrix(true_label=ground_truth_fixed,
                                                                              pre_label=np.argmax(target_distribution, axis=1))
        print('Epoch {}, RTDS {}/{}, acc: {}'.format(epoch, c_epo + 1, 60, accuracy))

    return target_distribution