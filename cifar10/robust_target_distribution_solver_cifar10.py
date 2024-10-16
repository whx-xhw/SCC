import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from rtds_train_cifar10 import *
from torchvision.models.resnet import resnet50, resnet18
from thop import profile, clever_format
import sep_dataset_cifar10
from noise_transition_matrix_estimator_cifar10 import *


class Model(nn.Module):
    def __init__(self, feature_dim=128, dataset='cifar10', arch='resnet50'):
        super(Model, self).__init__()

        self.f = []
        if arch == 'resnet18':
            temp_model = resnet18().named_children()
            embedding_size = 512
        elif arch == 'resnet50':
            temp_model = resnet50().named_children()
            embedding_size = 20484

        elif arch == 'resnet34':
            temp_model = resnet50().named_children()
            embedding_size = 512

        else:
            raise NotImplementedError

        for name, module in temp_model:
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10' or dataset == 'cifar100':
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    self.f.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(embedding_size, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1), feature


class classifier1(nn.Module):
    def __init__(self, res_bottleneck):
        super(classifier1, self).__init__()
        self.res_bottleneck = res_bottleneck
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        _, _, feat = self.res_bottleneck(x)
        logit = F.relu(self.linear1(feat))
        logit = self.linear2(logit)
        return logit


class classifier2(nn.Module):
    def __init__(self, res_bottleneck):
        super(classifier2, self).__init__()
        self.res_bottleneck = res_bottleneck
        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        _, _, feat = self.res_bottleneck(x)
        logit = F.relu(self.linear1(feat))
        logit = self.linear2(logit)
        return logit


def get_res(self_supervised_pretrain_path):
    model = Model(feature_dim=1024, dataset='cifar10', arch='resnet18').cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    model.load_state_dict(torch.load(self_supervised_pretrain_path))
    return model


def rtds_cifar10(param, ground_truth_fixed, select_idx, device, epoch, select_pseudo_label, membership, eta):

    dual_model1 = classifier1(res_bottleneck=get_res(self_supervised_pretrain_path='./weight/classifier_pretrain_backbone.pth')).to(device)
    dual_model2 = classifier2(res_bottleneck=get_res(self_supervised_pretrain_path='./weight/classifier_pretrain_backbone.pth')).to(device)

    CE_Loss = nn.CrossEntropyLoss()
    Conf_Penalty = NegEntropy()

    optim1 = torch.optim.SGD(params=dual_model1.parameters(), lr=0.1)
    optim2 = torch.optim.SGD(params=dual_model2.parameters(), lr=0.1)

    train_trans = get_train_trans()
    train_dataset = sep_dataset_cifar10.cifar10_for_warm_up(root='./data', transform=train_trans,
                                                            pseudo_labels=select_pseudo_label, fixed_labels=ground_truth_fixed,
                                                            select_index=select_idx, aug_times=param.aug_times)
    dual_model_train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=16, drop_last=False)

    test_trans = get_test_trans()
    test_dataset = sep_dataset_cifar10.cifar10_for_test(root='./data', transform=test_trans, fixed_labels=ground_truth_fixed)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=16, drop_last=False)

    print('Epoch: {}, warming up for sample selection...'.format(epoch))
    dual_model1, datas4eval1, pseudo_labelss4eval1, fixed_labelss4eval1, index_4eval1 = warm_up(model=dual_model1, optim=optim1, epoch=1+epoch, loss1=CE_Loss, loss2=Conf_Penalty,
                                                                                                dataloader=dual_model_train_loader, device=device, crit=1.0,
                                                                                                bs=512, aug_times=param.aug_times)

    dual_model2, datas4eval2, pseudo_labelss4eval2, fixed_labelss4eval2, index_4eval2 = warm_up(model=dual_model2, optim=optim2, epoch=1+epoch, loss1=CE_Loss, loss2=Conf_Penalty,
                                                                                                dataloader=dual_model_train_loader, device=device, crit=1.0,
                                                                                                bs=512, aug_times=param.aug_times)

    test(model1=dual_model1, model2=dual_model2, loader=test_loader, device=device, epoch=epoch, bs=100)

    dual_model1.train()
    dual_model2.train()

    prob1, pl1, cl1, ids1 = eval(model=dual_model1, device=device, aug_times=param.aug_times, bs=512, data4eval=datas4eval1, pseudo_labels4eval=pseudo_labelss4eval1, fixed_labels4eval=fixed_labelss4eval1, s_idx4eval=index_4eval1)

    select_info1 = ensemble_selector(prob=prob1, aug_times=param.aug_times, bs=512, pseudo_labels=pl1, fixed_labels=cl1, idx_s=ids1,
                                     threshold=0.2, epoch=epoch, number=1, select_idx=select_idx)

    prob2, pl2, cl2, ids2 = eval(model=dual_model2, device=device, aug_times=param.aug_times, bs=512, data4eval=datas4eval2, pseudo_labels4eval=pseudo_labelss4eval2, fixed_labels4eval=fixed_labelss4eval2, s_idx4eval=index_4eval2)

    select_info2 = ensemble_selector(prob=prob2, aug_times=param.aug_times, bs=512, pseudo_labels=pl2, fixed_labels=cl2, idx_s=ids2,
                                     threshold=0.2, epoch=epoch, number=2, select_idx=select_idx)


    accurate_ntm_estimate = False
    # If accurate_ntm_estimate is True, then another temp dual classifiers will be generated and trained with more epochs.
    # These two classifiers will only be used to estimate ntm.
    # By this means, the ntm can be estimated accurately.
    # However, if we do not use this strategy, the misassignment distribution can still be correctly estimated.
    # accurate_ntm_estimate = True will take a larger time-consuming with no contribution to clustering assignment
    # Not recommend.
    if accurate_ntm_estimate:
        temp_dual_model1_ntm = dual_model1().to(device)
        temp_dual_model2_ntm = dual_model2().to(device)

        dual_model1_ntm, dual_model2_ntm = ntm_process(model1=temp_dual_model1_ntm, model2=temp_dual_model2_ntm,
                                                       warm_up_epoch=20, dual_model_train_loader=dual_model_train_loader,
                                                       device=device)
        sigma = sigma_estimator(membership=membership, eta=eta, device=device, model1=dual_model1_ntm,
                                model2=dual_model2_ntm, param=param)

        del temp_dual_model1_ntm
        del temp_dual_model2_ntm
        del dual_model1_ntm
        del dual_model2_ntm
    else:
        sigma = sigma_estimator(membership=membership, eta=eta, device=device, model1=dual_model1,
                                model2=dual_model2, param=param)

    if sigma == 0:
        print('Epoch: {}, estimate as asymmetric noise.'.format(epoch))

    else:
        print('Epoch: {}, estimate as symmetric noise.'.format(epoch))

    sigma = 0

    labeled_loader1, unlabeled_loader1 = labeled_and_unlabeled_loader(select_info=select_info1, param=param)
    labeled_loader2, unlabeled_loader2 = labeled_and_unlabeled_loader(select_info=select_info2, param=param)

    semi_loss = nn.CrossEntropyLoss()

    w_b1 = select_info1[2]
    w_b2 = select_info2[2]

    optim1 = torch.optim.SGD(params=dual_model1.parameters(), lr=0.1, momentum=0.99)
    optim2 = torch.optim.SGD(params=dual_model2.parameters(), lr=0.1, momentum=0.99)

    for c_epo in range(10):
        dual_model2, pre_max2 = mixmatch(model1=dual_model2, model2=dual_model1, w_b=w_b1, labeled_loader=labeled_loader1,
                                         unlabeled_loader=unlabeled_loader1, device=device, param=param, semi_loss=semi_loss,
                                         sigma=sigma, optim=optim2, epoch=epoch, model_number=2, rtds_current_epoch=c_epo)

        dual_model1, pre_max1 = mixmatch(model1=dual_model1, model2=dual_model2, w_b=w_b2, labeled_loader=labeled_loader2,
                                         unlabeled_loader=unlabeled_loader2, device=device, param=param, semi_loss=semi_loss,
                                         sigma=sigma, optim=optim1, epoch=epoch, model_number=2, rtds_current_epoch=c_epo)

        target_distribution = np.argmax(pre_max1 + pre_max2, axis=1)

        accuracy, per_class_correct, nmi, ari = tools_cifar10.clustering_metrix(true_label=ground_truth_fixed, pre_label=target_distribution)
        print('Epoch {}, RTDS {}/{}, acc: {}'.format(epoch, c_epo + 1, 60, accuracy))

        accuracy, per_class_correct, nmi, ari = tools_cifar10.clustering_metrix(true_label=ground_truth_fixed, pre_label=np.argmax(pre_max2, axis=1))
        print('Epoch {}, RTDS {}/{}, acc: {}'.format(epoch, c_epo + 1, 60, accuracy))

        accuracy, per_class_correct, nmi, ari = tools_cifar10.clustering_metrix(true_label=ground_truth_fixed, pre_label=np.argmax(pre_max1, axis=1))
        print('Epoch {}, RTDS {}/{}, acc: {}'.format(epoch, c_epo + 1, 60, accuracy))

    return (pre_max1 + pre_max2) / 2