import torch
import numpy as np
from robust_target_distribution_solver_cifar10 import rtds_cifar10
from torch.utils.data import DataLoader
from clustering_module_cifar10 import get_resnet_cifar, encoder, clustering_model
import argparse
from tqdm import tqdm
from torchvision import datasets, transforms
import tools_cifar10
import membership_selector_cifar10
import sep_dataset_cifar10
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--eta_0', type=float, default=0.6)
parser.add_argument('--delta_eta', type=float, default=0.025)
parser.add_argument('--fuzzifier', type=float, default=1.04)
parser.add_argument('--aug_times', type=int, default=5)
param = parser.parse_args()


def main():
    device = torch.device('cuda:{}'.format(param.device))

    res = get_resnet_cifar(name='ResNet34')
    res.load_state_dict(torch.load('./weight/init_weight.pth'))
    enc = encoder(resnet=res)
    clustering_module = clustering_model(feat_extractor=enc, fuzzifier=param.fuzzifier, class_number=10, device=device).to(device)
    init_clustering_center = np.load('./weight/init_clustering_center.npy')
    clustering_module.clustering_layer.data = torch.tensor(init_clustering_center).to(device)
    print('Model has been initialized.')

    trans = transforms.Compose([transforms.Resize(size=(40, 40)),
                                transforms.CenterCrop(size=(32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    raw_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=trans)
    raw_loader = DataLoader(dataset=raw_set, batch_size=500, shuffle=False, num_workers=16)

    init_membership = np.zeros(shape=(len(raw_set), 10))
    raw_label = np.zeros(shape=(len(raw_set), ), dtype=np.int32)

    for idx, (x, y) in enumerate(tqdm(raw_loader)):
        x = x.to(device)
        y = y.to(device)

        y_np = y.detach().cpu().numpy()
        raw_label[500 * idx: 500 * (idx + 1)] = y_np

        pred = clustering_module(x)

        pred_np = pred.detach().cpu().numpy()
        init_membership[500 * idx: 500 * (idx + 1)] = pred_np

    init_pseudo_label = np.argmax(init_membership, axis=1)
    accuracy, per_class_correct, nmi, ari = tools_cifar10.clustering_metrix(true_label=raw_label, pre_label=init_pseudo_label)
    print('Initialization: acc:{}, per class correct:{}, nmi:{}, ari:{}'.format(accuracy, per_class_correct, nmi, ari))

    ground_truth_fixed = tools_cifar10.label_fixer(true_label=raw_label, pre_label=init_pseudo_label)
    np.save('./data/ground_truth_fixed.npy', ground_truth_fixed)

    membership = init_membership

    epoch = 0

    while True:

        last_distribution = np.argmax(membership, axis=1)
        eta = param.eta_0 + param.delta_eta * epoch
        print('stopping criterion 1: {}% / {}%'.format(eta * 100, 100))

        if eta >= 1:
            print('early stop, stopping criterion 1.')
            break

        select_idx, select_pseudo_label = membership_selector_cifar10.membership_selector(membership=membership, eta=int(eta * 5000))
        selected_ground_truth = raw_label[select_idx]
        acc, per_class_num = tools_cifar10.cluster_acc(y_true=selected_ground_truth, y_pred=select_pseudo_label)
        print('Epoch {}, selected samples, acc:{}, per class correct:{}'.format(epoch, acc, per_class_num))

        tools_cifar10.select_samples_detailed_info(class_num=10, select_idx=select_idx, ground_truth_fixed=ground_truth_fixed, pseudo_labels=select_pseudo_label)

        target_distribution = rtds_cifar10(param=param, ground_truth_fixed=ground_truth_fixed, select_idx=select_idx, device=device, epoch=epoch,
                                           select_pseudo_label=select_pseudo_label, membership=membership,eta=eta)

        target_distribution_pseudo_label = np.argmax(target_distribution, axis=1)
        acc, per_class_num = tools_cifar10.cluster_acc(y_true=raw_label, y_pred=target_distribution_pseudo_label)
        print('Epoch {}, target distribution, acc:{}, per class correct:{}'.format(epoch, acc, per_class_num))

        clustering_module_optim_set = sep_dataset_cifar10.cifar10_for_finetune(root='./data', transform=trans, pseudo_labels=target_distribution_pseudo_label)
        clustering_module_optim_loader = DataLoader(dataset=clustering_module_optim_set, batch_size=512, shuffle=True, drop_last=False, num_workers=16)
        clustering_module_optim_crit = nn.KLDivLoss(reduction='batchmean')
        clustering_module_optim = torch.optim.SGD(params=clustering_module.parameters(), lr=0.001)

        acc_list = np.zeros(shape=(20, ))
        nmi_list = np.zeros(shape=(20, ))
        ari_list = np.zeros(shape=(20, ))
        for mm in range(20):
            for _, (x, y) in enumerate(clustering_module_optim_loader):
                x = x.to(device)
                x = x.float()
                y = y.to(device)
                y = F.one_hot(y, num_classes=10)
                y = y.float()
                out = clustering_module(x)
                loss = clustering_module_optim_crit(out.log(), y)
                clustering_module_optim.zero_grad()
                loss.backward()
                clustering_module_optim.step()

            membership = np.zeros(shape=(50000, 10))

            for idx, (x, y) in enumerate(raw_loader):
                x = x.to(device)
                y = y.to(device)

                y_np = y.detach().cpu().numpy()
                raw_label[500 * idx: 500 * (idx + 1)] = y_np

                pred = clustering_module(x)

                pred_np = pred.detach().cpu().numpy()
                membership[500 * idx: 500 * (idx + 1)] = pred_np

            pseudo_label = np.argmax(membership, axis=1)

            acc, per_class_num, nmi, ari = tools_cifar10.clustering_metrix(true_label=raw_label, pre_label=pseudo_label)
            acc_list[mm] = acc
            nmi_list[mm] = nmi
            ari_list[mm] = ari

        best_acc_idx = np.argmax(acc_list, axis=0)
        acc_b = acc_list[best_acc_idx]
        nmi_b = nmi_list[best_acc_idx]
        ari_b = ari_list[best_acc_idx]
        print('Epoch {}, best clustering assignments, acc:{}, nmi:{}, ari:{}'.format(epoch, acc_b, nmi_b, ari_b))
        acc_l = acc_list[-1]
        nmi_l = nmi_list[-1]
        ari_l = ari_list[-1]
        print('Epoch {}, last clustering assignments, acc:{}, nmi:{}, ari:{}'.format(epoch, acc_l, nmi_l, ari_l))

        current_distribution = pseudo_label

        last_current_distribution_sim, _, _, _ = tools_cifar10.clustering_metrix(true_label=current_distribution, pre_label=last_distribution)
        print('stopping criterion 2: {}% / {}%'.format(last_current_distribution_sim * 100, 0.998 * 100))

        if last_current_distribution_sim >= 0.998:
            print('early stop, stopping criterion 2.')
            break

        epoch += 1


if __name__ == '__main__':
    main()


