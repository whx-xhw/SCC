import torch
import argparse
import numpy as np
from clustering_module_mnist import clustering_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import tools_mnist
import os
import membership_selector_mnist
from robust_target_distribution_solver_mnist import rtds_mnist
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--eta_0', type=float, default=0.6)
parser.add_argument('--delta_eta', type=float, default=0.025)
parser.add_argument('--fuzzifier', type=float, default=1.1)
parser.add_argument('--aug_times', type=int, default=20)
param = parser.parse_args()


def main():
    device = torch.device('cuda:{}'.format(param.device))

    init_clustering_center = np.load('./weight/init_clustering_center.npy')
    clustering_module = clustering_model(fuzzifier=param.fuzzifier, class_number=10, device=device).to(device)
    clustering_module.clustering_layer.data = torch.tensor(init_clustering_center.T).to(device)
    clustering_module.load_init_weight(pretrain_path='./weight/init_weight.pth', gpu=param.device)
    print('Model has been initialized.')

    trans = transforms.Compose([transforms.ToTensor()])
    raw_set = datasets.MNIST(root='./data', transform=trans, train=True, download=True)
    raw_loader = DataLoader(dataset=raw_set, shuffle=False, batch_size=10000, drop_last=False)

    raw_data = torch.zeros(size=(len(raw_set), 1, 28, 28))
    raw_label = torch.zeros(size=(len(raw_set), ), dtype=torch.int32)
    init_membership = torch.zeros(size=(len(raw_set), 10))

    for idx, (x, y) in enumerate(raw_loader):
        x = x.to(device)
        y = y.to(device)
        raw_data[10000 * idx: 10000 * (idx + 1)] = x
        raw_label[10000 * idx: 10000 * (idx + 1)] = y
        x_flatten = x.view(-1, 784)
        init_membership[10000 * idx: 10000 * (idx + 1)] = clustering_module(x_flatten)

    init_membership_np = init_membership.detach().cpu().numpy()
    raw_data_np = np.squeeze(raw_data.detach().cpu().numpy())
    raw_data_tensor = torch.tensor(raw_data_np)
    raw_label_np = raw_label.detach().cpu().numpy()

    init_pseudo_label = np.argmax(init_membership_np, axis=1)
    accuracy, per_class_correct, nmi, ari = tools_mnist.clustering_metrix(true_label=raw_label_np, pre_label=init_pseudo_label)
    print('Initialization: acc:{}, per class correct:{}, nmi:{}, ari:{}'.format(accuracy, per_class_correct, nmi, ari))

    ground_truth_fixed = tools_mnist.label_fixer(true_label=raw_label_np, pre_label=init_pseudo_label)
    np.save('./data/ground_truth_fixed.npy', ground_truth_fixed)

    pre_aug_data_path = './data/pre_aug_data_aligned.npy'

    if os.path.exists(pre_aug_data_path):
        pre_aug_data = np.load(pre_aug_data_path)

    else:
        tools_mnist.auto_augmentation(data_path='./data/', aug_times=param.aug_times, ground_truth=raw_label_np, ground_truth_fixed=ground_truth_fixed)
        pre_aug_data = np.load(pre_aug_data_path)

    membership = init_membership_np

    epoch = 0

    while True:

        last_distribution = np.argmax(membership, axis=1)
        eta = param.eta_0 + param.delta_eta * epoch
        print('stopping criterion 1: {}% / {}%'.format(eta * 100, 100))
        if eta > 1:
            print('early stop, stopping criterion 1.')
            break

        select_idx, select_pseudo_label = membership_selector_mnist.membership_selector(membership=membership, eta=int(eta * 6000))
        selected_ground_truth = raw_label_np[select_idx]
        acc, per_class_num = tools_mnist.cluster_acc(y_true=selected_ground_truth, y_pred=select_pseudo_label)
        print('Epoch {}, selected samples, acc:{}'.format(epoch, acc))

        target_distribution = rtds_mnist(param=param, select_idx=select_idx, select_pseudo_label=select_pseudo_label, device=device,
                                         epoch=epoch, aug_data=pre_aug_data, ground_truth_fixed=ground_truth_fixed,
                                         raw_data_tensor=raw_data_tensor, membership=membership, eta=eta)

        target_distribution_pseudo_label = np.argmax(target_distribution, axis=1)
        acc, per_class_num = tools_mnist.cluster_acc(y_true=raw_label_np, y_pred=target_distribution_pseudo_label)
        print('Epoch {}, target distribution, acc:{}'.format(epoch, acc))

        enc = OneHotEncoder(sparse_output=False)
        target_distribution_one_hot = enc.fit_transform(np.reshape(target_distribution_pseudo_label, newshape=(-1, 1)))
        target_distribution_one_hot_tensor = torch.tensor(target_distribution_one_hot)
        clustering_module_optim_set = TensorDataset(raw_data_tensor, target_distribution_one_hot_tensor)
        clustering_module_optim_loader = DataLoader(dataset=clustering_module_optim_set, batch_size=512, shuffle=True, drop_last=False, num_workers=16)
        clustering_module_optim_crit = nn.KLDivLoss(reduction='batchmean')
        clustering_module_optim = torch.optim.SGD(params=clustering_module.parameters(), lr=0.01, momentum=0.99)

        acc_list = np.zeros(shape=(20, ))
        nmi_list = np.zeros(shape=(20, ))
        ari_list = np.zeros(shape=(20, ))

        for mm in range(20):
            for _, (x, y) in enumerate(clustering_module_optim_loader):
                x = x.to(device)
                x = x.view(-1, 784)
                x = x.float()
                y = y.to(device)
                out = clustering_module(x)
                loss = clustering_module_optim_crit(out.log(), y)
                clustering_module_optim.zero_grad()
                loss.backward()
                clustering_module_optim.step()

            raw_data_tensor_cuda = raw_data_tensor.to(device)
            raw_data_tensor_cuda = raw_data_tensor_cuda.view(-1, 784)
            raw_data_tensor_cuda = raw_data_tensor_cuda.float()

            membership = clustering_module(raw_data_tensor_cuda)
            membership = membership.detach().cpu().numpy()
            pseudo_label = np.argmax(membership, axis=1)
            acc, per_class_num, nmi, ari = tools_mnist.clustering_metrix(true_label=raw_label_np, pre_label=pseudo_label)

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

        last_current_distribution_sim, _, _, _ = tools_mnist.clustering_metrix(true_label=current_distribution, pre_label=last_distribution)

        print('stopping criterion 2: {}% / {}%'.format(last_current_distribution_sim * 100, 0.998 * 100))

        if last_current_distribution_sim >= 0.998:
            print('early stop, stopping criterion 2.')
            break

        epoch += 1


if __name__ == '__main__':
    main()