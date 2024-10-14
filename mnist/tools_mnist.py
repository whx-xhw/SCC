import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
import Augmentor
from torch.utils.data import DataLoader
import os
import shutil
import torchvision
import cv2
from torchvision import datasets, transforms


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


def auto_augmentation(data_path, aug_times, ground_truth, ground_truth_fixed):
    pre_set = datasets.MNIST(root='./dataset/mnist/', transform=transforms.ToTensor(), download=True, train=True)
    pre_loader = DataLoader(dataset=pre_set, shuffle=False, batch_size=10000, drop_last=False)

    pre_aug_data = np.zeros(shape=(60000, aug_times, 28, 28))
    pre_aug_label = np.zeros(shape=(60000, aug_times), dtype=np.int32)

    counter = 0
    print('automate augmentation, please wait...')
    for idx, (x, y) in enumerate(pre_loader):
        x_np = np.squeeze(x.detach().cpu().numpy())
        y_np = y.detach().cpu().numpy()

        for i in range(10000):
            y_np_i = y_np[i]
            x_np_i = x_np[i]

            dic_path = './cache/'
            img_path = dic_path + 'img.png'

            if not os.path.exists(dic_path):
                os.mkdir(dic_path)
            else:
                shutil.rmtree(dic_path)
                os.mkdir(dic_path)

            torchvision.utils.save_image(x[i], img_path)
            print('automate augmentation: {}/{}'.format(counter + 1, 60000))

            p = Augmentor.Pipeline(dic_path)
            p.rotate(probability=0.6, max_left_rotation=10, max_right_rotation=10)
            p.random_distortion(probability=0.8, grid_width=3, grid_height=3, magnitude=2)
            p.skew(probability=0.8, magnitude=0.3)
            p.shear(probability=0.8, max_shear_left=4, max_shear_right=4)
            p.sample(aug_times)

            out_path = dic_path + 'output/'
            img_name = os.listdir(out_path)

            for j in range(len(img_name)):
                aug_img_path = out_path + img_name[j]
                img = cv2.imread(aug_img_path, 0)
                img_np = np.array(img)
                pre_aug_data[counter, j, :, :] = img_np / 255.
                pre_aug_label[counter, j] = y_np_i

            counter += 1

    pre_aug_label_ = pre_aug_label + 1234

    pairs = []
    for i in range(10):
        true_i_idx = np.where(ground_truth == i)[0]
        fixed_i_label = ground_truth_fixed[true_i_idx[0]]
        i_pair = [i, fixed_i_label]
        pairs.append(i_pair)

    for i in range(10):
        fake_label = i + 1234
        fake_i_idx = np.where(pre_aug_label_[:, 0] == fake_label)[0]
        pre_aug_label_[fake_i_idx] = pairs[i][1]

    pre_aug_data_aligned = np.zeros(shape=(60000 * aug_times, 28, 28), dtype=np.float32)
    pre_aug_label_aligned = np.zeros(shape=(60000 * aug_times,), dtype=np.int64)

    counter = 0
    for i in range(60000):
        for j in range(aug_times):
            pre_aug_data_aligned[counter] = pre_aug_data[i, j]
            pre_aug_label_aligned[counter] = pre_aug_label_[i, j]
            counter += 1

    np.save(data_path + 'pre_aug_data_aligned.npy', pre_aug_data_aligned)
    np.save(data_path + 'pre_aug_label_aligned.npy', pre_aug_label_aligned)