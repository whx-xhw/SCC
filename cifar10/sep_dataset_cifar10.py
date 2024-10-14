import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


def unpickle(file):
    # Tool for reading cifar data
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar10_for_warm_up(Dataset):
    def __init__(self, root, transform, pseudo_labels, fixed_labels, select_index, aug_times):
        self.transform = transform
        self.root = root
        train_data = []
        clean_label = []
        for n in range(1, 6):
            dpath = self.root + '/cifar-10-batches-py/data_batch_{}'.format(n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
            clean_label.append(data_dic['labels'])
        train_data = np.concatenate(train_data)
        clean_label = np.concatenate(clean_label)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))

        self.select_index = select_index
        self.aug_times = aug_times
        self.select_pseudo_labels = pseudo_labels

        self.train_data = train_data[self.select_index]
        self.fixed_labels = fixed_labels[self.select_index]

    def __getitem__(self, index):
        img, pt, ft, s_idx = self.train_data[index], self.select_pseudo_labels[index], self.fixed_labels[index], self.select_index[index]
        img = Image.fromarray(img)
        img_list = []
        pt_list = []
        ft_list = []
        s_idx_list = []
        for i in range(self.aug_times):
            img_ = self.transform(img)
            img_list.append(img_)
            pt_list.append(pt)
            ft_list.append(ft)
            s_idx_list.append(s_idx)
        return img_list, pt_list, ft_list, s_idx_list

    def __len__(self):
        return len(self.train_data)


class cifar10_for_test(Dataset):
    def __init__(self, root, transform, fixed_labels):
        self.transform = transform
        self.root = root
        train_data = []
        for n in range(1, 6):
            dpath = self.root + '/cifar-10-batches-py/data_batch_{}'.format(n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))

        self.data = train_data
        self.label = fixed_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = Image.fromarray(img)
        return self.transform(img), target

    def __len__(self):
        return len(self.data)


class cifar10_for_mixup(Dataset):
    def __init__(self, root, transform, pseudo_labels, select_index, aug_times):
        self.transform = transform
        self.root = root
        train_data = []
        clean_label = []
        for n in range(1, 6):
            dpath = self.root + '/cifar-10-batches-py/data_batch_{}'.format(n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
            clean_label.append(data_dic['labels'])
        train_data = np.concatenate(train_data)
        clean_label = np.concatenate(clean_label)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))

        self.select_index = select_index
        self.aug_times = aug_times
        self.select_pseudo_labels = pseudo_labels

        self.train_data = train_data[self.select_index]

    def __getitem__(self, index):
        img, pt = self.train_data[index], self.select_pseudo_labels[index]
        img = Image.fromarray(img)
        img_list = []
        pt_list = []
        for i in range(self.aug_times):
            img_ = self.transform(img)
            img_list.append(img_)
            pt_list.append(pt)
        return img_list, pt_list

    def __len__(self):
        return len(self.train_data)


class cifar10_for_finetune(Dataset):
    def __init__(self, root, transform, pseudo_labels):
        self.transform = transform
        self.root = root
        train_data = []
        for n in range(1, 6):
            dpath = self.root + '/cifar-10-batches-py/data_batch_{}'.format(n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))

        self.data = train_data
        self.label = pseudo_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = Image.fromarray(img)
        return self.transform(img), target

    def __len__(self):
        return len(self.data)


class cifar10_for_ntm(Dataset):
    def __init__(self, root, transform, pseudo_labels, select_index, aug_times):
        super().__init__(root, transform=transform)
        self.root = root
        self.transform = transform

        train_data = []
        for n in range(1, 6):
            dpath = self.root + '/cifar-10-batches-py/data_batch_{}'.format(n)
            data_dic = unpickle(dpath)
            train_data.append(data_dic['data'])
        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))

        self.data = train_data
        self.pseudo_labels = pseudo_labels
        self.aug_times = aug_times
        self.data = self.data[select_index]
        self.select_idx = select_index

    def __getitem__(self, index):
        img, pt = self.data[index], int(self.pseudo_labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img_list = []
        pt_list = []
        for i in range(self.aug_times):
            img_ = self.transform(img)
            img_list.append(img_)
            pt_list.append(pt)
        return img_list, pt_list

    def __len__(self):
        return self.data.shape[0]