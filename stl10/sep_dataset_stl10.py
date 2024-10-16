import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class stl10_for_finetune(VisionDataset):
    def __init__(self, root, transform, pseudo_labels, base_folder='stl10_binary/'):
        super().__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        self.base_folder = base_folder
        self.train_data = self.get_data(data_file='train_X.bin')
        self.test_data = self.get_data(data_file='test_X.bin')
        self.data = np.concatenate([self.train_data, self.test_data], axis=0)
        self.pseudo_labels = pseudo_labels

    def get_data(self, data_file):
        path = self.root + self.base_folder + data_file
        with open(path, "rb") as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        return images

    def __getitem__(self, index):
        img, pt = self.data[index], int(self.pseudo_labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        return self.transform(img), pt

    def __len__(self):
        return self.data.shape[0]


class stl10_for_test(VisionDataset):
    def __init__(self, root, transform, fixed_labels, base_folder='stl10_binary/'):
        super().__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        self.base_folder = base_folder
        self.train_data = self.get_data(data_file='train_X.bin')
        self.test_data = self.get_data(data_file='test_X.bin')
        self.data = np.concatenate([self.train_data, self.test_data], axis=0)
        self.fixed_labels = fixed_labels

    def get_data(self, data_file):
        path = self.root + self.base_folder + data_file
        with open(path, "rb") as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        return images

    def __getitem__(self, index):
        img, ft = self.data[index], int(self.fixed_labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        return self.transform(img), ft

    def __len__(self):
        return self.data.shape[0]


class stl10_for_rtds(VisionDataset):
    def __init__(self, root, transform, pseudo_labels, fixed_labels, select_index, aug_times, base_folder='stl10_binary/'):
        super().__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        self.base_folder = base_folder
        self.train_data = self.get_data(data_file='train_X.bin')
        self.test_data = self.get_data(data_file='test_X.bin')
        self.data = np.concatenate([self.train_data, self.test_data], axis=0)
        self.pseudo_labels = pseudo_labels
        self.aug_times = aug_times
        self.data = self.data[select_index]
        self.select_idx = select_index
        self.fixed_labels = fixed_labels
        self.fixed_labels = self.fixed_labels[select_index]
        print(np.mean(self.fixed_labels == self.pseudo_labels))

    def get_data(self, data_file):
        path = self.root + self.base_folder + data_file
        with open(path, "rb") as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        return images

    def __getitem__(self, index):
        img, pt, ft, s_idx = self.data[index], int(self.pseudo_labels[index]), int(self.fixed_labels[index]), int(self.select_idx[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
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
        return self.data.shape[0]


class stl10_for_ntm(VisionDataset):
    def __init__(self, root, transform, pseudo_labels, select_index, aug_times, base_folder='stl10_binary/'):
        super().__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        self.base_folder = base_folder
        self.train_data = self.get_data(data_file='train_X.bin')
        self.test_data = self.get_data(data_file='test_X.bin')
        self.data = np.concatenate([self.train_data, self.test_data], axis=0)
        self.pseudo_labels = pseudo_labels
        self.aug_times = aug_times
        self.data = self.data[select_index]
        self.select_idx = select_index

    def get_data(self, data_file):
        path = self.root + self.base_folder + data_file
        with open(path, "rb") as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        return images

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


class stl10_for_mixup(VisionDataset):
    def __init__(self, root, transform, pseudo_labels, select_index, aug_times, base_folder='stl10_binary/'):
        super().__init__(root, transform=transform)
        self.root = root
        self.transform = transform
        self.base_folder = base_folder
        self.train_data = self.get_data(data_file='train_X.bin')
        self.test_data = self.get_data(data_file='test_X.bin')
        self.data = np.concatenate([self.train_data, self.test_data], axis=0)
        self.pseudo_labels = pseudo_labels
        self.aug_times = aug_times
        self.data = self.data[select_index]
        self.select_idx = select_index

    def get_data(self, data_file):
        path = self.root + self.base_folder + data_file
        with open(path, "rb") as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        return images

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