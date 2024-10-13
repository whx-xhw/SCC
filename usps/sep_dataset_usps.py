import numpy as np
from PIL import Image
import torch
from torchvision.datasets.vision import VisionDataset
import bz2


class USPS(VisionDataset):

    def __init__(self, root, transform, select_idx, pseudo_labels):
        super().__init__(root, transform=transform)

        full_path = root + '/' + 'usps.bz2'
        with bz2.open(full_path) as fp:
            raw_data = [line.decode().split() for line in fp.readlines()]
            tmp_list = [[x.split(":")[-1] for x in data[1:]] for data in raw_data]
            imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
            imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)

        self.data = imgs[select_idx]
        self.pseudo = torch.tensor(pseudo_labels, dtype=torch.int64)

    def __getitem__(self, index):
        img, pseudo = self.data[index], int(self.pseudo[index])
        img = Image.fromarray(img, mode="L")
        return self.transform(img), pseudo, index

    def __len__(self) -> int:
        return len(self.data)