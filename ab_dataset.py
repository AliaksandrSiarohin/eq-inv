import os
import numpy as np
np.random.seed(0)
from torch.utils.data import Dataset

class ABDataset(Dataset):
    def __init__(self, root_dir, partition, transform=None, AtoB=True):
        assert partition in ['test', 'train', 'val']

        dir_A = os.listdir(os.path.join(root_dir, partition + '_A'))
        dir_B = os.listdir(os.path.join(root_dir, partition + '_B'))

        dir_A, dir_B = dir_A, dir_B if AtoB else dir_B, dir_A

        self.A = dir_A
        self.B = dir_B

        self.transform = transform

    def __len__(self):
        return max(len(dir_A), len(dir_B))

    def __getitem__(self, idx):
        idx_A = np.random.rand(len(dir_A))
        idx_B = np.random.rand(len(dir_B))

        image_A = self.A[idx_A]
        image_B = self.B[idx_B]
        if self.transform is not None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {'A': image_A, 'B': image_B}
