import os
import numpy as np
np.random.seed(0)
from torch.utils.data import Dataset
from PIL import Image

class ABDataset(Dataset):
    def __init__(self, root_dir, partition, transform=None, AtoB=True):
        assert partition in ['test', 'train', 'val']

        dir_A = os.path.join(root_dir, partition + 'A')
        dir_B = os.path.join(root_dir, partition + 'B')

        dir_A, dir_B = (dir_A, dir_B) if AtoB else (dir_B, dir_A)

        self.A = os.listdir(dir_A)
        self.B = os.listdir(dir_B)

        self.dir_A = dir_A
        self.dir_B = dir_B


        self.transform = transform

    def __len__(self):
        return max(len(self.A), len(self.B))

    def __getitem__(self, idx):
        idx_A = np.random.randint(len(self.A))
        idx_B = np.random.randint(len(self.B))

        image_A = Image.open(os.path.join(self.dir_A, self.A[idx_A]))
        image_B = Image.open(os.path.join(self.dir_B, self.B[idx_B]))
        if self.transform is not None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {'A': image_A, 'B': image_B}
