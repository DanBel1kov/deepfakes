import os

from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image


class ImageDatasetFullyRAM(Dataset):  # loads the WHOLE dataset into RAM
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = [(1, cur) for cur in os.listdir(os.path.join(self.data_dir, 'fake'))]
        self.files.extend([(0, cur) for cur in os.listdir(os.path.join(self.data_dir, 'real'))])
        self.items = []

        for cur in self.files:
            label, file = cur
            folder = 'fake' if label else 'real'
            image = Image.open(os.path.join(self.data_dir, folder, file))

            if transform:
                image = transform(image)

            self.items.append((image, label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.items[idx]
