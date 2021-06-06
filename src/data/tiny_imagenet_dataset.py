import os
import glob
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms

from PIL import Image
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class Dataset(datasets.VisionDataset):
    
    def __init__(
                self,
            mode: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            img_size : int = None,
        ) -> None:
        self.mode = mode
        if 'train' in self.mode or 'TRAIN' in self.mode:
            self.mode = 'TRAIN'
            self.dataset = sorted(glob.glob('../../data/processed/tinyimagenet/train/*/images/*.JPEG'))
            self.target = sorted(glob.glob('../../data/processed/tinyimagenet/train/*/images/*.JPEG'))
            self.target = [f.split('/')[-3] for f in self.target]
        elif 'val' in self.mode or 'VAL' in self.mode:
            self.mode = 'VALID'
            self.dataset = sorted(glob.glob('../../data/processed/tinyimagenet/val/images/*.JPEG'))
            self.target = sorted(glob.glob('../../data/processed/tinyimagenet/val/images/*'))
            self.target = [int(f.split('/')[-1].replace('val_', '').replace('.JPEG', '')) for f in self.target]
        elif 'test' in self.mode or 'TEST' in self.mode:
            self.mode = 'TEST'
            self.dataset = sorted(glob.glob('../../data/processed/tinyimagenet/test/images/*.JPEG'))
            self.target = sorted(glob.glob('../../data/processed/tinyimagenet/test/images/*'))
            self.target = [int(f.split('/')[-1].replace('test_', '').replace('.JPEG', '')) for f in self.target]
        else:
            raise Exception('Mode not defined properly.')
        self.transform = transform
        self.base_transform = transforms.Compose([
                                        transforms.Resize(img_size),
                                        transforms.ToTensor(),])
    
    def __getitem__(self, ind):
        image = Image.open(self.dataset[ind])
        if self.transform is not None:
            transform_image = self.transform(image)
        if self.target_transform is not None:
            image = self.target_transform(image)
        if image.shape[0] == 3 and transform_image.shape[0] == 3:
            return [image, transform_image], self.target[ind]
        else:
            return [image.repeat(3, 1, 1), transform_image.repeat(3, 1, 1)], self.target[ind]


    def __len__(self):
        return len(self.dataset)
