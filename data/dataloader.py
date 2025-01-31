from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import os


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        self.count_back = 0
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        cur_fname = str(index).zfill(5) + '.png'
        cur_fname = os.path.join(self.root, cur_fname)
        index = index + self.count_back
        if cur_fname not in self.fpaths:
            print('Not found! Filename:', cur_fname)
            self.count_back -= 1
            return None
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

@register_dataset(name='ffhq_input')
class FFHQInputDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        self.count_back = 0
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        cur_fname = str(index).zfill(5) + '.png'
        cur_fname = os.path.join(self.root, cur_fname)
        index = index + self.count_back
        if cur_fname not in self.fpaths:
            print('Not found! Filename:', cur_fname)
            self.count_back -= 1
            return None
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

@register_dataset(name='ffhq_dps')
class FFHQDPSDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        self.count_back = 0
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        cur_fname = str(index).zfill(5) + '.png'
        cur_fname = os.path.join(self.root, cur_fname)
        index = index + self.count_back
        if cur_fname not in self.fpaths:
            print('Not found! Filename:', cur_fname)
            self.count_back -= 1
            return None
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

@register_dataset(name='ffhq_annealed')
class FFHQAnnealedDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.root = root

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        self.count_back = 0

        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        cur_fname = str(index).zfill(5) + '.png'
        cur_fname = os.path.join(self.root, cur_fname)
        index = index + self.count_back
        if cur_fname not in self.fpaths:
            print('Not found! Filename:', cur_fname)
            self.count_back -= 1
            return None
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
