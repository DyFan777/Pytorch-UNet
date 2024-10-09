import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import scipy.io

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == '.mat':
        try:
            # Try to open with h5py for MATLAB version 7.3+
            mat_file = h5py.File(filename, 'r')
            keys = list(mat_file.keys())
            if not keys:
                raise ValueError("No datasets found in the .mat file.")
            dataset_name = keys[0]
            imgfile = Image.fromarray(mat_file[dataset_name][:])  
            return imgfile
        except OSError:
            # If it's not an HDF5 file, fall back to scipy for older versions
            print(f"{filename} is not an HDF5 file, trying scipy.io.loadmat()")
            mat_file = scipy.io.loadmat(filename)
            keys = list(mat_file.keys())
            # Skip MATLAB-specific meta fields like __header__, __version__, __globals__
            data_keys = [key for key in keys if not key.startswith('__')]
            if not data_keys:
                raise ValueError("No datasets found in the .mat file.")
            dataset_name = data_keys[0]
            imgfile = Image.fromarray(mat_file[dataset_name])  # Load array from mat file
            return imgfile
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    #print(idx)

    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset_2(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, D_dir: str,D_range: list, scale: float = 1.0, mask_suffix: str = '', D_suffix: str = '' ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.D_dir = Path(D_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.D_suffix = D_suffix
        self.D_range = D_range

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        #logging.info(self.ids)
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, D_range, img_type):
        #type 1 = mask, type 2: D, type 3: img
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if img_type == 1 or img_type ==2 else Image.BICUBIC)
        img = np.asarray(pil_img)

        if img_type == 1:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask
        elif img_type == 2:
            img = (img - D_range[0])/(D_range[1] - D_range[0])
            return img
        elif img_type == 3:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
#                img = img / 255.0
                img = (img - np.min(img) )/np.max(img)  #Normalize based on the largest value 

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        D_file = list(self.D_dir.glob(name + self.D_suffix + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(D_file) == 1, f'Either no D_map or multiple D_map found for the ID {name}: {D_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        D = load_image(D_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale,  self.D_range, img_type = 3)
        mask = self.preprocess(self.mask_values, mask, self.scale,  self.D_range, img_type = 1)
        D = self.preprocess(self.mask_values, D ,self.scale,  self.D_range, img_type = 2 )


        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'D': torch.as_tensor(D.copy()).float().contiguous()
        }


class CarvanaDataset_2(BasicDataset_2):
    def __init__(self, images_dir, mask_dir, D_dir, D_range, scale=1):
        super().__init__(images_dir, mask_dir, D_dir, D_range, scale, mask_suffix='_mask', D_suffix = '_D')

