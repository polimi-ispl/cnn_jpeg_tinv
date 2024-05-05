"""
Small file for data utils
"""

# --- Libraries import
from typing import Union, Tuple, Any, Optional, Callable, List
import cv2
import numpy as np
import torch.utils.data
import pandas as pd
import torch
import os


# --- Classes


class CustomBlockJPEGBalancedDataset(torch.utils.data.Dataset):
    """
    Class for loading JPEG images and their uncompressed versions in a balanced way.
    The class handles also images compressed with a block size different from 8x8, e.g., 7x7, 9x9, 12x12.
    """
    def __init__(self, data_root: str, jpeg_bs: int, patch_size: int = 256, transforms: Optional[Callable] = None,
                 grayscale: bool=False, disaligned_grid_patch: Union[int, List[int]] = 0,
                 coherent_grid_patch: bool = True):
        """
        Constructor for the CustomBlockJPEGBalancedDataset class
        :param df_path: str, path to the dataframe containing the paths to the uncompressed and compressed images
        :param data_root: str, root directory where the images are stored
        :param patch_size: int, size of the patches to be extracted
        :param transforms: Albumentations transforms to be executed on the samples
        :param grayscale: bool, if True the images will be loaded as grayscale
        :param disaligned_grid_patch: disalignment of the patches with respect to the JPEG grid
        :param coherent_grid_patch: bool, if True the patches will be coherent with the JPEG grid
        """
        super(CustomBlockJPEGBalancedDataset, self).__init__()
        self.data_root = data_root
        self.block_size = jpeg_bs
        self.df = pd.read_pickle(os.path.join(data_root, 'data_df.pkl'))
        # Select only samples with the correct block size
        self.df = self.df.loc[self.block_size]
        self.df = self.df.loc[self.df['jpeg_image_path']!='EMPTY']  # filter out samples with problems
        self.transforms = transforms
        self.grayscale = grayscale
        if not ((type(disaligned_grid_patch) == list) or (type(disaligned_grid_patch) == tuple)):
            self.disaligned_grid_patch = [disaligned_grid_patch, disaligned_grid_patch]
        else:
            self.disaligned_grid_patch = disaligned_grid_patch
        # Compute patch_size as the closest multiple of the block size used for the JPEG compression
        # In this way we can choose to work with either aligned or disaligned patches to the JPEG grid
        self.patch_size = self.block_size*round(patch_size/self.block_size) if ((patch_size % self.block_size != 0) and coherent_grid_patch) \
                          else patch_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Select sample
        row = self.df.iloc[idx]

        # --- Load and transform uncompressed sample
        if self.grayscale:
            uncompressed = cv2.imread(os.path.join(self.data_root, row.name),
                                      cv2.IMREAD_GRAYSCALE).astype(np.float32)
        else:
            uncompressed = cv2.cvtColor(os.path.join(self.data_root, cv2.imread(row.name)),
                                        cv2.COLOR_BGR2RGB).astype(np.float32)
        uncompressed -= 127
        uncompressed /= 127  # normalize between -1 and 1
        if len(self.transforms) > 1:  # random crop has been enabled
            uncompressed = self.transforms(image=uncompressed)['image']
        else:
            if (self.disaligned_grid_patch[0] > 0) or (self.disaligned_grid_patch[1] > 0):
                if ((self.patch_size + self.disaligned_grid_patch[0] > uncompressed.shape[0]) or
                        (self.patch_size + self.disaligned_grid_patch[1] > uncompressed.shape[1])):
                    mirrored = cv2.copyMakeBorder(uncompressed,
                                                  0,
                                                  np.abs(self.patch_size + self.disaligned_grid_patch[0] - uncompressed.shape[0]),
                                                  0,
                                                  np.abs(self.patch_size + self.disaligned_grid_patch[1] - uncompressed.shape[1]),
                                                  cv2.BORDER_WRAP)
                    mirrored = mirrored[self.disaligned_grid_patch[0]:self.disaligned_grid_patch[0]+self.patch_size,
                                        self.disaligned_grid_patch[1]:self.disaligned_grid_patch[1]+self.patch_size]
                    uncompressed = self.transforms(image=mirrored)['image']
                else:
                    uncompressed = self.transforms(image=uncompressed[self.disaligned_grid_patch[0]:self.disaligned_grid_patch[0]+self.patch_size,
                                                                      self.disaligned_grid_patch[1]:self.disaligned_grid_patch[1]+self.patch_size])['image']
            else:
                min_shape = np.min(uncompressed.shape)
                if self.patch_size > min_shape:
                    mirrored = cv2.copyMakeBorder(uncompressed, 0, self.patch_size-min_shape, 0, self.patch_size-min_shape,
                                                  cv2.BORDER_WRAP)
                    mirrored = mirrored[:self.patch_size, :] if uncompressed.shape[0] > uncompressed.shape[1] \
                               else mirrored[:, :self.patch_size]
                    uncompressed = self.transforms(image=mirrored)['image']
                else:
                    uncompressed = self.transforms(image=uncompressed[0:self.patch_size, 0:self.patch_size])['image']

        # --- Load and transform compressed sample
        if self.grayscale:
            compressed = cv2.imread(os.path.join(self.data_root, row['jpeg_image_path']),
                                    cv2.IMREAD_GRAYSCALE).astype(np.float32)
        else:
            compressed = cv2.cvtColor(os.path.join(self.data_root, cv2.imread(row['jpeg_image_path'])),
                                      cv2.COLOR_BGR2RGB).astype(np.float32)
        compressed -= 127
        compressed /= 127  # normalize between -1 and 1
        if len(self.transforms) > 1:  # random crop has been enabled
            compressed = self.transforms(image=compressed)['image']
        else:
            if (self.disaligned_grid_patch[0] > 0) or (self.disaligned_grid_patch[1] > 0):
                if ((self.patch_size + self.disaligned_grid_patch[0] > compressed.shape[0]) or
                        (self.patch_size + self.disaligned_grid_patch[1] > compressed.shape[1])):
                    mirrored = cv2.copyMakeBorder(compressed,
                                                  0,
                                                  np.abs(self.patch_size + self.disaligned_grid_patch[0] - compressed.shape[0]),
                                                  0,
                                                  np.abs(self.patch_size + self.disaligned_grid_patch[1] - compressed.shape[1]),
                                                  cv2.BORDER_WRAP)
                    mirrored = mirrored[self.disaligned_grid_patch[0]:self.disaligned_grid_patch[0] + self.patch_size,
                               self.disaligned_grid_patch[1]:self.disaligned_grid_patch[1] + self.patch_size]
                    compressed = self.transforms(image=mirrored)['image']
                else:
                    compressed = self.transforms(image=compressed[self.disaligned_grid_patch[0]:self.disaligned_grid_patch[0]+self.patch_size,
                                                                  self.disaligned_grid_patch[1]:self.disaligned_grid_patch[1]+self.patch_size])['image']
            else:
                min_shape = np.min(compressed.shape)
                if self.patch_size > min_shape:
                    mirrored = cv2.copyMakeBorder(compressed, 0, self.patch_size - min_shape, 0,
                                                  self.patch_size - min_shape,
                                                  cv2.BORDER_WRAP)
                    mirrored = mirrored[:self.patch_size, :] if compressed.shape[0] > compressed.shape[1] \
                        else mirrored[:, :self.patch_size]
                    compressed = self.transforms(image=mirrored)['image']
                else:
                    compressed = self.transforms(image=compressed[0:self.patch_size, 0:self.patch_size])['image']
        # Patch together the samples
        sample = torch.cat((uncompressed.unsqueeze(0), compressed.unsqueeze(0)))
        target = torch.Tensor((0, 1))
        return sample, target


class JPEG8x8BalancedDataset(torch.utils.data.Dataset):
    """
        Custom class for balancing the datasets to have couple of uncompressed-compressed images.
        This dataset works only with 8x8 JPEG compressed images, and is meant for the SOTA models in order to extract
        a greater number of patches from the same image.
    """

    def __init__(self, data_root: str, patch_size: int = 256, transforms: Optional[Callable] = None,
                 grayscale: bool = False, disaligned_grid_patch: bool = False):
        super(JPEG8x8BalancedDataset, self).__init__()
        self.data_root = data_root
        self.df = pd.read_pickle(os.path.join(data_root, 'data_df.pkl'))
        # Select 8x8 samples
        self.df = self.df.loc[8]
        self.df = self.df.loc[self.df['jpeg_image_path'] != 'EMPTY']  # filter out samples with problems
        self.transforms = transforms
        self.grayscale = grayscale
        self.disaligned_grid_patch = disaligned_grid_patch
        # Compute patch_size as the closest multiple of the block size used for the JPEG compression
        # In this way we can choose to work with either aligned or disaligned patches to the JPEG grid
        self.block_size = 8
        self.patch_size = self.block_size * round(patch_size / self.block_size) if patch_size % self.block_size != 0 \
            else patch_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Select sample
        row = self.df.iloc[idx]

        # --- Load and transform uncompressed sample
        if self.grayscale:
            # to pass it to pre-trained models on ImageNet, we replicate the grayscale to three channels
            uncompressed = np.repeat(cv2.imread(os.path.join(self.data_root, row.name),
                                                  cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis], 3, axis=2)
        else:
            uncompressed = cv2.cvtColor(cv2.imread(os.path.join(self.data_root, row.name)),
                                        cv2.COLOR_BGR2RGB).astype(np.float32)
        uncompressed = self.transforms(image=uncompressed)['image']

        # --- Load and transform compressed sample
        if self.grayscale:
            compressed = np.repeat(cv2.imread(os.path.join(self.data_root, row['jpeg_image_path']),
                                              cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis], 3,
                                   axis=2)
        else:
            compressed = cv2.cvtColor(os.path.join(self.data_root, cv2.imread(row['jpeg_image_path'])),
                                      cv2.COLOR_BGR2RGB).astype(np.float32)
        compressed = self.transforms(image=compressed)['image']

        if self.disaligned_grid_patch:
            # FOR RANDOM DISALIGNMENT:
            # extract squared patches such that they are DISALIGNED to the 8 x 8 pixels grid.
            # all possible row and col starting points
            row_idxs = range(0, uncompressed.shape[1] - self.patch_size)
            col_idxs = range(0, uncompressed.shape[2] - self.patch_size)
            # consider the list of multiples of 8
            aligned_row_8multiples = range(0, uncompressed.shape[1] - self.patch_size, 8)
            aligned_col_8multiples = range(0, uncompressed.shape[2] - self.patch_size, 8)
            # remove these lists from the original ones.
            row_idxs = [i for i in row_idxs if i not in aligned_row_8multiples]
            col_idxs = [i for i in col_idxs if i not in aligned_col_8multiples]
        else:
            # extract squared patches such that they are aligned to the 8 x 8 pixels grid.
            row_idxs = range(0, uncompressed.shape[1] - self.patch_size, 8)
            col_idxs = range(0, uncompressed.shape[2] - self.patch_size, 8)

        # considers only 25 patches per img?
        row_idxs = np.random.choice(row_idxs, size=5)
        col_idxs = np.random.choice(col_idxs, size=5)
        patch_list = []
        label_patch_list = []
        for r in row_idxs:
            for c in col_idxs:
                patch_list.append(uncompressed[:, r:r + self.patch_size, c:c + self.patch_size])
                patch_list.append(compressed[:, r:r + self.patch_size, c:c + self.patch_size])
                label_patch_list.append([0, 1])

        # Patch together the samples
        sample = torch.stack(patch_list, dim=0)
        target = torch.FloatTensor(np.concatenate(label_patch_list))
        return sample, target


# --- Functions
def balanced_collate_fn(batch):
    """
    Collate function for the balanced datasets
    :param batch: list, batch of samples and targets
    :return: list, batch of samples and targets concatenated with torch
    """
    return [torch.cat([elem[0] for elem in batch]), torch.cat([elem[1] for elem in batch])]


