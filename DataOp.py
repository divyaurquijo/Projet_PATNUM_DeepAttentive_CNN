import os
import glob

from volumentations.core.composition import Compose
import albumentations as A
import nibabel as nib
import numpy as np
import random

import torch
from torch.utils.data import Dataset

from volumentations import *


def get_list(dir_path):
    """
    This function is to read data from data dir.
    The data dir should be set as follow:
    -- Data
        -- train
            -- 01-T2SPIR-src.nii.gz
            -- 01-T2SPIR-mask.nii.gzv
        -- test
        ...
    """
    print("Reading Data...")
    train_dict_list = []
    test_dict_list = []

    train_path = os.path.join(dir_path, 'train')
    test_path = os.path.join(dir_path, 'test')

    train_ids = [1,2,3,5,8,10,13,19]
    test_ids = [21,22,32,39]

    for id in train_ids:
        train_dict_list.append(
            {
                'image_path': os.path.join(train_path, '%02d-T2SPIR-src.nii.gz'%(id)),
                'label_path': os.path.join(train_path, '%02d-T2SPIR-mask.nii.gz'%(id)),
            }
        )

    for id in test_ids:
        test_dict_list.append(
            {
                'image_path': os.path.join(test_path, '%02d-T2SPIR-src.nii.gz'%(id)),
            }
        )

    # we split the data set to train set(0.75), val set(0.25)
    train_ratio = 0.75
    train_num = round(len(train_dict_list)*train_ratio)
    train_list = train_dict_list[:train_num]
    val_list = train_dict_list[train_num:]

    test_list = test_dict_list[:]
    print("Finished! Train:{} Val:{} Test:{}".format(len(train_list), len(val_list), len(test_list)))

    return train_list, val_list, test_list

def get_augmentation():
    """
    here is the data augmentation compose function by packages volumentations:
    https://github.com/ashawkey/volumentations
    It provide a various augmentation strategy in 3D data
    """
    return Compose([
        Flip(0),
        #Flip(1),
        #Flip(2),
        RandomRotate90((0, 1)),
        #RandomRotate90((0, 2)),
        #RandomRotate90((1, 2))
    ], p=0.5)


class TrainGenerator(object):
    """
    This is the class to generate the patches
    """
    def __init__(self, data_list, batch_size, patch_size):
        self.data_list = data_list
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.aug = get_augmentation()

    def get_item(self):

        dict_list = random.sample(self.data_list, self.batch_size)

        image_list = [dict_item['image_path'] for dict_item in dict_list]
        label_list = [dict_item['label_path'] for dict_item in dict_list]

        image_patch, label_patch = self._sample_patch(image_list, label_list)

        return image_patch, label_patch

    def _sample_patch(self, image_list, clean_list):
        half_size = self.patch_size // 2
        image_patch_list = []
        label_patch_list = []

        for image_path, clean_path in zip(image_list, clean_list):
            image = nib.load(image_path).get_fdata()
            label = nib.load(clean_path).get_fdata()

            # here we augment the corresponding data and label
            data = {'image': image, 'label': label}
            aug_data = self.aug(**data)
            image, label = aug_data['image'], aug_data['label']

            w, h, d = image.shape

            label_index1 = np.where(label == 1)
            label_index2 = np.where(label == 2)
            label_index3 = np.where(label == 3)
            label_index4 = np.where(label == 4)

            label_concatx = np.concatenate((np.array(label_index1)[0], np.array(label_index2)[0].T, np.array(label_index3)[0].T,np.array(label_index4)[0].T), axis=0)
            label_concaty = np.concatenate((np.array(label_index1)[1], np.array(label_index2)[1].T, np.array(label_index3)[1].T,np.array(label_index4)[1].T), axis=0)
            label_concatz = np.concatenate((np.array(label_index1)[2], np.array(label_index2)[2].T, np.array(label_index3)[2].T,np.array(label_index4)[2].T), axis=0)

            label_index=(label_concatx, label_concaty, label_concatz)
            length_label = label_index[0].shape[0]

            p = random.random()
            # we set a probability(p) to make most of the center of sampling patches
            # locate to the regions with labels not background (for label 1, 2, 3 and 4)
            if p < 0.875:
                sample_id = random.randint(1, length_label-1)
                x, y, z = label_index[0][sample_id], label_index[1][sample_id], label_index[2][sample_id]
            else:
                x, y, z = random.randint(0, w), random.randint(0, h), random.randint(0, d)

            # here we prevent the sampling patch overflow volume
            if x < half_size:
                x = half_size
            elif x > w-half_size:
                x = w-half_size-1

            if y < half_size:
                y = half_size
            elif y > h-half_size:
                y = h-half_size-1

            if z < half_size:
                z = half_size
            elif z > d-half_size:
                z = d-half_size-1

            image_patch = image[x-half_size:x+half_size, y-half_size:y+half_size, z-half_size:z+half_size].astype(np.float32)
            label_patch = label[x-half_size:x+half_size, y-half_size:y+half_size, z-half_size:z+half_size].astype(np.float32)

            image_patch_list.append(image_patch[np.newaxis, np.newaxis, ...])
            label_patch_list.append(label_patch[np.newaxis, np.newaxis, ...])

        image_out = np.concatenate(image_patch_list, axis=0)
        label_out = np.concatenate(label_patch_list, axis=0)

        return image_out, label_out

