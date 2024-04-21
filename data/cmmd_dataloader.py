import os
import optparse as op
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
from collections import Counter
from utils_data.crop_breast import suppress_artifacts, crop_max_bg
# import cv2
import random
import pandas as pd

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
# from models.mEfficientNet import EfficientNet
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import wandb
from glob import glob
from torchvision.transforms import functional as transF

train_h, train_w = 1536, 768

def get_test_transform(config):
    trans_val_test = transforms.Compose([
            transforms.Resize(size=[train_h, train_w]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True}

    return trans_val_test, kwargs


class CMMD_dataset(Dataset):
    """Image and label dataset."""

    def __init__(self, transform=None, args=None, mode="train"):
        """
        Args:
            csv_file (string): Csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        root = "/mnt/HD/Dataset/CMMD/"
        csv_file = os.path.join(root, "CMMD_clinicaldata_revision.xlsx")
        self.df = pd.read_excel(csv_file)
        self.root_dir = os.path.join(root, "IMAGE_CROP")
        self.transform = transform
        self.args = args
        self.mode = mode
        self.memory = None

    # def aux_sampler(self, current_ax, ipsi_view):
    #     input_index = current_ax[current_ax["ImageViewPosition"] == ipsi_view].index[0]
    #     df_ipsi = current_ax.loc[input_index]
    #     ipsi_data, ipsi_label = self.image_sampler(df_ipsi, ipsi_view)
    #     return ipsi_data, ipsi_label

    def image_sampler(self, file_path, viewpos):
        path = file_path
        img = Image.open(path)
        img = img.convert('RGB')
        normalization = []
        data = None
        if self.transform:
            data = self.transform(img)
            # img_mean = torch.mean(data)
            # img_std = torch.maximum(torch.std(data), torch.tensor(10 **xxxxxxx (-5)))
            # normalization.append(transforms.Normalize(mean=[img_mean], std=[img_std]))
            if viewpos == 'R':
                flip = transforms.RandomHorizontalFlip(p=1)
                normalization.append(flip)
            elif viewpos == 'L':
                do_sth = "nothing"
            else:
                raise ("Incorrect view position: {}".format(viewpos))
        transform_2 = transforms.Compose(normalization)
        data = transform_2(data)
        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        df_main = self.df.iloc[idx]

        laterality = df_main.LeftRight
        files = sorted(glob(os.path.join(self.root_dir, "{}_*_{}*".format(df_main.ID1, laterality))))


        # current_ax = self.df[self.df['AccessionId'] == ax_id]
        # viewpos = df_main.ImageViewPosition.split('-')
        # ipsi_view = "{}-{}".format(viewpos[0], ['MLO' if viewpos[1] == 'CC' else 'CC'][0])
        # bi_view = "{}-{}".format(['R' if viewpos[0]=='L' else 'L'][0], viewpos[1])

        main_view_pos = "CC"
        ipsi_view_pos = "MLO"
        assert main_view_pos != ipsi_view_pos

        main_data = self.image_sampler(files[0], laterality)
        ipsi_data = self.image_sampler(files[1], laterality)
        # ipsi_data, ipsi_label = self.aux_sampler(current_ax, ipsi_view)
        # bi_data, bi_label = self.aux_sampler(current_ax, bi_view)

        if random.random() > 0.5 and self.mode == "train":
            main_data = TF.vflip(main_data)
            ipsi_data = TF.vflip(ipsi_data)
            # bi_data = TF.vflip(bi_data)
        
        stack_tensor = torch.cat((
            main_data.unsqueeze(0),
            ipsi_data.unsqueeze(0),
            # bi_data.unsqueeze(0),
        ), dim=0)

        if df_main.classification == "Malignant":
            stack_label = float(1)
        elif df_main.classification == "Benign":
            stack_label = float(0)
        else:
            raise ValueError("Unknow label {}".format(df_main.classification))


        # skip = "YES"
        # if list(stack_label) == [0, 1, 1] or list(stack_label) == [0, 0, 1]:
        #     skip = "NO"
        

        # TODO -> Add gt mask
        affine_trans = transforms.RandomAffine(degrees=10, translate=(0, 0.2), scale=(0.9, 1.1), shear=(0, 0, 0, 45), fill=-2.1179)
        # CC -> 1, MLP ->2
        # view_marker = torch.tensor(1) if ipsi_view_pos == 'CC' else torch.tensor(2)
        view_marker = torch.tensor([1, 2])

        if self.mode == "train":
            # return affine_trans(stack_tensor), stack_label
            return affine_trans(stack_tensor), stack_label, view_marker#, df_main.name#ImageFilePath.split("/")[-1]
        elif self.mode == "test":
            # return stack_tensor, stack_label
            return stack_tensor, stack_label, view_marker#, df_main.name#.ImageFilePath.split("/")[-1]
        else:
            raise ValueError("Undefined Mode")
            
            
class CMMD_dataset_train(Dataset): 
    """Image and label dataset."""

    def __init__(self, csv_file, root_dir, transform=None, args=None, mode="labled"):

        """
        Args:
            csv_file (string): Csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.args = args
        self.mode = mode
        self.memory = None
        self.use_labelsm = args.label_smooth

    # def aux_sampler(self, current_ax, ipsi_view):
    #     input_index = current_ax[current_ax["ImageViewPosition"] == ipsi_view].index[0]
    #     df_ipsi = current_ax.loc[input_index]
    #     ipsi_data, ipsi_label = self.image_sampler(df_ipsi, ipsi_view)
    #     return ipsi_data, ipsi_label

    def image_sampler(self, df_main):
        path, label = df_main[0], df_main[1]
        img = Image.open(path)
        #############
        # img_path = os.path.join(self.root_dir, df_idx.ImageFilePath)
        # imgTmp = Image.open(img_path)
        # table = [i / 256 for i in range(65536)]
        # imgTmp = imgTmp.point(table, 'L')
        # img = np.asarray(imgTmp)
        # org = img.copy()
        # pad = 25
        # # y
        # org[0:pad, :] = 0view_marker
        # (img_suppr, breast_mask) = suppress_artifacts(org, global_threshold=.1, fill_holes=True,
        #                                               smooth_boundary=True, kernel_size=15)
        # img_breast_only = crop_max_bg(img_suppr, breast_mask)
        # imgTmp = Image.fromarray(img_breast_only)

        # # img = imgTmp.convert('RGB')
        # img = imgTmp.convert('L')
        # img = SquarePad(img, df_idx, train_h / train_w)
        #############

        img = img.convert('RGB')
        target = float(label)

        # normalization = []
        # data = None
        # if self.transform:
        # data = self.transform(img)
        # # img_mean = torch.mean(data)
        # # img_std = torch.maximum(torch.std(data), torch.tensor(10 ** (-5)))
        # # normalization.append(transforms.Normalize(mean=[img_mean], std=[img_std]))
        # if viewpos[0] == 'R':
        # # flip = transforms.RandomHorizontalFlip(p=1)
        # # normalization.append(flip)
        # do_sth = "nothing"
        # elif viewpos[0] == 'L':
        # do_sth = "nothing"
        # else:
        # raise ("Incorrect view position: {}".format(viewpos))
        # transform_2 = transforms.Compose(normalization)
        # data = transform_2(data)

        if self.transform == 'affine_consistent_train':
            # resize
            trans_resize = transforms.Compose([
                transforms.RandomResizedCrop(size=(1536, 1536 // 2), scale=(0.85, 1.0), ratio=(0.3, 0.7)),
                transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)),
                transforms.RandomHorizontalFlip(p=0.5)
            ])
            img_PIL = trans_resize(img)

            # affine
            trans_affine_param = transforms.RandomAffine(degrees=[-10, 10]).get_params(degrees=[-10, 10],
                                                                                       translate=[0.15, 0.15],
                                                                                       scale_ranges=[0.95, 1.05],
                                                                                       shears=[-10, 10, -10, 10],
                                                                                       img_size=[1536,
                                                                                                 1536 // 2])
            img_affine = transF.affine(img_PIL, *trans_affine_param, fill=0)

            # plt.imshow(np.array(img_affine), 'gray')
            # plt.show()
            # plt.imshow(np.array(img_PIL), 'gray')
            # plt.show()

            # normalize
            trans_normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            img_normalize = trans_normalize(img_affine)
            img_PIL = trans_normalize(img_PIL)

            # print(trans_affine_param)

            trans_affine_param_new = list(trans_affine_param)
            trans_affine_param_new[0] = (trans_affine_param[0], 0)
            trans_affine_param_new[2] = (trans_affine_param[2], 0)
            trans_affine_param_new = np.array(trans_affine_param_new)
            # print(trans_affine_param_new)

            data = (img_normalize, img_PIL, trans_affine_param_new)

        else:

            data = self.transform(img)

        target = torch.tensor(target)

        return data, target, path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
    
        flag_islabeled = torch.tensor(1).long()

        df_main = self.df[idx]

        main_data, main_label, imageFileFullPath = self.image_sampler(df_main)

        stack_tensor = main_data
        stack_label = main_label

        # TODO -> Add gt mask
        # affine_trans = transforms.RandomAffine(degrees=10, translate=(0, 0.2), scale=(0.9, 1.1), shear=(0, 0, 0, 45), fill=-2.1179)
        # CC -> 1, MLP ->2
        # view_marker = torch.tensor(1) if ipsi_view_pos == 'CC' else torch.tensor(2)

        # print(stack_label, imageFileFullPath)

        if self.mode == "unlabled":
            return stack_tensor, stack_label.long(), flag_islabeled, imageFileFullPath
        elif self.mode == "labeled":
            return stack_tensor, stack_label.long(), imageFileFullPath
        else:
            raise ValueError("Undefined Mode")
