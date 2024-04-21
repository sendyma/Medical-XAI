import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
from utils_data.crop_breast import suppress_artifacts, crop_max_bg
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import wandb
from data.data_oper import GammaCorrection, SquarePad
from torchvision.transforms import functional as transF


train_h, train_w = 1536, 768


class MAMMO_Dataset(Dataset):
    """Image and label dataset."""

    def __init__(self, csv_file, indexs, root_dir, transform=None, args=None, mode="train"):
        """
        Args:
            csv_file (string): Csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = csv_file
        self.indexs = indexs
        self.root_dir = root_dir
        self.transform = transform
        self.args = args
        self.mode = mode
        self.use_labelsm = args.label_smooth

    def image_sampler(self, df_idx, viewpos):
        path = os.path.join(self.root_dir, df_idx.image_relative_filepath)
        img = Image.open(path)

        #############  used for online cropping
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
        target = float(df_idx.image_outcome)

        if self.transform == 'affine_consistent_train':
            # resize
            trans_resize = transforms.Compose([
                transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.85, 1.0), ratio=(0.3, 0.7)),
                transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)),
                transforms.RandomHorizontalFlip(p=0.5)
                ])
            img_PIL = trans_resize(img)

            # affine
            trans_affine_param = transforms.RandomAffine(degrees=[-10, 10]).get_params(degrees=[-10, 10],
                                                                                       translate=[0.15, 0.15],
                                                                                       scale_ranges=[0.95, 1.05],
                                                                                       shears=[-10, 10, -10, 10],
                                                                                       img_size=[train_h, train_w])
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
        return len(self.indexs)

    def label_smooth(self, target):
        if target.item() == float(1):
            return target - 0.1
        elif target.item() == float(0):
            return torch.tensor(0.1)
        else:
            raise ValueError("Unknown label {}".format(target))

    def __getitem__(self, index):
    
        real_index = self.indexs[index]
        
        df_main = self.df.iloc[real_index]
        viewpos = [df_main.image_laterality, df_main.image_view_position]

        main_data, main_label, imageFileFullPath = self.image_sampler(df_main, viewpos)

        stack_tensor = main_data
        stack_label = main_label
        
        if self.mode == "train":
            return stack_tensor, stack_label.long(), imageFileFullPath
        elif self.mode == "test":
            return stack_tensor, stack_label.long(), imageFileFullPath
        elif self.mode == "push":
            return stack_tensor, stack_label.long(), imageFileFullPath
        else:
            raise ValueError("Undefined Mode")


class MAMMO_Dataset_unlabeled(Dataset):
    """Image and label dataset."""

    def __init__(self, csv_file, indexs, root_dir, labeled_indexs, transform=None, args=None, mode="train"):
        """
        Args:
            csv_file (string): Csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = csv_file
        self.indexs = indexs
        self.labeled_indexs = labeled_indexs
        self.root_dir = root_dir
        self.transform = transform
        self.args = args
        self.mode = mode
        self.memory = None
        self.use_labelsm = args.label_smooth

    def aux_sampler(self, current_ax, ipsi_view):
        input_index = current_ax[current_ax["ImageViewPosition"] == ipsi_view].index[0]
        df_ipsi = current_ax.loc[input_index]
        ipsi_data, ipsi_label = self.image_sampler(df_ipsi, ipsi_view)
        return ipsi_data, ipsi_label

    def image_sampler(self, df_idx, viewpos):
        path = os.path.join(self.root_dir, df_idx.image_relative_filepath)
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
        target = float(df_idx.image_outcome)
        # normalization = []
        # data = None
        # if self.transform:
        #     data = self.transform(img)
        #     # img_mean = torch.mean(data)
        #     # img_std = torch.maximum(torch.std(data), torch.tensor(10 ** (-5)))
        #     # normalization.append(transforms.Normalize(mean=[img_mean], std=[img_std]))
        #     if viewpos[0] == 'R':
        #         # flip = transforms.RandomHorizontalFlip(p=1)
        #         # normalization.append(flip)
        #         do_sth = "nothing"
        #     elif viewpos[0] == 'L':
        #         do_sth = "nothing"
        #     else:
        #         raise ("Incorrect view position: {}".format(viewpos))
        # # transform_2 = transforms.Compose(normalization)
        # # data = transform_2(data)

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

            data = (img_PIL, img_normalize, trans_affine_param_new)

        else:

            data = self.transform(img)

        target = torch.tensor(target)
        return data, target, path

    def __len__(self):
        return len(self.indexs)

    def label_smooth(self, target):
        if target.item() == float(1):
            return target - 0.1
        elif target.item() == float(0):
            return torch.tensor(0.1)
        else:
            raise ValueError("Unknown label {}".format(target))

    def __getitem__(self, index):

        real_index = self.indexs[index]

        if real_index in self.labeled_indexs:
            flag_islabeled = torch.tensor(1).long()
        else:
            flag_islabeled = torch.tensor(0).long()

        df_main = self.df.iloc[real_index]
        # if self.memory is None:
        #    self.memory = self.df.iloc[[index]]
        # else:
        #    self.memory = self.memory.append(self.df.iloc[[index]])

        # ax_id = df_main.AccessionId
        # current_ax = self.df[self.df['AccessionId'] == ax_id]
        # viewpos = df_main.image_view_position.split('-')

        viewpos = [df_main.image_laterality, df_main.image_view_position]

        # ipsi_view = "{}-{}".format(viewpos[0], ['MLO' if viewpos[1] == 'CC' else 'CC'][0])
        # bi_view = "{}-{}".format(['R' if viewpos[0]=='L' else 'L'][0], viewpos[1])

        # main_view_pos = viewpos[1]
        # ipsi_view_pos = ipsi_view.split("-")[1]
        # assert main_view_pos != ipsi_view_pos

        main_data, main_label, imageFileFullPath = self.image_sampler(df_main, viewpos)

        # if random.random() > 0.5 and self.mode == "train":
        # main_data = TF.vflip(main_data)

        # if self.mode == "train" and self.use_labelsm:
        # main_label = self.label_smooth(main_label)

        stack_tensor = main_data
        stack_label = main_label

        # TODO -> Add gt mask
        # affine_trans = transforms.RandomAffine(degrees=10, translate=(0, 0.2), scale=(0.9, 1.1), shear=(0, 0, 0, 45), fill=-2.1179)
        # CC -> 1, MLP ->2
        # view_marker = torch.tensor(1) if ipsi_view_pos == 'CC' else torch.tensor(2)

        # print(stack_label, imageFileFullPath)

        if self.mode == "train":
            return stack_tensor, stack_label.long(), flag_islabeled, imageFileFullPath
        elif self.mode == "test":
            return stack_tensor, stack_label.long(), flag_islabeled, imageFileFullPath
        else:
            raise ValueError("Undefined Mode")