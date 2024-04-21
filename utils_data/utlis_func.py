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
from data.mammo_dataloader import MAMMO_Dataset
from data.mammo_dataloader import MAMMO_Dataset_unlabeled
from data.cmmd_dataloader import CMMD_dataset_train
import wandb


train_h, train_w = 1536, 768


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def config_dataset_pretrain(params):

    # Set seeds and deterministic pytorch for reproducibility
    random.seed(params.seed)  # python random seed
    np.random.seed(params.seed)  # numpy random seed
    torch.manual_seed(params.seed)  # pytorch random seed
    torch.cuda.manual_seed(params.seed)  # pytorch random seed
    torch.backends.cudnn.deterministic = True
       

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.85, 1.0), ratio=(0.3, 0.7)),  # the ratio must be centered at 0.5      #####################
            transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)),  # this transform cannot convert gray to color
            transforms.RandomHorizontalFlip(p=0.5),      #############################
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.95, 1.05), shear=(-10, 10, -10, 10), fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    transform_push = transforms.Compose([
            transforms.Resize(size=[train_h, train_w]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    transform_test_val = transforms.Compose([
        transforms.Resize(size=[train_h, train_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load csv file
    file_paths = params.file_paths
    folder = file_paths.csv_path

    df_train = pd.read_csv(os.path.join(folder, file_paths.train_name), index_col="image_data_sha256", dtype=str)
    df_test = pd.read_csv(os.path.join(folder, file_paths.test_name), index_col="image_data_sha256", dtype=str)
    df_val = pd.read_csv(os.path.join(folder, file_paths.valid_name), index_col="image_data_sha256", dtype=str)
    
    
    # df = pd.read_csv(os.path.join(folder), index_col="image_data_sha256", dtype=str)
    # df["ImageOutcome"].replace({"2":"1", "3": "0", "4": "0"}, inplace=True)
    
    # df_train = df.loc[(df['train_valid_test'] == '0')]
    # df_val = df.loc[(df['train_valid_test'] == '1')]
    # df_test = df.loc[(df['train_valid_test'] == '2')]
    

    img_path = file_paths.img_path


    train_dataset = MAMMO_Dataset(csv_file=df_train, indexs=np.array(range(len(df_train))), root_dir=img_path, transform='affine_consistent_train', args=params, mode="train")
    train_push_dataset = MAMMO_Dataset(csv_file=df_train, indexs=np.array(range(len(df_train))), root_dir=img_path, transform=transform_test_val, args=params, mode="train")
    
    test_dataset = MAMMO_Dataset(csv_file=df_test, indexs=np.array(range(len(df_test))), root_dir=img_path, transform=transform_test_val, args=params, mode="test")
    valid_dataset = MAMMO_Dataset(csv_file=df_val, indexs=np.array(range(len(df_val))), root_dir=img_path, transform=transform_test_val, args=params, mode="test")
    

    train_loader = DataLoader(train_dataset, batch_size=params.train_batch_size, shuffle=True, drop_last=False, num_workers=0)
    train_push_loader = DataLoader(train_push_dataset, batch_size=params.train_push_batch_size, shuffle=True, drop_last=False, num_workers=0)

    test_loader = DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=0)

    print('Number of train images: {}'.format(len(df_train)))
    print('Number of train push images: {}'.format(len(df_train)))   # for prototype projection and visualisation
    print('Number of test images: {}'.format(len(df_test)))
    print('Number of valid images: {}'.format(len(df_val)))
    print('')

    return train_loader, train_push_loader, test_loader, valid_loader


def config_dataset_2stages(params):

    # Set seeds and deterministic pytorch for reproducibility
    random.seed(params.seed)  # python random seed
    np.random.seed(params.seed)  # numpy random seed
    torch.manual_seed(params.seed)  # pytorch random seed
    torch.cuda.manual_seed(params.seed)  # pytorch random seed
    torch.backends.cudnn.deterministic = True

       
    transform_labeled = transforms.Compose([
            transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.85, 1.0), ratio=(0.3, 0.7)),  # the ratio must be centered 0.5                #####################
            transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)),  # this transform cannot convert gray to color
            transforms.RandomHorizontalFlip(p=0.5),      #############################
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.95, 1.05), shear=(-10, 10, -10, 10), fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
    transform_unlabeled = TransformMPL(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transform_unlabeled = transforms.Compose([
            # transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.85, 1.0), ratio=(0.3, 0.7)),  # the ratio must be centered 0.5                #####################
            # transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)),  # this transform cannot convert gray to color
            # transforms.RandomHorizontalFlip(p=0.5),      #############################
            # transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.95, 1.05), shear=(-10, 10, -10, 10), fill=0),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # maybe different !!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ])
            
    transform_push = transforms.Compose([
            transforms.Resize(size=[train_h, train_w]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])   
            
    transform_test_val = transforms.Compose([
        transforms.Resize(size=[train_h, train_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load csv file
    file_paths = params.file_paths
    folder = file_paths.csv_path

    df_train = pd.read_csv(os.path.join(folder, file_paths.train_name), index_col="image_data_sha256", dtype=str)
    df_train_unlabeled = pd.read_csv(os.path.join(folder, file_paths.train_name_unlabeled), index_col="image_data_sha256", dtype=str)
    df_test = pd.read_csv(os.path.join(folder, file_paths.test_name), index_col="image_data_sha256", dtype=str)
    df_val = pd.read_csv(os.path.join(folder, file_paths.valid_name), index_col="image_data_sha256", dtype=str)
    

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(1.0, df_train)    # 1/8, 1/4, 1/2


    df_train_all = df_train.append(df_train_unlabeled)
    

    img_path = file_paths.img_path

    train_labeled_dataset = MAMMO_Dataset(csv_file=df_train, indexs=train_labeled_idxs, root_dir=img_path, transform=transform_labeled, args=params, mode="train")

    # train_unlabeled_dataset = MAMMO_Dataset_unlabeled(csv_file=df_train, indexs=train_unlabeled_idxs, root_dir=img_path, labeled_indexs=train_labeled_idxs, transform='affine_consistent_train', args=params, mode="train")    # fully
    train_unlabeled_dataset = MAMMO_Dataset_unlabeled(csv_file=df_train_all, indexs=np.array(range(len(df_train_all))), root_dir=img_path, labeled_indexs=np.array(range(len(df_train))), transform='affine_consistent_train', args=params, mode="train")    # semi
    # train_unlabeled_dataset = MAMMO_Dataset_unlabeled(csv_file=df_train_unlabeled, indexs=np.array(range(len(df_train_unlabeled))), root_dir=img_path, labeled_indexs=np.array(range(len(df_train_unlabeled))), transform=transform_labeled, args=params, mode="train")  # retrain/semi

    train_push_dataset = MAMMO_Dataset(csv_file=df_train, indexs=train_labeled_idxs, root_dir=img_path, transform=transform_test_val, args=params, mode="train")
    test_dataset = MAMMO_Dataset(csv_file=df_test, indexs=np.array(range(len(df_test))), root_dir=img_path, transform=transform_test_val, args=params, mode="test")
    valid_dataset = MAMMO_Dataset(csv_file=df_val, indexs=np.array(range(len(df_val))), root_dir=img_path, transform=transform_test_val, args=params, mode="test")
    

    labeled_loader = DataLoader(train_labeled_dataset, batch_size=params.train_batch_size, shuffle=True, drop_last=False, num_workers=0)
    unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=params.train_batch_size, shuffle=True, drop_last=False, num_workers=0)
    train_push_loader = DataLoader(train_push_dataset, batch_size=params.train_push_batch_size, shuffle=True, drop_last=False, num_workers=0)

    test_loader = DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=0)    # 0
    valid_loader = DataLoader(valid_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=0)

    print('Number of train labeled images: {}'.format(train_labeled_idxs.shape[0]))
    print('Number of train unlabeled images: {}'.format(train_unlabeled_idxs.shape[0]))
    print('Number of train push images: {}'.format(train_labeled_idxs.shape[0]))
    print('Number of test images: {}'.format(len(df_test)))
    print('Number of valid images: {}'.format(len(df_val)))
    print('')

    return labeled_loader, unlabeled_loader, train_push_loader, test_loader, valid_loader



def config_dataset_cmmd(params):
    # config.update(params)
    # config.model_name = params.model_name  # String. Network model
    # config.use_pretrained = params.use_pretrained  # Boolean. Use pretrained network
    # config.use_augmentation = params.use_augmentation  # Boolean. Use data augmentation
    # config.use_gamma = params.use_gamma  # Boolean. Use gamma correction
    # config.use_crop = params.use_crop  # Boolean. Use image cropping
    # config.num_workers = params.num_workers  # number of CPU threads
    # config.batch_size = params.batch_size  # input batch size for training (default: 64)
    # config.epochs = params.epochs  # number of epochs to train (default: 10)
    # config.lr = params.lr  # learning rate (default: 0.01)
    # config.seed = params.seed  # random seed (default: 42) #103 #414
    # config.earlystop_patience = params.earlystop_patience  # epochs to wait without improvements before stopping training
    # config.reduceLR_patience = params.reduceLR_patience  # epoch to wait without improvements before reducing learning rate
    # config.fc_size = params.fc_size  # size of fully connected layer
    # config.manufacturer_train = params.manufacturer_train
    kwargs = {'num_workers': params.num_workers, 'pin_memory': True}
    # cv2.setNumThreads(0)
    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(params.seed)  # python random seed
    torch.manual_seed(params.seed)  # pytorch random seed
    np.random.seed(params.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
       
   
       
    transform_labeled = transforms.Compose([
            transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.85, 1.0), ratio=(0.3, 0.7)),  # the ratio must be centered 0.5                #####################
            transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)),  # this transform cannot convert gray to color
            transforms.RandomHorizontalFlip(p=0.5),      #############################
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.95, 1.05), shear=(-10, 10, -10, 10), fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
    # transform_labeled = transforms.Compose([
        # transforms.Resize(size=[train_h, train_w]),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
            
    # transform_unlabeled = TransformMPL(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_unlabeled = transforms.Compose([
            transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.85, 1.0), ratio=(0.3, 0.7)),  # the ratio must be centered 0.5                #####################
            transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)),  # this transform cannot convert gray to color
            transforms.RandomHorizontalFlip(p=0.5),      #############################
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.95, 1.05), shear=(-10, 10, -10, 10), fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
    transform_push = transforms.Compose([
            transforms.Resize(size=[train_h, train_w]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])   
            
    transform_test_val = transforms.Compose([
        transforms.Resize(size=[train_h, train_w]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load csv file
    file_paths = params.file_paths
    folder = file_paths.csv_path

    df_train = pd.read_csv(os.path.join(folder, file_paths.train_name), dtype=str)
    df_train = df_train.loc[(df_train['ID1'] != 'D1-1343')]
    df_test = pd.read_csv(os.path.join(folder, file_paths.test_name), dtype=str)
    df_val = pd.read_csv(os.path.join(folder, file_paths.valid_name), dtype=str)
    
    img_path = file_paths.img_path


    list_train = df2list_cmmd(img_path, df_train)
    list_valid = df2list_cmmd(img_path, df_val)
    list_test = df2list_cmmd(img_path, df_test)
    list_test.extend(list_valid)


    train_labeled_dataset = CMMD_dataset_train(csv_file=list_train, root_dir=img_path, transform=transform_labeled, args=params, mode="labeled")
    train_unlabeled_dataset = CMMD_dataset_train(csv_file=list_train, root_dir=img_path, transform='affine_consistent_train', args=params, mode="unlabled")
    train_push_dataset = CMMD_dataset_train(csv_file=list_train, root_dir=img_path, transform=transform_test_val, args=params, mode="labeled")
    
    test_dataset = CMMD_dataset_train(csv_file=list_test, root_dir=img_path, transform=transform_test_val, args=params, mode="labeled")
    valid_dataset = CMMD_dataset_train(csv_file=list_valid, root_dir=img_path, transform=transform_test_val, args=params, mode="labeled")
    

    labeled_loader = DataLoader(train_labeled_dataset, batch_size=params.train_batch_size, shuffle=True, drop_last=False, num_workers=6)
    unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=params.train_batch_size, shuffle=True, drop_last=False, num_workers=6)
    train_push_loader = DataLoader(train_push_dataset, batch_size=params.train_push_batch_size, shuffle=True, drop_last=False, num_workers=6)
    
    # test_loader = DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=6)
    valid_loader = DataLoader(valid_dataset, batch_size=params.test_batch_size, shuffle=False, drop_last=False, num_workers=6)

    print('Number of train labeled images: {}'.format(len(train_labeled_dataset)))
    print('Number of train unlabeled images: {}'.format(len(train_unlabeled_dataset)))
    print('Number of train push images: {}'.format(len(train_push_dataset)))
    print('Number of test images: {}'.format(len(test_dataset)))
    print('Number of vald images: {}'.format(len(valid_dataset)))
    print('')

    return labeled_loader, unlabeled_loader, train_push_loader, test_loader, valid_loader



def x_u_split(num_labeled_ratio, image_list_train):

    targets = np.array([float(image_list_train.iloc[index].image_outcome) for index in range(len(image_list_train))])
    cancer_idx = np.where(targets == 1)[0]
    non_cancer_idx = np.where(targets == 0)[0]

    np.random.seed(42)   # fixed  1234
    cancer_labeled_idx = np.random.choice(cancer_idx, int(num_labeled_ratio * cancer_idx.shape[0]), False)
    np.random.seed(42)   # fixed  1234
    non_cancer_labeled_idx = np.random.choice(non_cancer_idx, int(num_labeled_ratio * non_cancer_idx.shape[0]), False)

    print('cancer_labeled_num:', cancer_labeled_idx.shape[0], 'cancer_labeled_idx:', cancer_labeled_idx)
    print('noncancer_labeled_num:', non_cancer_labeled_idx.shape[0], 'noncancer_labeled_idx:', non_cancer_labeled_idx)

    labeled_idx = np.array(cancer_labeled_idx.tolist() + non_cancer_labeled_idx.tolist())
    np.random.shuffle(labeled_idx)

    # all training data as unlabeled
    unlabeled_idx = np.array(range(targets.shape[0]))

    return labeled_idx, unlabeled_idx


def df2list_cmmd(root_dir, df):
   from glob import glob

   num = len(df)

   df_list = []
   for i in range(num):
       df_main = df.iloc[i]
       laterality = df_main.LeftRight
       files = sorted(glob(os.path.join(root_dir, "{}_*_{}*".format(df_main.ID1, laterality))))
       label = int(df_main.label)
       files_with_labels = []
       for f in files:
           files_with_labels.append([f, label])
       df_list.extend(files_with_labels)

   return df_list


class TransformMPL(object):
    def __init__(self, mean, std):
        # if args.randaug:
        #     n, m = args.randaug
        # else:
        #     n, m = 2, 10  # default
        
        self.ori = transforms.Compose([
            transforms.RandomResizedCrop(size=[train_h, train_w], scale=(0.85, 1.0), ratio=(0.3, 0.7)),  # the ratio must be centered 0.5
            transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.01, 0.01)),  # this transform cannot convert gray to color
            transforms.RandomHorizontalFlip(p=0.5),
            ])

        self.aug = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.95, 1.05), shear=(-10, 10, -10, 10), fill=0),
            ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(ori)
        return self.normalize(ori), self.normalize(aug)
        

