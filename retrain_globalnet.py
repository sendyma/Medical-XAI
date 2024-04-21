import os
import shutil
import math
import torch
import time
import argparse
import re
import numpy as np
import wandb
from utils_model.helpers import makedir
from models import model_globalnet
from models import model_protopnet
from utils_model import push, save
from utils_model.log import create_logger
import train_and_test as tnt
from utils_data.preprocess import mean, std, preprocess_input_function
from utils_data.config import create_config, optionFlags
from utils_data.utlis_func import *


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
args = parser.parse_args()


from utils_model.settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

model_dir ='saved_models/{}/'.format(datestr()) + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'utils_model/settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models/model_protopnet.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'utils_model/push.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'utils_data/utlis_func_aux_v2.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'data/mammo_dataloader.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'configs/mammogram-1-1f.yml'), dst=model_dir)


os.makedirs(os.path.join(model_dir, 'efficientnet'), exist_ok=True)
shutil.copy(src=os.path.join(os.getcwd(), 'efficientnet/model.py'), dst=os.path.join(model_dir, 'efficientnet'))
shutil.copy(src=os.path.join(os.getcwd(), 'efficientnet/utils.py'), dst=os.path.join(model_dir, 'efficientnet'))


log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


os.environ['WANDB_START_METHOD'] = 'fork'

# WandB – Initialize a new run
wandb.init(project='InterNRL', mode='disabled')     # mode='disabled'
wandb.run.name = 'mammo-GlobalNet-' + wandb.run.id



# load the data
from utils_model.settings import train_batch_size, test_batch_size, train_push_batch_size


def load_para(params):
    params = create_config(params.config_file)
    return params


params = optionFlags()
params.config_file = "configs/mammogram-1-1f.yml"
params = load_para(params)

params.train_batch_size = train_batch_size
params.test_batch_size = test_batch_size
params.train_push_batch_size = train_push_batch_size

print(params)

labeled_loader, unlabeled_loader, train_push_loader, test_loader, valid_loader = config_dataset_2stages(params)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training labeled set size: {0}'.format(len(labeled_loader.dataset)))
log('training unlabeled set size: {0}'.format(len(unlabeled_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('valid set size: {0}'.format(len(valid_loader.dataset)))
log('train labeled batch size: {0}'.format(train_batch_size))
log('train unlabeled batch size: {0}'.format(train_batch_size))
log('train push batch size: {0}'.format(train_push_batch_size))
log('test batch size: {0}'.format(test_batch_size))


# construct the model
globalnet = model_globalnet.construct_GlobalNet(base_architecture=base_architecture,
                                                pretrained=True,
                                                num_classes=num_classes)
globalnet = globalnet.cuda()
protopnet = model_protopnet.construct_ProtoPNet(base_architecture=base_architecture,  # 'vgg19'
                                                pretrained=True, img_size=img_size,  # 224
                                                prototype_shape=prototype_shape,  # (2000, 128, 1, 1)
                                                num_classes=num_classes,  # 200
                                                prototype_activation_function=prototype_activation_function,  # 'log'
                                                add_on_layers_type=add_on_layers_type)   # 'regular'
protopnet = protopnet.cuda()


# load ProtoPNet
ret_s = protopnet.load_state_dict(torch.load('demo_ckpt/epoch_explain_backbone27_S_0.8581.pth'), strict=False)   # pretrained accurate student model
print(ret_s)

# load Backbone
# ret_t = globalnet.load_state_dict(torch.load('demo_ckpt/epoch_explain_backbone27_S_0.8581.pth'), strict=False)   # use pretrained teacher
pretrained_dict = torch.load('demo_ckpt/epoch_explain_backbone27_S_0.8581.pth')   # only load backbone, retrain teacher from scratch
backbone_dict = {k: pretrained_dict[k] for k in pretrained_dict.keys() if 'backbone' in k}
ret_t = globalnet.load_state_dict(backbone_dict, strict=False)
print(ret_t)


protopnet_multi = torch.nn.DataParallel(protopnet)
globalnet_multi = torch.nn.DataParallel(globalnet)


class_specific = True

# define optimizer
from utils_model.settings import joint_optimizer_lrs, joint_lr_step_size
global_optimizer_specs = \
[
 {'params': globalnet.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
 {'params': globalnet.global_classification_layers.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},      ##########################  'lr': 1e-4
]
global_optimizer = torch.optim.Adam(global_optimizer_specs, amsgrad=True)
global_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(global_optimizer, factor=0.2, patience=3, threshold=1e-8, verbose=True)   # factor=0.5


wandb.watch(protopnet_multi, log="all")
wandb.watch(globalnet_multi, log="all")


# weighting of different training losses
from utils_model.settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from utils_model.settings import num_train_epochs, num_warm_epochs, push_start, push_epochs


# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config  # Initialize config

config.model_name = base_architecture  # String. Network model
config.use_pretrained = True  # Boolean. Use pretrained network
config.use_augmentation = True  # Boolean. Use data augmentation
config.use_gamma = False  # Boolean. Use gamma correction
config.use_crop = True  # Boolean. Use image cropping
config.num_workers = 8  # number of CPU threads
config.batch_size = 16  # input batch size for training (default: 64)
config.epochs = 100  # number of epochs to train (default: 10)
config.lr = 1e-3  # learning rate (default: 0.01)
config.seed = 42  # random seed (default: 42) #103 #414
config.earlystop_patience = None  # epochs to wait without improvements before stopping training
config.reduceLR_patience = None  # epoch to wait without improvements before reducing learning rate
config.fc_size = 4096  # size of fully connected layer
config.manufacturer_train = None
kwargs = {'num_workers': config.num_workers, 'pin_memory': True}


# train the model
log('start training')

valid_loss_min_teacher = np.Inf  # track change in validation loss

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    makedir(os.path.join(model_dir, 'model_regular'))

    # auc = tnt.test(globalnet_multi, protopnet_multi, dataloader=test_loader, class_specific=class_specific, log=log, train_type='retraining_globalnet')
    # break

    tnt.global_retraining(globalnet_multi, log=log)
    _ = tnt.train(globalnet_multi, protopnet_multi, labeled_loader, unlabeled_loader, global_optimizer, None,
                  class_specific=class_specific, coefs=coefs, log=log, epoch=epoch, train_type='retraining_globalnet')
    valid_loss_min_teacher = tnt.valid(globalnet_multi, None, dataloader=valid_loader, class_specific=class_specific,
                                       log=log, train_type='retraining_globalnet',
                                       valid_loss_min=valid_loss_min_teacher)
    global_scheduler.step(valid_loss_min_teacher)
    wandb.log({
        "Optimizer LR": global_optimizer.param_groups[0]['lr'],
        "Valid Loss Descend": valid_loss_min_teacher
    })
    if epoch % 1 == 0:
        auc = tnt.test(globalnet_multi, protopnet_multi, dataloader=test_loader, class_specific=class_specific,
                       log=log, train_type='retraining_globalnet')
        save.save_model_w_condition(globalnet, protopnet, model_dir=model_dir,
                                    model_name='GlobalNet_retrain' + str(epoch), auc=auc, target_auc=0.70, log=log)

logclose()

