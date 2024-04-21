import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time
from PIL import Image
from utils_model.receptive_field import compute_rf_prototype
from utils_model.helpers import makedir, find_high_activation_crop
import torchvision.transforms as transforms


# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel,  # (ppnet_multi_teacher, ppnet_multi_student)
                    class_specific=True,
                    preprocess_input_function=None,  # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved here
                    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,  # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel[0].eval()
    prototype_network_parallel[1].eval()

    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel[1].module.prototype_shape
    n_prototypes = prototype_network_parallel[1].module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    global_min_ori_img_names = ['.'] * n_prototypes

    global_min_prototype_info = {j: {
            'input_image_idx': [],
            'input_image_name': [],
            'patch_spatial_idx': [],
            'min_distance': [],
            'min_f_vector': [],
            'nearest_input': [],
            } for j in range(n_prototypes)}

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                 fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, 'epoch-' + str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel[1].module.num_classes

    for push_iter, (search_batch_input, search_y, img_name) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''

        start_index_of_search_batch = push_iter * search_batch_size

                                         

        if preprocess_input_function is not None:
            # print('preprocessing input for pushing ...')
            # search_batch = copy.deepcopy(search_batch_input)
            search_batch = preprocess_input_function(search_batch_input)
        else:
            search_batch = search_batch_input   # use here

        with torch.no_grad():
            search_batch = search_batch.cuda()
            # this computation currently is not parallelized
            feat_backbone = prototype_network_parallel[0].module.forward_backbone(search_batch).detach()
            protoL_input_torch, proto_dist_torch = prototype_network_parallel[1].module.push_forward(feat_backbone)

        protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())  # backbone CNN features   # [8, 128, 48, 24]
        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())  # distance maps  # [8, 100, 48, 24]

        del protoL_input_torch, proto_dist_torch

        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

        prototype_shape = prototype_network_parallel[1].module.prototype_shape
        n_prototypes = prototype_shape[0]
        proto_h = prototype_shape[2]
        proto_w = prototype_shape[3]

        for j in range(n_prototypes):

            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel[1].module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch, we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue

            for img_index_in_batch in class_to_img_index_dict[target_class]:
                proto_dist_j = proto_dist_[img_index_in_batch, j, :, :]    # [48, 24]
                batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))

                # retrieve the corresponding feature map patch
                fmap_height_start_index = batch_argmin_proto_dist_j[0] * prototype_layer_stride
                fmap_height_end_index = fmap_height_start_index + proto_h
                fmap_width_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
                fmap_width_end_index = fmap_width_start_index + proto_w

                batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                         :,
                                         fmap_height_start_index:fmap_height_end_index,
                                         fmap_width_start_index:fmap_width_end_index]

                batch_min_proto_dist_j = proto_dist_[img_index_in_batch,
                                         j,
                                         fmap_height_start_index:fmap_height_end_index,
                                         fmap_width_start_index:fmap_width_end_index]

                global_min_prototype_info[j]['input_image_idx'].append(push_iter * search_batch_size + img_index_in_batch)
                global_min_prototype_info[j]['input_image_name'].append(img_name[img_index_in_batch])
                global_min_prototype_info[j]['patch_spatial_idx'].append(batch_argmin_proto_dist_j)
                global_min_prototype_info[j]['min_distance'].append(batch_min_proto_dist_j[0][0])
                # global_min_prototype_info[j]['min_f_vector'].append(batch_min_fmap_patch_j)
                # global_min_prototype_info[j]['nearest_input'].append(search_batch[img_index_in_batch].cpu().numpy().transpose(1, 2, 0))

        # if push_iter % 100 == 0:
        #     print(push_iter, j)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    dir_for_saving_prototypes = proto_epoch_dir

    #################################################################################################
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform_push = transforms.Compose([
        transforms.Resize(size=(1536, 1536 // 2)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    has_pushed_img = []
    for j in range(n_prototypes):
        prototype_info_j = global_min_prototype_info[j]

        sorted_idx = np.argsort(np.array(prototype_info_j['min_distance']))
        for push_idx in sorted_idx:
            img_name = prototype_info_j['input_image_name'][push_idx]

            if img_name in has_pushed_img:
                # print('skip the smae image:', img_name)
                continue

            img_PIL = Image.open(img_name).convert('RGB')
            img_torch = transform_push(img_PIL).unsqueeze(0).cuda()
            with torch.no_grad():
                feat_backbone = prototype_network_parallel[0].module.forward_backbone(img_torch).detach()
                protoL_feat_torch, proto_dist_torch = prototype_network_parallel[1].module.push_forward(feat_backbone)

                spatial_idx = prototype_info_j['patch_spatial_idx'][push_idx]
                push_f_vector = protoL_feat_torch[:, :, spatial_idx[0], spatial_idx[1]].squeeze()

                                                                                                                                                      
            # prototype_network_parallel.module.prototype_vectors[j].data.copy_(push_f_vector.unsqueeze(1).unsqueeze(1))
            global_min_ori_img_names[j] = img_name
            has_pushed_img.append(img_name)

            # get the whole image
            original_img_j = img_torch.cpu().numpy().squeeze()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size1, original_img_size2 = original_img_j.shape[0], original_img_j.shape[1]
            proto_dist_img_j = proto_dist_torch[:, j, :, :].squeeze().cpu().numpy()
            if prototype_network_parallel[1].module.prototype_activation_function == 'log':   # use this!
                # proto_act_img_j = np.exp(-proto_dist_img_j / 128.0)
                proto_act_img_j = 1 - (proto_dist_img_j - proto_dist_img_j.min()) / (proto_dist_img_j.max() - proto_dist_img_j.min())

            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size2, original_img_size1))
            proto_bound_j, highest_index = find_high_activation_crop(upsampled_act_img_j, percentile=95)   # 95

            highest_index = np.array(np.unravel_index(np.argmax(proto_act_img_j), proto_act_img_j.shape)) * \
                            int(upsampled_act_img_j.shape[0] / proto_act_img_j.shape[0])

            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1], proto_bound_j[2]:proto_bound_j[3], :]

            # save the whole image containing the prototype as png
            original_img_j = (original_img_j - np.min(original_img_j)) / (np.max(original_img_j) - np.min(original_img_j))
            save_name = os.path.join(dir_for_saving_prototypes, str(j) + prototype_img_filename_prefix + '-original' + '.png')
            imsave_with_bbox(save_name, original_img_j, proto_bound_j[0], proto_bound_j[1], proto_bound_j[2], proto_bound_j[3], highest_index)

            rescaled_act_img_j = np.clip(upsampled_act_img_j, 0, 1)
            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]
            overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
            save_name = os.path.join(dir_for_saving_prototypes, str(j) + prototype_img_filename_prefix + '-original_with_self_act' + '.png')
            imsave_with_bbox(save_name, overlayed_original_img_j, proto_bound_j[0], proto_bound_j[1], proto_bound_j[2], proto_bound_j[3], highest_index)

            # save the prototype image (highly activated region of the whole image)
            proto_img_j = (proto_img_j - np.min(proto_img_j)) / (np.max(proto_img_j) - np.min(proto_img_j) + 0.000001)
            plt.imsave(os.path.join(dir_for_saving_prototypes, str(j) + prototype_img_filename_prefix + '.png'), proto_img_j, vmin=0.0, vmax=1.0)

            break    # The current prototype has been pushed


    #################################################################################################
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end - start))

    with open(os.path.join(proto_epoch_dir, 'global_min_ori_img_names' + ".txt"), 'w') as f:
        for i in global_min_ori_img_names:
            f.write(i + '\n')


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end, highest_index, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    cv2.circle(img_bgr_uint8, center=(highest_index[1], highest_index[0]), radius=6, color=color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float, vmin=0.0, vmax=1.0)
    # plt.imsave('fname.png', img_rgb_float, vmin=0.0, vmax=1.0)