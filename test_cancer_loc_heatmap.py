import os
import shutil
import torch
import torch.utils.data
import numpy as np
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import argparse
import re
import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from models import model_protopnet
from models import model_globalnet
from utils_model.helpers import makedir


parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
args = parser.parse_args()


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # "1, 2"
# print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])



def find_high_activation_crop(activation_map, percentile=95):
    rescaled_act_img_j = activation_map.copy()
    threshold = np.percentile(rescaled_act_img_j, percentile)
    mask = np.ones(rescaled_act_img_j.shape)
    mask[rescaled_act_img_j < threshold] = 0

    highest_index = list(np.unravel_index(np.argmax(rescaled_act_img_j), rescaled_act_img_j.shape))

    # only keep the largest activated region
    n_labels, img_labeled, lab_stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    for label in range(1, n_labels):
        temp = img_labeled == label
        if temp[highest_index[0], highest_index[1]] == False:
            mask[temp] = 0
    return mask


def FillHole(mask):
    mask = mask.astype(np.uint8)*255
    im_floodfill = mask.copy()
    maskS = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(im_floodfill, maskS, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = mask | im_floodfill_inv
    im_out[im_out == 255] = 1
    return im_out



from utils_model.settings import base_architecture, img_size, prototype_shape, num_classes, \
     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)


# construct the model
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

class_specific = True


protopnet.load_state_dict(torch.load("demo_ckpt/GlobalNet_retrain8_S_0.8718.pth"))
globalnet.load_state_dict(torch.load("demo_ckpt/GlobalNet_retrain8_T_0.8718.pth"))


protopnet = torch.nn.DataParallel(protopnet)
globalnet = torch.nn.DataParallel(globalnet)

protopnet.eval()
globalnet.eval()


test_files = sorted(glob.glob('./demo_test_imgs/*.png'))
test_files = [f for f in test_files if '_annotation_' not in f]

transform = transforms.Compose([
            transforms.Resize(size=(1536, 1536//2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


save_dir = './demo_test_heatmaps'
makedir(save_dir)



probabilities = []
test_targets = []
for i, img_name in enumerate(test_files):

    img_ori = Image.open(img_name)
    img_ori = img_ori.convert('RGB')
    img = transform(img_ori)
    label = float(1.0)

    img_np = img.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    input = img.unsqueeze(0).cuda()
    target = torch.tensor(label).cuda()
    test_targets.append(label)

    # original_img_size1, original_img_size2 = img_np.shape[0], img_np.shape[1]
    original_img_size1, original_img_size2 = img_ori.size[1], img_ori.size[0]

    grad_req = torch.no_grad()
    with grad_req:

        feat_backbone = globalnet.module.forward_backbone(input).detach()
        explain_logits, min_distances, similarity_maps = protopnet(feat_backbone)
        global_logits = globalnet(feat_backbone)

        prob_explain = F.softmax(explain_logits, dim=1).cpu().numpy()
        prob_global = F.softmax(global_logits, dim=1).cpu().numpy()
        prob_ensemble = (prob_explain + prob_global) * 0.5
        prob = prob_ensemble

        if prob[0][1] > 0.02:   # threshold
            predicted = 1.0
        else:
            predicted = 0.0
        probabilities.append(prob)

        print(i, prob[0][1], img_name)

        num_proto_per_class = similarity_maps.shape[1]//2
        most_simi_index = torch.argmin(min_distances[0][num_proto_per_class:]) + num_proto_per_class   # only show the heatmap with the top-1 similar prototype

        proto_act_img_j = similarity_maps[:, most_simi_index].squeeze().detach().cpu().numpy()
        upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size2, original_img_size1), interpolation=cv2.INTER_CUBIC)

        rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
        rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]

        original_img_j = np.array(img_ori).astype(float) / 255

        overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap

        highest_index = list(np.unravel_index(np.argmax(rescaled_act_img_j), rescaled_act_img_j.shape))
        img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * overlayed_original_img_j), cv2.COLOR_RGB2BGR)
        cv2.circle(img_bgr_uint8, center=(highest_index[1], highest_index[0]), radius=6, color=(0, 255, 255), thickness=2)
        img_rgb_uint8 = img_bgr_uint8[..., ::-1]
        img_rgb_float = np.float32(img_rgb_uint8) / 255

        plt.imsave(os.path.join(save_dir, os.path.basename(img_name).replace('.png', '_overlap_' + str(prob[0][1]).split('.')[1][0:3] + '.jpg')), img_rgb_float, vmin=0.0, vmax=1.0)

    # break


