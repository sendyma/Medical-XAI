import os
import torch
import numpy as np
import cv2


def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)  # 2000*2000


def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + '\n')


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0

    # from PIL import Image
    # image_output = Image.fromarray(mask * 255)
    # image_output.convert('RGB').save("image_output.jpg")

    highest_index = list(np.unravel_index(np.argmax(activation_map), activation_map.shape))

    n_labels, img_labeled, lab_stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    for label in range(1, n_labels):
        temp = img_labeled == label
        if temp[highest_index[0], highest_index[1]] == False:
            mask[temp] = 0

    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break

    if upper_x <= lower_x:
        upper_x = lower_x

    if upper_y <= lower_y:
        upper_y = lower_y

    return (lower_y, upper_y + 1, lower_x, upper_x + 1), highest_index
