# This file generates binary mask for breast region and remove texts and the background area containing no ROI in the mammogram

import cv2, os, csv, math
import numpy as np


def select_largest_obj(img_bin, lab_val=255, fill_holes=True,
                       smooth_boundary=True, kernel_size=15,
                       org=None, low_th=None):
    tmp_list = []

    n_labels, img_labeled, lab_stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8, ltype=cv2.CV_32S)
    _, org_tmp = cv2.threshold(org.copy(), low_th, maxval=255, type=cv2.THRESH_BINARY)

    topk = np.argsort(lab_stats[:, 4])[::-1][:4]

    [tmp_list.append(np.sum(org_tmp[img_labeled == i])) for i in topk]
    # [tmp_list.append(np.sum(org_tmp[img_labeled == i])) for i in range(lab_stats.shape[0])]
    largest_obj_lab = topk[np.argmax(tmp_list)]
    # largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1

    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)

    largest_mask[img_labeled == largest_obj_lab] = 255
    largest_mask[img_labeled != largest_obj_lab] = 0
    # import pdb; pdb.set_trace()

    if fill_holes:
        largest_mask = cv2.blur(largest_mask.copy(), (20, 20))
        largest_mask[largest_mask != 0] = 255
        largest_mask[largest_mask != 255] = 0

    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)

    del tmp_list
    return largest_mask


def max_pix_val(dtype):
    if dtype == np.dtype('uint8'):
        maxval = 2**8 - 1
    elif dtype == np.dtype('uint16'):
        maxval = 2**16 - 1
    else:
        raise Exception('Unknown dtype found in input image array')
    return maxval


def suppress_artifacts(img, global_threshold=.05, fill_holes=True,
                       smooth_boundary=True, kernel_size=15):
    maxval = max_pix_val(img.dtype)
    if global_threshold < 1.:
        low_th = int(img.max()*global_threshold)
    else:
        low_th = int(global_threshold)

    _, img_bin = cv2.threshold(img, 5, maxval=maxval, type=cv2.THRESH_BINARY)
    breast_mask = select_largest_obj(img_bin, lab_val=maxval,
                                     fill_holes=fill_holes,
                                     smooth_boundary=smooth_boundary,
                                     kernel_size=kernel_size,
                                     org=img, low_th=low_th)
    img_suppr = cv2.bitwise_and(img, breast_mask)
    return img_suppr, breast_mask


def crop_max_bg(img, breast_mask):
    contours, _ = cv2.findContours(breast_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [cv2.contourArea(cont) for cont in contours]
    idx = np.argmax(cont_areas)
    x, y, w, h = cv2.boundingRect(contours[idx])

    img_breast_only = img[y:y + h, x:x + w]
    img_breast_only = img_breast_only[25:-25, :]
    # img_breast_only = img_breast_only[:, 25:-25]

    # gt_annotation = gt_mask[y:y + h, x:x + w]
    # gt_annotation = gt_annotation[25:-25, :]
    # gt_annotation = gt_annotation[:, 25:-25]

    return img_breast_only#, gt_annotation