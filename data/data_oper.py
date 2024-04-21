import numpy as np
from collections import Counter
import torchvision.transforms.functional as TF
from PIL import Image


class GammaCorrection(object):
    def __call__(self, image):
        data = Counter(np.ravel(image))
        if data.most_common()[1][0] > 85:
            imgTmp = TF.adjust_gamma(image, 4.0, gain=4)
        elif data.most_common()[1][0] > 70:
            imgTmp = TF.adjust_gamma(image, 3.0, gain=3)
        elif data.most_common()[1][0] > 35:
            imgTmp = TF.adjust_gamma(image, 2.0, gain=2)
        else:
            imgTmp = image
        return imgTmp


def SquarePad(image, df_idx, ratio_hw):
    w, h = image.size
    place = (0, 0)
    if h / w > ratio_hw:
        if df_idx.ImageLaterality == 'R':
            place = (int(h / ratio_hw) - w, 0)
        result = Image.new(image.mode, (int(h // ratio_hw), h), 0)
    elif h / w == ratio_hw:
        return image
    else:
        result = Image.new(image.mode, (w, int(w * ratio_hw)), 0)
    result.paste(image, place)
    return result
