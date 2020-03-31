import numpy as np
import cv2


def convert_to_binary(a):
    val = np.unique(a)[1]
    above_threshold = a >= val
    under_threshold = a < val
    a[above_threshold] = 1
    a[under_threshold] = 0
    return a.astype(np.uint8).reshape(int(np.sqrt(len(a))), int(np.sqrt(len(a))))


def get_mask(img):
    # if image is all black: ignore
    if not img.any():
        return img.astype(np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,label,center = cv2.kmeans(img.reshape(np.prod(img.shape), 1), 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return convert_to_binary(center[label])


def threshold(x):
    mask = np.copy(x)
    mask = mask.ravel()
    above_threshold = mask > x.mean()+x.std()
    under_threshold = mask <= x.mean() + x.std()
    mask[above_threshold] = 1
    mask[under_threshold] = 0
    return mask.astype(np.uint8).reshape((x.shape))


# intersection over union (evaluation function)
def iou(a,b):
    i = a&b
    u = a|b
    if u.sum() == 0:
        return 0
    return i.sum()/u.sum()
