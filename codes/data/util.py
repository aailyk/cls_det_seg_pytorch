import cv2
import numpy as np
import functools
from imgaug import augmenters as iaa 

def read_img(image_path, target_size=None):
	# img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	img = cv2.imread(image_path, 1)
	if target_size is not None:
		img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
	img = img.astype(np.float32)
	return img 

def read_mask(mask_path, target_size=224):
	mask = cv2.imread(mask_path, -1)
	if len(mask.shape) != 2:
		mask = mask[:, :, 0]
	if target_size is not None:
		mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
	mask = np.where(mask > 125, 1, 0)
	mask = mask.astype('int32')
	return mask

def rotate(image, angle, center=None, scale=1.0):
	h, w = image.shape[:2]

	if center is None:
		center = (w / 2, h / 2)

	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	return rotated

@functools.lru_cache(maxsize=1)
def get_augmentor():
	seq = iaa.Sequential([
	    # iaa.Affine(rotate=(-25, 25)), ## The Affine function raise Segmentation fault
	    iaa.AdditiveGaussianNoise(scale=(10, 60)),
	    iaa.Crop(percent=(0, 0.2))
	], random_order=True)
	
	return seq
