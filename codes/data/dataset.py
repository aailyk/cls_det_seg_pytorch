import os
import torch
import torch.utils.data as data
import data.util as util
import numpy as np
import random

class Dataset(data.Dataset):
	def __init__(self, dataset_info_path, target_size=224):
		super(Dataset, self).__init__()
		self.target_size = target_size
		self.dataset_info_path = dataset_info_path
		self.image_paths = []
		self.image_labels = []
		with open(self.dataset_info_path, 'r') as f:
			for line in f.readlines():
				image_path, image_label = line.strip().split(' ')
				self.image_paths.append(image_path)
				self.image_labels.append(int(image_label))

	def __getitem__(self, index):
		image_path = self.image_paths[index]
		label = self.image_labels[index]
		image = util.read_img(image_path, self.target_size)
		mask = util.read_mask(image_path.replace('images', 'segmentations').replace('jpg', 'png'), self.target_size)

		if image.shape[2] == 3:
			image = image[:, :, [2, 1, 0]]

			## data augmentation
			if random.random() > 1.0:
				aug = util.get_augmentor()
				angle = random.randint(-25, 25)
				image = util.rotate(image, angle)
				image = aug.augment_image(image)

			image -= np.array([123.68, 116.779, 103.939])
			image /= np.array([58.393, 57.12, 57.375])
		else:
			raise Exception('The shape of {} is not 3.'.format(image_path))
		image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()
		return {'inputs': image, 'labels': label, 'mask': mask}

	def __len__(self):
		return len(self.image_paths)
