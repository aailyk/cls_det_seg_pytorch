import logging
import torch
import torch.utils.data

def create_dataloader(dataset, phase, **options):
	if phase == 'train':
		for key in ['batch_size', 'num_workers']:
			if key not in options:
				raise KeyError('The key [{}] is not in DataLoader options.'.format(key))
		return torch.utils.data.DataLoader(dataset, batch_size=options['batch_size'], shuffle=True, 
									num_workers=options['num_workers'], sampler=None, drop_last=True,
									pin_memory=False)
	elif phase == 'test':
		return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
									num_workers=1, sampler=None, drop_last=False, 
									pin_memory=False)
	else:
		raise NotImplementedError('The dataset phases [{}] not in [train, test]'.format(phase))

def create_dataset(data_info_path, target_size=224):
	from data.dataset import Dataset 
	dataset = Dataset(data_info_path, target_size)
	return dataset
