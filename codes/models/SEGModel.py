import torch
import torch.nn as nn
import os
from collections import OrderedDict
import logging
from models.modules.deeplabv3p import DeepLabv3p

class SEGModel:
	def __init__(self, **options):
		for key in ['which_model', 'pretrained', 'device', 'learning_rate']:
			if key not in options:
				raise KeyError('The key [] not is not in options'.format(key))
		self.options = options
		self.netG = DeepLabv3p(which_model=options['which_model'], output_stride=16, num_classes=options['num_classes'], pretrained=options['pretrained']).to(options['device'])
		self.netG.train()
		optim_params = []
		for k, v in self.netG.named_parameters():
			if v.requires_grad:
				optim_params.append(v)
		wd = options['weight_decay'] if 'weight_decay' in options else 0
		# self.optimizer = torch.optim.SGD(optim_params, lr=options['learning_rate'], weight_decay=wd)
		self.optimizer = torch.optim.SGD(optim_params, lr=options['learning_rate'], weight_decay=wd, momentum=0.9)
		# self.optimizer = torch.optim.Adam(optim_params, lr=options['learning_rate'], weight_decay=wd)
		self.loss_func = nn.CrossEntropyLoss()
		self.log_dict = OrderedDict()
		self.clear_log()

	def clear_log(self):
		self.log_dict['loss'] = 0

	def feed_data(self, data):
		self.inputs = data['inputs'].to(self.options['device'])
		self.labels = data['mask'].to(device=self.options['device'], dtype=torch.long)

	def adjust_learning_rate(self, target_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = target_lr

	def optimize_parameters(self):
		self.optimizer.zero_grad()

		self.outputs = self.netG(self.inputs)
		loss = self.loss_func(self.outputs, self.labels)
		loss.backward()
		self.optimizer.step()
		self.log_dict['loss'] += loss.item()

	def test(self):
		self.netG.eval()

		with torch.no_grad():
			self.outputs = self.netG(self.inputs)
			loss = self.loss_func(self.outputs, self.labels)

		self.log_dict['loss'] += loss.item()
		self.netG.train()

	def get_current_log(self):
		return self.log_dict

	def save_network(self, network_label, epoch_label):
		assert 'model_store_path' in self.options, "The key [model_store_path] is not in options."
		save_filename = '{}_{}.pth'.format(epoch_label, network_label)
		
		if not os.path.exists(self.options['model_store_path']):
			os.makedirs(self.options['model_store_path'])

		save_path = os.path.join(self.options['model_store_path'], save_filename)
		network = self.netG
		if isinstance(network, nn.parallel.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
			network = network.module
		state_dict = network.state_dict()
		for key, param in state_dict.items():
			state_dict[key] = param.cpu()
		torch.save(state_dict, save_path)

	def load_network(self, load_path, network, strict=True):
		if isinstance(network, nn.parallel.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
			network = network.module 
		load_net = torch.load(load_path)
		load_net_clean = OrderedDict()
		for k, v in load_net.items():
			if k.startswith('module.'):
				load_net_clean[k[7:]] = v
			else:
				load_net_clean[k] = v
		network.load_state_dict(load_net_clean, strict=strict)
