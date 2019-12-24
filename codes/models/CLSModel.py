import os
import logging
from collections import OrderedDict
from models.backbone import create_network
import torch
import torch.nn as nn


class CLSModel:
	def __init__(self, **options):
		for key in ['which_model', 'pretrained', 'device', 'learning_rate']:
			if key not in options:
				raise KeyError('The key [] not is not in options'.format(key))
		num_classes = options['num_classes'] if 'num_classes' in options else 1000
		self.options = options 
		self.netG = create_network.define_net(self.options['which_model'], task_type='cls', pretrained=self.options['pretrained'], num_classes=num_classes).to(self.options['device'])
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
		self.log_dict['correct'] = 0

	def feed_data(self, data):
		self.inputs = data['inputs'].to(self.options['device'])
		self.labels = data['labels'].to(self.options['device'])

	def adjust_learning_rate(self, target_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = target_lr

	def optimize_parameters(self):
		self.optimizer.zero_grad()

		self.outputs = self.netG(self.inputs)
		loss = self.loss_func(self.outputs, self.labels)
		pred = torch.max(self.outputs, 1)[1]
		correct = (pred == self.labels).sum()

		loss.backward()
		self.optimizer.step()

		self.log_dict['loss'] += loss.item()
		self.log_dict['correct'] += correct

	def test(self):
		self.netG.eval()

		with torch.no_grad():
			self.outputs = self.netG(self.inputs)
			loss = self.loss_func(self.outputs, self.labels)
			pred = torch.max(self.outputs, 1)[1]
			correct = (pred == self.labels).sum()

		self.log_dict['loss'] += loss.item()
		self.log_dict['correct'] += correct

		self.netG.train()

	def get_current_log(self):
		return self.log_dict

	def save_network(self, network_label, iter_label):
		assert 'model_save_path' in self.options, "The key [model_save_path] is not in options."
		save_filename = '{}_{}.pth'.format(iter_label, network_label)
		
		if not os.path.exists(self.options['model_save_path']):
			os.makedirs(self.options['model_save_path'])

		save_path = os.path.join(self.options['model_save_path'], save_filename)
		network = self.netG
		if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
			network = network.module
		state_dict = network.state_dict()
		for key, param in state_dict.items():
			state[key] = param.cpu()
		torch.save(state_dict, save_path)
