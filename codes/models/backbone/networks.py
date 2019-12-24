import torch 
import torchvision
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torch.utils.model_zoo as model_zoo
import math
import logging

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

logger = logging.getLogger('base')

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Basic Block for model whose layers less 50 layers. (18, 34)
class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.stride = stride
		self.downsample = downsample

	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		if self.downsample is not None:
			residual = self.downsample(residual)
		x = residual + x
		x = self.relu(x)
		return x

# Bottleneck for model whose layers equal and greater 50 layers. (50, 101, 152)
class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)

		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)

		if self.downsample is not None:
			residual = self.downsample(residual)
		x = x + residual
		x = self.relu(x)

		return x

class Backbone(nn.Module):
	def __init__(self, block, layers):
		super(Backbone, self).__init__()
		self.in_planes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.in_planes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.in_planes, planes, stride, downsample))
		self.in_planes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_planes, planes))

		return nn.Sequential(*layers)

class ResNetCls(Backbone):
	def __init__(self, block, layers, num_classes=1000):
		super(ResNetCls, self).__init__(block, layers)
		self.num_classes = num_classes
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, self.num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

class ResNetSeg(Backbone):
	def __init__(self, block, layers):
		super(ResNetSeg, self).__init__(block, layers)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		low_level_feat = x
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x, low_level_feat		

def resnet18(pretrained=False, **kwargs):
	"""Construct a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet.
	"""
	assert 'task_type' in kwargs, "The model task is not confirmed."
	if kwargs['task_type'] == 'cls':
		assert 'num_classes' in kwargs, "There is not exists num_classes."
		model = ResNetCls(BasicBlock, [2, 2, 2, 2], kwargs['num_classes'])
	elif kwargs['task_type'] == 'seg':
		model = ResNetSeg(BasicBlock, [2, 2, 2, 2])
	else:
		raise Exception("The task [{}] has not implemented.".format(kwargs['task_type']))

	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
		model_dict = torch.load('/gpfs/share/home/1701214057/CUB2011/codes/models/model_zoo/resnet18_params.pth')
		if kwargs['task_type'] != 'cls' or kwargs['num_classes'] != 1000:
			model_dict.pop('fc.weight')
			model_dict.pop('fc.bias')
			logger.info('The network fc.weight and fc.bias has removed.')
		model.load_state_dict(model_dict, strict=False)
	return model 

def resnet34(pretrained=False, **kwargs):
	"""Construct a ResNet-34 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet.
	"""
	assert 'task_type' in kwargs, "The model task is not confirmed."
	if kwargs['task_type'] == 'cls':
		assert 'num_classes' in kwargs, "There is not exists num_classes."
		model = ResNetCls(BasicBlock, [3, 4, 6, 3], kwargs['num_classes'])
	elif kwargs['task_type'] == 'seg':
		model = ResNetSeg(BasicBlock, [3, 4, 6, 3])
	else:
		raise Exception("The task [{}] has not implemented.".format(kwargs['task_type']))
	
	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
		model_dict = torch.load('/gpfs/share/home/1701214057/CUB2011/codes/models/model_zoo/resnet34_params.pth')
		if kwargs['task_type'] != 'cls' or kwargs['num_classes'] != 1000:
			model_dict.pop('fc.weight')
			model_dict.pop('fc.bias')
			logger.info('The network fc.weight and fc.bias has removed.')
		model.load_state_dict(model_dict, strict=False)
	return model 

def resnet50(pretrained=False, **kwargs):
	"""Construct a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet.
	"""
	assert 'task_type' in kwargs, "The model task is not confirmed."
	if kwargs['task_type'] == 'cls':
		assert 'num_classes' in kwargs, "There is not exists num_classes."
		model = ResNetCls(Bottleneck, [3, 4, 6, 3], kwargs['num_classes'])
	elif kwargs['task_type'] == 'seg':
		model = ResNetSeg(Bottleneck, [3, 4, 6, 3])
	else:
		raise Exception("The task [{}] has not implemented.".format(kwargs['task_type']))

	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
		model_dict = torch.load('/gpfs/share/home/1701214057/CUB2011/codes/models/model_zoo/resnet50_params.pth')
		if kwargs['task_type'] != 'cls' or kwargs['num_classes'] != 1000:
			model_dict.pop('fc.weight')
			model_dict.pop('fc.bias')
			logger.info('The network fc.weight and fc.bias has removed.')
		model.load_state_dict(model_dict, strict=False)
	return model 

def resnet101(pretrained=False, **kwargs):
	"""Construct a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet.
	"""
	assert 'task_type' in kwargs, "The model task is not confirmed."
	if kwargs['task_type'] == 'cls':
		assert 'num_classes' in kwargs, "There is not exists num_classes."
		model = ResNetCls(Bottleneck, [3, 4, 23, 3], kwargs['num_classes'])
	elif kwargs['task_type'] == 'seg':
		model = ResNetSeg(Bottleneck, [3, 4, 23, 3])
	else:
		raise Exception("The task [{}] has not implemented.".format(kwargs['task_type']))

	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
		model_dict = torch.load('/gpfs/share/home/1701214057/CUB2011/codes/models/model_zoo/resnet101_params.pth')
		if kwargs['task_type'] != 'cls' or kwargs['num_classes'] != 1000:
			model_dict.pop('fc.weight')
			model_dict.pop('fc.bias')
			logger.info('The network fc.weight and fc.bias has removed.')
		model.load_state_dict(model_dict, strict=False)
	return model 

def resnet152(pretrained=False, **kwargs):
	"""Construct a ResNet-152 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet.
	"""
	assert 'task_type' in kwargs, "The model task is not confirmed."
	if kwargs['task_type'] == 'cls':
		assert 'num_classes' in kwargs, "There is not exists num_classes."
		model = ResNetCls(Bottleneck, [3, 8, 36, 3], kwargs['num_classes'])
	elif kwargs['task_type'] == 'seg':
		model = ResNetSeg(Bottleneck, [3, 8, 36, 3])
	else:
		raise Exception("The task [{}] has not implemented.".format(kwargs['task_type']))
	
	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
		model_dict = torch.load('/gpfs/share/home/1701214057/CUB2011/codes/models/model_zoo/resnet152_params.pth')
		if kwargs['task_type'] != 'cls' or kwargs['num_classes'] != 1000:
			model_dict.pop('fc.weight')
			model_dict.pop('fc.bias')
			logger.info('The network fc.weight and fc.bias has removed.')
		model.load_state_dict(model_dict, strict=False)
	return model 
