import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

class _ASPPModule(nn.Module):
	def __init__(self, in_planes, planes, kernel_size, padding, dilation):
		super(_ASPPModule, self).__init__()
		self.atrous_conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)

		self.bn = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.atrous_conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

class ASPP(nn.Module):
	def __init__(self, backbone, output_stride):
		super(ASPP, self).__init__()

		if 'resnet' in backbone:
			if backbone in ['resnet-18', 'resnet-34']:
				in_planes = 512
			elif backbone in ['resnet-50', 'resnet-101', 'resnet-152']:
				in_planes = 2048
			else:
				raise NotImplementedError('The network [{}] is not implemented.'.format(backbone))
		else:
			raise NotImplementedError('The backbone is not implemented.')

		if output_stride == 16:
			dilations = [1, 6, 12, 18]
		elif output_stride == 8:
			dilations = [1, 12, 24, 36]
		else:
			raise NotImplementedError('The output_stride is not in [16, 8].')

		self.aspp1 = _ASPPModule(in_planes, 256, 1, padding=0, dilation=dilations[0])
		self.aspp2 = _ASPPModule(in_planes, 256, 3, padding=dilations[1], dilation=dilations[1])
		self.aspp3 = _ASPPModule(in_planes, 256, 3, padding=dilations[2], dilation=dilations[2])
		self.aspp4 = _ASPPModule(in_planes, 256, 3, padding=dilations[3], dilation=dilations[3])

		self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
											nn.Conv2d(in_planes, 256, 1, stride=1, bias=False),
											nn.BatchNorm2d(256),
											nn.ReLU())
		self.conv = nn.Conv2d(256 * 5, 256, kernel_size=1, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU(inplace=True)
		self.dropout = nn.Dropout(0.5)
		
		self._init_weight()

	def forward(self, x): 
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		x5 = self.global_avg_pool(x)
		x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x1, x2, x3, x4, x5), dim=1)

		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		x = self.dropout(x)

		return x

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
