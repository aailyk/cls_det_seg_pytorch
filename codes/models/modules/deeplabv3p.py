import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.backbone import create_network
from .aspp import ASPP 

class Decoder(nn.Module):
	def __init__(self, backbone, num_classes):
		super(Decoder, self).__init__()
		if 'resnet' in backbone:
			if backbone in ['resnet-18', 'resnet-34']:
				low_level_inplaces = 64
			elif backbone in ['resnet-50', 'resnet-101', 'resnet-152']:
				low_level_inplaces = 256
			else:
				raise NotImplementedError('The network [{}] is not implemented.'.format(backbone))
		else:
			raise NotImplementError('The backbone is not implemented.')

		self.conv = nn.Conv2d(low_level_inplaces, 48, 1, stride=1, bias=False)
		self.bn = nn.BatchNorm2d(48)
		self.relu = nn.ReLU()

		self.last_conv = nn.Sequential(nn.Conv2d(304, 256, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Conv2d(256, num_classes, 1))
		self._init_weight()

	def forward(self, x, low_level_feat):
		low_level_feat = self.conv(low_level_feat)
		low_level_feat = self.bn(low_level_feat)
		low_level_feat = self.relu(low_level_feat)

		x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x, low_level_feat), dim=1)
		x = self.last_conv(x)

		return x

	def _init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

class DeepLabv3p(nn.Module):
	def __init__(self, which_model, output_stride, num_classes=21, pretrained=False, freezze_bn=False):
		super(DeepLabv3p, self).__init__()
		self.backbone = create_network.define_net(which_model, task_type='seg')
		self.aspp = ASPP(which_model, output_stride)
		self.decoder = Decoder(which_model, num_classes)

		self.freezze_bn = freezze_bn

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()

	def forward(self, input_):
		x, low_level_feat = self.backbone(input_)
		x = self.aspp(x)
		x = self.decoder(x, low_level_feat)
		x = F.interpolate(x, size=input_.size()[2:], mode='bilinear', align_corners=True)

		return x
		