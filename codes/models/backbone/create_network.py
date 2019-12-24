import logging
import models.backbone.networks as networks
logger = logging.getLogger('base')

def define_net(which_model, task_type, pretrained=False, num_classes=1000):
	if which_model == 'resnet-18':
		net = networks.resnet18(pretrained=pretrained, task_type=task_type, num_classes=num_classes)
	elif which_model == 'resnet-34':
		net = networks.resnet34(pretrained=pretrained, task_type=task_type, num_classes=num_classes)
	elif which_model == 'resnet-50':
		net = networks.resnet50(pretrained=pretrained, task_type=task_type, num_classes=num_classes)
	elif which_model == 'resnet-101':
		net = networks.resnet101(pretrained=pretrained, task_type=task_type, num_classes=num_classes)
	elif which_model == 'resnet-152':
		net = networks.resnet152(pretrained=pretrained, task_type=task_type, num_classes=num_classes)
	else:
		raise NotImplementedError('The model [{}] is not implemented.'.format(which_model))
	logger.info('The network [{}] has created.'.format(which_model))
	return net
