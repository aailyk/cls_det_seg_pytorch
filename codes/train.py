import os
import logging
import torch
from config import Config as config
from data import create_dataloader, create_dataset
from models import create_model
import random
import numpy as np 

def setup_logger(logger_name, phase, level=logging.INFO, tofile=False):
	if tofile:
		loger_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), '{}.log'.format(phase))
		logger = logging.getLogger(logger_name)
		logger.setLevel(level)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		fh = logging.FileHandler(loger_file, mode='w')
		fh.setFormatter(formatter)
		logger.addHandler(fh)
	else:
		logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def set_random_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def main():
	import argparse

	parser = argparse.ArgumentParser(description='The args for training the ResNet on CUB2011 Datasets.')
	parser.add_argument('--which-model', type=str,
						required=True, 
						choices=['resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152']
						help='the classifier network')
	parser.add_argument('--task-type', type=str,
						default='cls', 
						choices=['cls', 'seg', 'det'],
						help='The model task, such as [cls, seg, det].')
	parser.add_argument('--pretrained', 
						action='store_true',
						help='To use the pretrained model on ImageNet.') 
	parser.add_argument('--num-classes', type=int,
						default=1000,
						help='The number of classes in the classification dataset.')
	parser.add_argument('--num-epochs', type=int, 
						default=100, 
						help='the max number of epochs') 
	parser.add_argument('--batch-size', type=int, 
						default=32, 
						help='the dataset batch size')
	parser.add_argument('--num-workers', type=int,
						default=6,
						help='The number of workers for loading dataset.')
	parser.add_argument('--learning-rate', '-lr', dest='lr', 
						type=float, 
						default=1.0e-2, 
						help='The learning rate for training networks.')
	parser.add_argument('--weight-decay', '-wd', dest='wd', 
						type=float, 
						default=1.0e-4, 
						help='the weight decay in network')
	parser.add_argument('--gpu', type=int, 
						default=0, 
						help='To assign the gpu to train the network.') 
	parser.add_argument('--seed', type=int,
						default=10007,
						help='the random seed')
	parser.add_argument('--lr-decay-epoch', type=int, 
						default=30, 
						help='Per epoch to decay the learning rate') 
	parser.add_argument('--lr-decay-rate', type=float, 
						default=0.1, 
						help='Learning rate decay rate') 
	parser.add_argument('--model-store-path', type=str, 
						default='/gpfs/share/home/1701214057/CUB2011/experiments/models/cls', 
						help='the classification model store path') 

	args = parser.parse_args()

	setup_logger('base', 'train')
	logger = logging.getLogger('base')

	set_random_seed(args.seed)

	train_dataset = create_dataset(config.train_data_info_path)
	train_dataloader = create_dataloader(train_dataset, phase='train', batch_size=args.batch_size, num_workers=args.num_workers)

	test_dataset = create_dataset(config.test_data_info_path)
	test_dataloader = create_dataloader(test_dataset, phase='test')

	model = create_model(which_model=args.which_model, 
						task_type=args.task_type, 
						pretrained=args.pretrained, 
						num_classes=args.num_classes, 
						device='cuda:{}'.format(args.gpu), 
						learning_rate=args.lr, 
						weight_decay=args.wd)
	
	for epoch in range(args.num_epochs):

		num_examples = 0
		model.clear_log()
		for data_iter in train_dataloader:
			model.feed_data(data_iter)
			model.optimize_parameters()
			num_examples += args.batch_size
		dict_log = model.get_current_log()
		train_loss = dict_log['loss'] * 1.0 * args.batch_size / num_examples
		train_accuracy = dict_log['correct'] * 1.0 / num_examples

		if (epoch + 1) % args.lr_decay_epoch == 0:
			target_lr = args.lr * (0.1 ** ((epoch + 1) // args.lr_decay_epoch))
			model.adjust_learning_rate(target_lr)
			logger.info('Updated the learning rate, the current learning rate: {}'.format(target_lr))

		num_examples = 0
		model.clear_log()
		for data_iter in test_dataloader:
			model.feed_data(data_iter)
			model.test()
			num_examples += 1
		dict_log = model.get_current_log()
		test_loss = dict_log['loss'] * 1.0 / num_examples
		test_accuracy = dict_log['correct'] * 1.0 / num_examples

		logger.info('Epoch: {0} / {1}, train loss: {2:.3f}, train accuracy: {3:.3f}%, test loss: {4:.3f}, test accuracy: {5:.3f}%'.format(epoch, args.num_epochs, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100))
		
		if epoch % 5 == 0:
			model.save_network('{}'.format(args.which_model), epoch)

if __name__ == '__main__':
	main()
