import logging
logger = logging.getLogger('base')

def create_model(**options):
	if options['task_type'] == 'cls':
		from .CLSModel import CLSModel as M 
		m = M(**options)
		logger.info('The model [{:s}] is created.'.format(m.__class__.__name__))
	elif options['task_type'] == 'seg':
		from .SEGModel import SEGModel as M 
		m = M(**options)
		logger.info('The model [{:s}] is created.'.format(m.__class__.__name__))
	else:
		NotImplementError('The task type [{}] is not implemented.'.format(options['task_type']))
	return m