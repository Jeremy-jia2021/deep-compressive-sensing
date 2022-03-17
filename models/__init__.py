import logging
from .SRGAN_model import SRGANModel as M
logger = logging.getLogger('base')


def create_model(opt):
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
