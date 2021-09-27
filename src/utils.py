import os
import torch
import importlib
import logging
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Union


logger = logging.getLogger(__name__)

def get_args():
    """
    Input arguments used during training.
    """

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    # arg("-i", "--image_path", type=Path, help="Path to folder with training images.", required=True)
    # arg("-a", "--annotations_path", type=Path, help="Path to csv file with training annotations.", required=True)
    return parser.parse_args()


def load_rgb(image_path: Union[Path, str]) -> np.array:
    """Load RGB image from path.
    Args:
        image_path: path to image
    Returns: 3 channel array with RGB image
    """

    if Path(image_path).is_file():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    raise FileNotFoundError(f"File not found {image_path}")


def load_obj(obj_path: str, default_obj_path: str = ''):
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted,
                      including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given
                            named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = (
        obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    )
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded'
                             f'from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def select_device(device='', batch_size=None):
    """
    Select the device on which to perform the computation.
    Example:
         device = 'cpu'
         or '0' (for 1 gpu)
         or '0,1,2,3' (for multiple gpus)
    """

    cpu_request = device.lower() == 'cpu'

    # if device requested other than 'cpu'
    if device and not cpu_request:

        # set environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = device

        # check availability
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()

        # check that batch_size is compatible with device_count
        if ng > 1 and batch_size:
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("\n%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                        (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info('Using CPU')

    logger.info('')
    return torch.device('cuda:0' if cuda else 'cpu')









