import os
import socket

from munch import munchify

hostname = socket.gethostname()

DEFAULTS = dict()

DEFAULTS['project_name'] = 'ccraft'


DEFAULTS['data_root'] = '/Users/jan.rojc/Documents/MagCode/Data/ccraft_data/'
DEFAULTS['aux_data'] = os.path.join(DEFAULTS['data_root'], 'aux_data')
DEFAULTS['project_dir'] = '/Users/jan.rojc/Documents/MagCode/'
DEFAULTS['config_dir'] = '/Users/jan.rojc/Documents/MagCode/Models/ContourCraft/configs/'

DEFAULTS['experiment_root'] = os.path.join(DEFAULTS['data_root'], 'experiments')

DEFAULTS['CMU_root'] = '/Users/jan.rojc/Documents/MagCode/Data/AMASS/smpl/CMU/'

DEFAULTS = munchify(DEFAULTS)
