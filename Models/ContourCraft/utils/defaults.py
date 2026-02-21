import os
import socket

from munch import munchify

hostname = socket.gethostname()

DEFAULTS = dict()

DEFAULTS['project_name'] = 'ccraft'


DEFAULTS['results_dir'] = '/mnt/d/ClothSim/Results/ccraft/'
DEFAULTS['data_root'] = '/mnt/d/ClothSim/ccraft_data/'
DEFAULTS['aux_data'] = os.path.join(DEFAULTS['data_root'], 'aux_data')
DEFAULTS['project_dir'] = '/home/janr/documents/MagCode/Models/ContourCraft/'
DEFAULTS['config_dir'] = '/home/janr/documents/MagCode/Models/ContourCraft/configs/'

DEFAULTS['experiment_root'] = os.path.join(DEFAULTS['data_root'], 'experiments')

DEFAULTS['CMU_root'] = '/mnt/d/ClothSim/AMASS/CMU_SeparateArms/'

DEFAULTS = munchify(DEFAULTS)
