"""
This contains the initialisation file of Foresight, allowing the module
to be hosted on Pypi

Author: Oliver Boom
Github Alias: OliverJBoom
"""

import pickle
from sklearn.preprocessing import MinMaxScaler

from .deeplearning import *
from .eval_inspect import *
from .models import *
from .preprocessing import *


torch.nn.Module.dump_patches = True
