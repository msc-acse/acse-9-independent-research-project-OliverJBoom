from .deeplearning import *
from .eval_inspect import *
from .models import *
from .preprocessing import *

import pickle
from sklearn.preprocessing import MinMaxScaler

torch.nn.Module.dump_patches = True
