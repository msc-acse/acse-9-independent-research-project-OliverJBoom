import numpy as np
import random
import torch


def set_seed(seed, device='cpu'):
    """Use this to set all the random seeds to a fixed value
    and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  
        torch.backends.cudnn.enabled = False

    return True



def model_save(model, name, val_score, path="Models/"):
    """Saving function to keep track of models"""
    val = str(val_score)[:5].replace(".", "_")
    print("Saving model:", path + name + '_' + val + '.pth')
    torch.save(model, path + name + '_' + val + '.pth')
    return


def model_load(model_name, path="Models/"):
    """Loading function for models from google drive"""
    model = torch.load(path + model_name + '.pth')
    return model