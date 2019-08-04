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
    val = str(val_score)[:6].replace(".", "_")
    print("Saving model:", path + name + '_' + val + '.pth')
    torch.save(model, path + name + '_' + val + '.pth')
    return


def model_load(model_name, path="Models/"):
    """Loading function for models from google drive"""
    model = torch.load(path + model_name + '.pth')
    return model



class early_stopping:
  """
  Counter to implement early stopping
  If validation accuracy has not relative improved below
  a relative tolerance set by the user than it breaks the 
  training
  If rel_tol is set to 0 it becomes a common counter
  """
  def __init__(self, patience, rel_tol, verbose=True):
    
    self.patience = patience
    self.rel_tol = rel_tol
    self.verbose = verbose
    self.best_score = np.inf
    self.counter = 0
    self.stop = False

  
  def __call__(self, score):
    
    # If the score is under the required relative tolerance
    # increase the counter is incremented
    print(self.best_score * (1 - self.rel_tol))
    if score > self.best_score * (1 - self.rel_tol):
        self.counter += 1
    else:
        self.counter = 0
        
    if score < self.best_score:
        self.best_score = score
      
    if self.counter >= self.patience:
        self.stop = True    
      
    if self.verbose:
        print("Count:", self.counter)