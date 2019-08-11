from deeplearning import model_load, DeepLearning
from eval_inspect import mean_directional_accuracy, mean_directional_accuracy_log_ret
from models import LSTM
import numpy as np
from numpy.random import rand
import pytest
from preprocessing import slice_series, universe_select
import torch 
from torch.optim import Adam

# For different LSTM class iterations the source code is saved 
# in the pth file. dump patches stops pytorch from trying to 
# write to file the old LSTM iterations. 
torch.nn.Module.dump_patches = True
    
"""This module contains functions relating to the testing of the Forecasting
Package"""


def test_import_data():
    """Checks that the data is loaded into a dictionary and that
    it is not empty"""
    path = "Data/Commodity_Data/"
    universe_dict = universe_select(path, "Cu")
    assert (type(universe_dict) is dict)
    assert (bool(universe_dict) is True)

    
def test_load_pth():
    """"Testing the model load functionality"""
    model_name = "Univariate_SS_H8_F5"
    device = 'cpu'
    path = "Results/Pths/Univariate/Hidden_Tests/"
    model = model_load(model_name=model_name, 
                       device=device, 
                       path=path)

    assert(type(model)==LSTM)
        
        
def test_train_val_test():
    """"Checking that the deeplearning class splits
    the data correctly"""
    model_name = "Univariate_SS_H8_F5"
    device = 'cpu'
    path = "Results/Pths/Univariate/Hidden_Tests/"
    
    model = model_load(model_name=model_name, 
                       device=device, 
                       path=path)
    
    data_X = rand(100, 20, 5)
    data_y = rand(100, 5)
    
    learning = DeepLearning(model=model, 
                        data_X=data_X, 
                        data_y=data_y,
                        optimiser=Adam(model.parameters()))
    
    learning.train_val_test()
    
    assert(list(learning.X_train.shape) == [60, 20, 5])
    assert(list(learning.X_val.size()) == [20, 20, 5])
    assert(list(learning.X_test.shape) == [20, 20, 5])
    
    
def test_slice_series():
    """Check that a series is correctly sliced into windows"""
    data_X = rand(100, 5)
    data_y = rand(100, 5)
    series_length = 10
    
    X, y = slice_series(data_X, data_y, series_length, dataset_pct=1.0)
    
    assert(list(X.shape)==[90, 10, 5])
    assert(list(y.shape)==[90, 5])
    
    
def test_mda():
    y_true = np.array([2, 4, 2, 4]).reshape(4, 1)
    y_pred = np.array([1, 3, 1, 3]).reshape(4, 1)
    
    # 3 directional decisions made
    # All 3 are correct
    mda_1 = mean_directional_accuracy(y_true, y_pred)
    assert (mda_1 == 1.0)
    
    # 5 directional decisions made
    # 3/5 correct
    y_true = np.array([2, 4, 2, 4, 2, 2]).reshape(6, 1)
    y_pred = np.array([1, 3, 1, -1, 2, 2]).reshape(6, 1)
    mda_2 = mean_directional_accuracy(y_true, y_pred)
    assert (mda_2 == 0.6)

    
def test_log_mda():
    y_true = np.array([1, -4, 2, -4, 5]).reshape(5, 1)
    y_pred = np.array([-1, -5, -5, -2, -4]).reshape(5, 1)

    # 5 directional decisions, 2 are correct
    mda = mean_directional_accuracy_log_ret(y_true, y_pred)
    assert (mda == 0.4)
    
    
def test_evaluate():
    data_X = rand(500, 250, 1)
    data_y = rand(500, 1)
    device = 'cpu'
    
    model = model_load("Univariate_SS_H8_F22_D5", device, path="Results/Pths/Univariate/Dropout/")
    model.device = device
    model.to(device)
    
    learning = DeepLearning(model=model, 
                            data_X=data_X, 
                            data_y=data_y,
                            optimiser=Adam(model.parameters()))

    learning.train_val_test()
    learning.create_data_loaders()
    assert(learning.evaluate(learning.model, learning.test_loader) < 1e6)
    
    
def test_validate():
    """Check that an output is obtained for single task mode;"""
    data_X = rand(10, 250, 1)
    data_y = rand(10, 1)
    device = 'cpu'
    
    model = model_load("Univariate_SS_H8_F22_D5", device, path="Results/Pths/Univariate/Dropout/")
    model.device = device
    model.to(device)
    
    learning = DeepLearning(model=model, 
                            data_X=data_X, 
                            data_y=data_y,
                            optimiser=Adam(model.parameters()),
                            n_epochs=1,
                            disp_freq=1e6,
                            fig_disp_freq=1e6)
    
    learning.train_val_test()
    learning.training_wrapper()
    assert(learning.best_val_score < np.inf)
    
    
def test_mtl():
    """Check that an output is obtained for MTL model"""
    data_X = rand(100, 132, 5)
    data_y = rand(100, 5)
    device = 'cpu'

    model = model_load("MTL_Auto_F5", device, path="Results/Pths/MTL/Univariate/")
    model.device = device
    model.to(device)

    learning = DeepLearning(model=model, 
                            data_X=data_X, 
                            data_y=data_y,
                            optimiser=Adam(model.parameters()),
                            n_epochs=1,
                           debug=False)


    learning.train_val_test()
    learning.create_data_loaders()
    learning.training_wrapper()
    assert(learning.best_val_score < np.inf)