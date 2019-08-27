"""
This module include a set of functions relating to the training,
validation and testing of neural networks.

Author: Oliver Boom
Github Alias: OliverJBoom
"""

from copy import deepcopy

import time
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def set_seed(seed):
    """Sets the random seeds to ensure deterministic behaviour.

    :param seed:            The number that is set for the random seeds
    :type  seed:            int

    :return:                Confirmation that seeds have been set
    :rtype:                 bool
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If running on CUDA additional seeds need to be set
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    warnings.filterwarnings("ignore")
    return True


def model_save(model, name, path="Results/Pths/"):
    """Saving function for the model.

    :param model:          The model to save
    :type  model:          torch.nn

    :param name:           The name to save the model under
    :type  name:           string

    :param path:           The directory path to save the model in
    :type  path:           string
    """
    print("Saving model:", path + name + '.pth')
    torch.save(model, path + name + '.pth')


def model_load(model_name, device, path="Results/Pths/"):
    """Loading function for the models.

    :param model_name:     The model name to load
    :type  model_name:     string

    :param device:         The device to run on (Cpu or CUDA)
    :type  device:         string

    :param path:           The directory path to load the model from
    :type  path:           string
    """
    model = torch.load(path + model_name + '.pth', map_location=device)
    return model


class EarlyStopping:
    """Used to facilitate early stopping during the training
    of neural networks.

    When called if the validation accuracy has not relative improved below a
    relative tolerance set by the user the a counter is incremented. If the
    counter passes a set value then the stop attribute is set to true. This
    should be used as a break condition in the training loop.

    If rel_tol is set to 0 then the metric just needs to improve from it's
    existing value

    :param patience:   The amount of epochs without improvement before stopping
    :type  patience:   int

    :param rel_tol:    The relative improvement % that must be achieved
    :type  rel_tol:    float

    :param verbose:    Whether to print the count number
    :type  verbose:    bool

    :param best:       The best score achieved so far
    :type  best:       float

    :param counter:    The amount of epochs without improvement so far
    :type  counter:    int

    :param stop:       Whether stopping criteria is achieved
    :type  stop:       bool
    """

    def __init__(self, patience, rel_tol, verbose=True):

        self.patience = patience
        self.rel_tol = rel_tol
        self.verbose = verbose
        self.best_score = np.inf
        self.counter = 0
        self.stop = False

    def __call__(self, score):
        """Every time the object is called the score is checked to see if it
        has improved. If it hasn't the counter is incremented, if it has
        improved then the counter resets"""
        if score >= self.best_score * (1 - self.rel_tol):
            self.counter += 1
        else:
            self.counter = 0

        if score < self.best_score:
            self.best_score = score

        if self.counter >= self.patience:
            self.stop = True

        if self.verbose:
            print("Count:", self.counter)


class DeepLearning:
    """Class to perform training and validation for a given model

    :param model:           The neural network model
    :type  model:           nn.module

    :param data_X:          The training dataset
    :type  data_X:          np.array

    :param data_y:          the target dataset
    :type  data_y:          np.array

    :param n_epochs:        The number of epochs of training
    :type  n_epochs:        int

    :param optimiser:       The type of optimiser used
    :type  optimiser:       torch.optim

    :param batch_size:      The batch size
    :type  batch_size:       int

    :param loss_function:   The loss function used
    :type  loss_function:   torch.nn.modules.loss

    :param device:          The device to run on (Cpu or CUDA)
    :type  device:          string

    :param seed:            The number that is set for the random seeds
    :type  seed:            int

    :param debug:           Whether to print some parameters for checking
    :type debug:            bool

    :param disp_freq:       The epoch frequency that training/validation
                            metrics will be printed on
    :type disp_freq:        int

    :param fig_disp_freq:   The frequency that training/validation prediction
                            figures will be made
    :type fig_disp_freq:    int

    :param early_stop:      Whether early stopping is utilized
    :type  early_stop:      bool

    :param early_verbose:   Whether to print out the early stopping counter
    :type  early_verbose:    bool

    :param patience:        The amount of epochs without improvement before
    :type  patience:        stopping int

    :param rel_tol:         The relative improvement percentage that must be
                            achieved float
    :type rel_tol:

    :param scaler_data_X:   The data X scaler object for inverse scaling
    :type  scaler_data_X:   sklearn.preprocessing.data.MinMaxScaler

    :param scaler_data_y:   The dataX y scaler object for inverse scaling
    :type scaler_data_y:    sklearn.preprocessing.data.MinMaxScaler
    """

    def __init__(self, model, data_X, data_y,
                 optimiser,
                 batch_size=128,
                 n_epochs=100,
                 loss_function=torch.nn.MSELoss(reduction='sum'),
                 device="cpu",
                 seed=42,
                 debug=True,
                 disp_freq=20,
                 fig_disp_freq=50,
                 early_stop=True,
                 early_verbose=False,
                 patience=50,
                 rel_tol=0,
                 scaler_data_X=None,
                 scaler_data_y=None):

        # Given parameters
        self.model = model
        self.optimiser = optimiser
        self.data_X = data_X
        self.data_y = data_y
        self.n_epochs = n_epochs
        self.loss_function = loss_function
        self.device = device
        self.seed = seed
        self.debug = debug
        self.disp_freq = disp_freq
        self.fig_disp_freq = fig_disp_freq
        self.batch_size = batch_size
        self.early_stop = early_stop

        # The maxminscaler objects used to scale the data.
        # Stored so can be used for inverse scaling later if loading
        # pickled objects
        self.scaler_data_X = scaler_data_X
        self.scaler_data_y = scaler_data_y

        # Store for training/val logs
        self.logs = {}

        # The array of predicted values calculated using the training function
        self.train_predictions = None

        # The array of predicted values calculated using the validate function
        self.val_predictions = None

        # The array of predicted values calculated using the evaluate function
        self.test_predictions = None

        # The data
        self.X_train = None
        self.X_val = None
        self.X_test = None

        # The targets
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # The data loader
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Storing the best model
        self.best_model = self.model

        # Dummy variable for inspecting quantities
        self.inspect = None

        # Used to running data loaders on CUDA
        self.pin_memory = False

        # Initialising the early stop object
        if self.early_stop:
            self.early = EarlyStopping(patience=patience,
                                       rel_tol=rel_tol,
                                       verbose=early_verbose)

        # Tag for using multi task learning
        self.mtl = False

        # The tracker for the best validation score
        self.best_val_score = np.inf

    def train_val_test(self):
        """Splits the DataFrames in to a training, validation
        and test set and creates torch tensors from the underlying
        numpy arrays"""
        # Splitting the datasets into (train/val) and test.
        self.X_train, self.X_test, \
        self.y_train, self.y_test = train_test_split(
            self.data_X, self.data_y, test_size=0.2, shuffle=False)

        # Splitting the (train/val) dataset into train and val datasets.
        self.X_train, self.X_val, \
        self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.25, shuffle=False)

        if self.debug:
            print("Train Length: \t\t%i\n"
                  "Validation Length: \t%i\n"
                  "Test Length:\t\t%i"
                  % (len(self.X_train), len(self.X_val), len(self.X_test)))

        # Tensor of training data/labels
        self.X_train = torch.from_numpy(self.X_train).float()
        self.y_train = torch.from_numpy(self.y_train).float()

        # Tensor of validation data/labels
        self.X_val = torch.from_numpy(self.X_val).float()
        self.y_val = torch.from_numpy(self.y_val).float()

        #  Tensor of test data/labels
        self.X_test = torch.from_numpy(self.X_test).float()
        self.y_test = torch.from_numpy(self.y_test).float()

        # Size Check
        if self.debug:
            print("\nInitial Size Check:")
            self.size_check()

        # If the labels have more than one target feature then using MTL
        if self.y_test.shape[1] > 1:
            self.mtl = True

    def size_check(self):
        """Checks the size of the datasets"""
        print("\nX Train Shape:\t\t", self.X_train.size())
        print("X Val Shape:\t\t", self.X_val.size())
        print("X Test Shape:\t\t", self.X_test.size())

        print("\ny Train Shape:\t\t", self.y_train.size())
        print("y Val Shape:\t\t", self.y_val.size())
        print("y Test Shape:\t\t", self.y_test.size())

    def create_data_loaders(self):
        """Forms iterators to pipeline in the data/labels"""
        # Creating Tensor datasets
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        test_dataset = TensorDataset(self.X_test, self.y_test)

        if self.device == 'cuda':
            self.pin_memory = True

        # Creating data loaders
        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False, num_workers=4,
                                       pin_memory=self.pin_memory)
        self.val_loader = DataLoader(dataset=val_dataset,
                                     batch_size=self.batch_size, shuffle=False,
                                     num_workers=4, pin_memory=self.pin_memory)
        self.test_loader = DataLoader(dataset=test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False, num_workers=4,
                                      pin_memory=self.pin_memory)

    def train(self, train_loader):
        """Performs a single training epoch and returns the loss metric
        for the training dataset.

        :param train_loader:     The iterator that feeds in the training data
        :type  train_loader:     torch.utils.data.dataloader.DataLoader

        :return:                 The error metric for that epoch
        :rtype:                  float
        """
        # Sets the model to train mode
        self.model.train()

        train_loss = 0.

        # Initialising the empty array for target predictions
        if self.mtl:
            pred_list = np.empty((0, self.y_test.shape[1]))
        else:
            pred_list = np.empty((0, 1))

        # The data loader creates batches of data to train
        for X_train_batch, y_train_batch in train_loader:
            # Sending the data to GPU if available
            X_train_batch = X_train_batch.to(self.device)
            y_train_batch = y_train_batch.to(self.device)

            # Zeros the gradients
            self.optimiser.zero_grad()

            # Some background operation in the underlying CUDA optimisation
            # code changes the seeds, so resetting it here ensures
            # deterministic results
            set_seed(42)

            # Perform forward pass
            y_pred = self.model(X_train_batch)

            # Calculate loss for the batch
            loss = self.loss_function(y_pred, y_train_batch)

            # Perform backward pass
            loss.backward()

            # Adding the predictions for this batch to prediction list
            pred_list = np.concatenate(
                [pred_list, y_pred.detach().cpu().numpy()], axis=0)

            # Calculate the training loss
            train_loss += (loss*X_train_batch.size()[0]).detach().cpu().numpy()

            # Update Parameters
            self.optimiser.step()

        # Storing the predictions
        self.train_predictions = pred_list

        return train_loss / len(train_loader.dataset.tensors[0])

    def validate(self, val_loader):
        """Evaluates the performance of the network on unseen validation data.

        :param val_loader:      the iterator that feeds in the validation data
        :type  val_loader:      torch.utils.data.dataloader.DataLoader

        :return:                the error metric for that epoch
        :rtype:                 float
        """
        # Set the model to evaluate mode
        self.model.eval()

        val_loss = 0.

        # Initialising the empty array for target predictions
        if self.mtl:
            val_pred_list = np.empty((0, self.y_test.shape[1]))
        else:
            val_pred_list = np.empty((0, 1))

        # The data loader creates batches of data to validate
        for X_val_batch, y_val_batch in val_loader:
            # Ensures that the gradients are not updated
            with torch.no_grad():
                # Sending the data to GPU if available
                X_val_batch = X_val_batch.to(self.device)
                y_val_batch = y_val_batch.to(self.device)

                # Perform forward pass
                y_pred = self.model(X_val_batch)

                # Calculate loss for the batch
                loss = self.loss_function(y_pred, y_val_batch)

                # Adding the predictions for this batch to prediction list
                val_pred_list = np.concatenate(
                    [val_pred_list, y_pred.detach().cpu().numpy()], axis=0)

                # Calculate the validation loss
                val_loss += (loss * X_val_batch.size()[
                    0]).detach().cpu().numpy()

        # Storing the predictions
        self.val_predictions = val_pred_list

        return val_loss / len(val_loader.dataset.tensors[0])

    def evaluate(self, model, test_loader):
        """Evaluates the performance of the network on given data for a given
        model.

        A lot of overlap of code with validation. Only kept separate due to the
        inspection of attributes being made easier when running simulations
        if kept separate.

        :param model:         The model to evaluate
        :type  model:         nn.module

        :param test_loader:   The iterator that feeds in the data of choice
        :type  test_loader:   torch.utils.data.dataloader.DataLoader

        :return:              The error metric for that dataset
        :rtype:               float
        """
        # Set the model to evaluate mode
        model = model.eval()

        test_loss = 0.

        # Selecting the number of output features
        if self.mtl:
            test_pred_list = np.empty((0, self.y_test.shape[1]))
        else:
            test_pred_list = np.empty((0, 1))

        # The data loader creates batches of data to validate
        for X_test_batch, y_test_batch in test_loader:
            # Ensures that the gradients are not updated
            with torch.no_grad():
                # Sending the data to GPU if available
                X_test_batch = X_test_batch.to(self.device)
                y_test_batch = y_test_batch.to(self.device)

                # Perform forward pass
                y_pred = model(X_test_batch)

                # Calculate loss for the batch
                loss = self.loss_function(y_pred, y_test_batch)

                # Adding the predictions for this batch to prediction list
                test_pred_list = np.concatenate(
                    [test_pred_list, y_pred.detach().cpu().numpy()], axis=0)

                # Calculate the validation loss
                test_loss += (loss * X_test_batch.size()[
                    0]).detach().cpu().numpy()

        # Storing the predictions
        self.test_predictions = test_pred_list

        return test_loss / len(test_loader.dataset.tensors[0])

    def live_pred_plot(self):
        """Plots the training predictions, validation predictions and the
         training/validation losses as they are predicted.
        """
        # More plots are required for MTL then single task
        if self.mtl:
            _, axes = plt.subplots(1, 5, figsize=(24, 5))

            axes[0].set_title("Training Predictions")
            axes[0].plot(self.train_predictions, label="Predicted")
            axes[0].plot(self.y_train.numpy(), '--', label="Observed")
            axes[0].legend()

            axes[1].set_title("Validation Predictions")
            axes[1].plot(self.val_predictions, label="Predicted")
            axes[1].plot(self.y_val.numpy(), '--', label="Observed")
            axes[1].legend()

            axes[2].set_title("Loss Plots")
            axes[2].plot(self.logs['Training Loss'], label="Training Loss")
            axes[2].plot(self.logs['Validation Loss'], label="Validation Loss")
            axes[2].legend()

            axes[3].set_title("Single Metal Inspection Train")
            axes[3].plot(self.train_predictions[:, 0], label="Predicted")
            axes[3].plot(self.y_train.numpy()[:, 0], label="Observed")
            axes[3].legend()

            axes[4].set_title("Single Metal Inspection Val")
            axes[4].plot(self.val_predictions[:, 0], label="Predicted")
            axes[4].plot(self.y_val.numpy()[:, 0], label="Observed")
            axes[4].legend()
            plt.show()

        else:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

            axes[0].set_title("Training Predictions")
            axes[0].plot(self.train_predictions, label="Predicted")
            axes[0].plot(self.y_train.numpy(), label="Observed")
            axes[0].legend()

            axes[1].set_title("Validation Predictions")
            axes[1].plot(self.val_predictions, label="Predicted")
            axes[1].plot(self.y_val.numpy(), label="Observed")
            axes[1].legend()

            axes[2].set_title("Loss Plots")
            axes[2].plot(self.logs['Training Loss'], label="Training Loss")
            axes[2].plot(self.logs['Validation Loss'], label="Validation Loss")
            axes[2].legend()
            plt.show()

    def training_wrapper(self):
        """The wrapper that performs the training and validation"""
        # start timer
        start_time = time.time()

        # set seed
        set_seed(int(self.seed))

        # Create data loaders
        self.create_data_loaders()

        train_log = []
        val_log = []

        # Begin training
        for epoch in range(self.n_epochs):

            # Train and validate the model
            train_loss = self.train(self.train_loader)
            val_loss = self.validate(self.val_loader)

            # Saves a copy of the model if it improves on previous
            # best val score
            if val_loss <= self.best_val_score:
                self.best_model = deepcopy(self.model)
                self.best_val_score = val_loss

            # Logging the performance of the models
            train_log.append(train_loss)
            val_log.append(val_loss)
            self.logs["Training Loss"] = train_log
            self.logs["Validation Loss"] = val_log
            self.logs["Time"] = time.time() - start_time

            # Checking stopping criteria
            if self.early_stop:
                self.early(val_loss)
                if self.early.stop:
                    print("Early Stopping")
                    self.model = self.best_model
                    break

            #  Printing key metrics to screen
            if self.disp_freq > 0:
                if epoch % self.disp_freq == 0:
                    print("Epoch: %i "
                          "Train: %.5f "
                          "Val: %.5f  "
                          "Time: %.3f  "
                          "Best Val: %.5f"
                          % (epoch, train_loss, val_loss,
                             (time.time() - start_time),
                             self.best_val_score))

            if self.fig_disp_freq > 0:
                # Plotting predictions and training metrics
                if epoch % self.fig_disp_freq == 0:
                    self.live_pred_plot()

        # Storing the best model
        self.model = self.best_model


def param_strip(param):
    """Strips the key text info out of certain parameters.
    Used to save the text info of which models/optimiser objects are used

    :param param:      The parameter object to find the name of
    :type  param:      object
    """
    return str(param)[:str(param).find('(')]


def full_save(model, model_name, optimiser, num_epoch, learning_rate, momentum,
              weight_decay, use_lg_returns,
              PCA_used, data_X, train_loss, val_loss, test_loss, train_time,
              hidden_dim, mse, mae, mde, path="Models/CSVs/"):
    """Saves the models run details and hyper-parameters to a csv file
    :param model:               The model run
    :type  model:               nn.module

    :param model_name:          The name the model is saved under
    :type  model_name:          strin

    :param optimiser:           The optimiser type used
    :type  optimiser:           torch.optim

    :param num_epoch:           The number of epochs run for
    :type  num_epoch:           int

    :param learning_rate:       The learning rate learning hyper-parameter
    :type  learning_rate:       float

    :param momentum:            The momentum learning hyper-parameter
    :type  momentum:            float

    :param weight_decay:        The weight decay learning hyper-parameter
    :type  weight_decay:        float

    :param use_lg_returns:      Whether log returns was used
    :type  use_lg_returns:      bool

    :param PCA_used:            Whether PCA was used
    :type  PCA_used:            bool

    :param data_X:              The training dataset (used to save the shape)
    :type  data_X:              np.array

    :param train_loss:          The loss on the training dataset
    :type  train_loss:          float

    :param val_loss:            The loss on the validation dataset
    :type  val_loss:            float

    :param test_loss:           The loss on the test dataset
    :type  test_loss:           float

    :param train_time:          The amount of time to train
    :type  train_time:          float

    :param hidden_dim:          The number of neurons in the hidden layers
    :type  hidden_dim:          int

    :param mse:                 The mean squared error metric
    :type  mse:                 floot

    :param mae:                 The mean absolute error metric
    :type  mae:                 floot

    :param mde:                 The mean direction error metric
    :type  mde:                 floot

    :param path:                The directory path to save in
    :type  path:                string
    """
    ind = ["Model Class",
           "Optimiser",
           "Epoch Number",
           'Learning Rate',
           "Momentum",
           "Weight Decay",
           "Log Returns Used",
           "PCA",
           "Num Features",
           "Dataset Length",
           "Series Length",
           "Training Loss",
           "Validation loss",
           "Test Loss",
           "Hidden Layer Dimensions",
           "Mean Squared Error",
           "Mean Absolute Error",
           "Mean Directional Accuracy",
           "Training Time"]

    model_class = param_strip(model)

    row = [model_class,
           param_strip(optimiser),
           num_epoch,
           learning_rate,
           momentum,
           weight_decay,
           use_lg_returns,
           PCA_used,
           data_X.shape[2],
           data_X.shape[0],
           data_X.shape[1],
           train_loss,
           val_loss,
           test_loss,
           hidden_dim,
           mse,
           mae,
           mde,
           train_time]

    ind = [str(i) for i in ind]
    row = [str(i) for i in row]

    ind = [",".join(ind)]
    row = [",".join(row)]

    np.savetxt(
        path + model_name + '_' + str(val_loss).replace(".", "_")[:5] + ".csv",
        np.r_[ind, row], fmt='%s', delimiter=',')
    return True
