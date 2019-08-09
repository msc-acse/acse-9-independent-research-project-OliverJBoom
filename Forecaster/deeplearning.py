from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from statistics import median


def set_seed(seed, device='cpu'):
    """Sets the random seeds to ensure detemrinistic behaviour
    
    :param seed: the random seed number that is set
    :param device: whether running on cpu or CUDA
    
    :type seed: int
    :type device: string
    
    :return: confirmation that seeds have been set
    :rtype: bool
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


def model_save(model, name, path="Models/"):
    """Saving function to keep track of models"""
    print("Saving model:", path + name + '.pth')
    torch.save(model, path + name + '.pth')


def model_load(model_name, path="Models/"):
    """Loading function for models from google drive"""
    model = torch.load(path + model_name + '.pth')
    return model



class early_stopping:
    """Counter to implement early stopping

    If validation accuracy has not relative improved below
    a relative tolerance set by the user than it breaks the
    training

    If rel_tol is set to 0 it becomes a common counter
  
    :param patience: the amount of epochs without improvement before stopping
    :param rel_tol: the relative improvement percentage that must be achieved
    :param verbose: whether to print the count number
    :param best_score: the best score achieved so far
    :param counter: the amount of epochs without improvement so far
    :param stop: whether stopping criteria is achieved

    :type patience: int
    :type rel_tol: float
    :type verbose: bool
    :type best_score: float
    :type counter: int
    :type stop: bool
    """
    def __init__(self, patience, rel_tol, verbose=False):

        self.patience = patience
        self.rel_tol = rel_tol
        self.verbose = verbose
        self.best_score = np.inf
        self.counter = 0
        self.stop = False


    def __call__(self, score):

        # If the score is under the required relative tolerance
        # increase the counter is incremented
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




class DeepLearning():
    """Class to perform training and validation for a given model
    
    :param model: the neural network model
    :param data_X: the training dataset
    :param data_y: the target dataset
    :param n_epochs: the number of epochs of training
    :param optimiser: the type of optimiser used
    :param batch_size: the batch size
    :param loss_function: the loss function used
    :param device: running on cpu or CUDA
    :param seed: the random seed set
    :param debug: whether to print some parameters for checking
    :param disp_freq: the frequency that training/validation metrics will be printed
    :param fig_disp_freq: the frequency that training/validation prediction figures will be made
    :param early_stop: whether early stopping is utilized
    :param early_verbose: whether to print out the early stopping counter
    :param patience: the amount of epochs without improvement before stopping
    :param rel_tol: the relative improvement percentage that must be achieved
    :param scaler_data_X: the data X scaler object for inverse scaling 
    :param scaler_data_y: the dataX y scaler object for inverse scaling 
    
    :type patience: int
    :type model: LSTM
    :type data_X: np.array
    :type data_y: np.array
    :type n_epochs: int
    :type optimiser: torch.optim
    :type batch_size: int
    :type loss_function: torch.nn.modules.loss
    :type device: string
    :type seed: int
    :type debug: bool
    :type disp_freq: int
    :type fig_disp_freq: int
    :type early_stop: bool
    :type early_verbose: bool
    :type patience: int
    :type rel_tol: float
    :rtype scaler_data_X: sklearn.preprocessing.data.MinMaxScaler
    :rtype scaler_data_y: sklearn.preprocessing.data.MinMaxScaler
    
    """
    def __init__(self, model, data_X, data_y,
                 n_epochs,
                 optimiser,
                 batch_size,
                 loss_function=torch.nn.MSELoss(size_average=False),
                 device="cpu",
                 seed=42,
                 debug = True,
                 disp_freq=20,
                 fig_disp_freq=50,
                 early_stop=True,
                 early_verbose=False,
                 patience=50,
                 rel_tol=0,
                 scaler_data_X=None,
                 scaler_data_y=None):

        # The neural network architecture
        self.model = model

        # The optimiser for gradient descent
        self.optimiser = optimiser

        # Dataframe of training values
        self.data_X = data_X

        # Dataframe of target values
        self.data_y = data_y

        # The number of epochs
        self.n_epochs = n_epochs

        #self.optimiser = optimiser
        self.loss_function = loss_function

        # Whether to run on cpu or gpu
        self.device = device

        # The random seed to set
        self.seed = seed

        # To activate debug mode
        self.debug = debug

        # For training/val logs
        self.logs = {}

        # The array of predicted lists for each batch for training
        self.train_predictions = None

        # The array of predicted lists for each batch for validation
        self.val_predictions = None

        # The array of predicted lists for each batch for validation
        self.test_predictions = None

        # The data
        self.X_train = None
        self.X_val = None
        self.X_test = None

        # The targets
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Val data loader
        self.val_loader = None

        # Test data loader
        self.test_loader = None

        # Batch size
        self.batch_size = batch_size

        # How frequently the loss is printed
        self.disp_freq = disp_freq

        # How frequently a figure is plotted
        self.fig_disp_freq = fig_disp_freq

        # Storing the best model
        self.best_model = self.model

        # Dummy variable for inspecting quantities
        self.inspect = None

        # For running dataloaders on CUDA
        self.pin_memory = False

        self.early_stop = early_stop

        if self.early_stop:
            self.early = early_stopping(patience=patience, rel_tol=rel_tol, verbose=early_verbose)
    
        self.MTL = False
        
        # For inverse scaling
        self.scaler_data_X = scaler_data_X
        self.scaler_data_y = scaler_data_y
        

    def train_val_test(self):
        """Splits the dataframes in to a training, validation
        and test set and creates torch tensors from the underlying
        numpy arrays"""
        # Splitting the sets into train, test and validation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_X, self.data_y, test_size=0.2, shuffle=False)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25, shuffle=False)

        if self.debug:
            print("Train Length: \t\t%i\nValidation Length: \t%i\nTest Length:\t\t%i"
                  % (len(X_train), len(X_val), len(X_test)))

        # Tensor of training data
        self.X_train = torch.from_numpy(self.X_train).float()
        self.y_train = torch.from_numpy(self.y_train).float()

        # Tensor of training labels
        self.X_val = torch.from_numpy(self.X_val).float()
        self.y_val = torch.from_numpy(self.y_val).float()

        #  Tensor of test data
        self.X_test = torch.from_numpy(self.X_test).float()
        self.y_test = torch.from_numpy(self.y_test).float()

        # Size Check
        if self.debug:
            print("\nInitial Size Check:")
            self.size_check()
            
        if self.y_test.dim()!=1:
            self.MTL = True

    def size_check(self):
        """Checks the size of the datasets"""
        if self.debug:
            print("\nX Train Shape:\t\t", self.X_train.size())
            print("X Val Shape:\t\t", self.X_val.size())
            print("X Test Shape:\t\t", self.X_test.size())

            print("\ny Train Shape:\t\t", self.y_train.size())
            print("y Val Shape:\t\t", self.y_val.size())
            print("y Test Shape:\t\t", self.y_test.size())

    def create_data_loaders(self):
        """Forms iterators to pipeline in the data
        """
        # Create tensor datasets
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        test_dataset = TensorDataset(self.X_test, self.y_test)

        if self.device =='cuda': self.pin_memory=True

        # Data loaders
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,  pin_memory=self.pin_memory)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,  pin_memory=self.pin_memory)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8,  pin_memory=self.pin_memory)


    def train(self, train_loader):
        """Performs a single training cycle and returns the
        loss metric for the training dataset
        
        :param train_loader: the iterator that feeds in the training data
        :type train_loader: torch.utils.data.dataloader.DataLoader

        :return: the error metric for that epoch
        :rtype: float
        """
        # Sets the model to train mode
        self.model.train()

        train_loss = 0.

        # Selecting the number of output features
        if self.MTL: 
            pred_list = np.empty((0, self.y_test.shape[1]))
        else: 
            pred_list = np.empty((0, 1))
        
        # The data loader creates batches of data to train
        for i_batch, (X_train_batch, y_train_batch) in enumerate(train_loader):  

            # Sending the data to GPU if available
            X_train_batch = X_train_batch.to(self.device)
            y_train_batch = y_train_batch.to(self.device)

            # Zeros the gradients
            self.optimiser.zero_grad()

            # Need to set seed here to make deterministic
            set_seed(42)

            # Perform forward pass
            y_pred = self.model(X_train_batch)

            # Calculate loss for the batch
            loss = self.loss_function(y_pred, y_train_batch)

            # Perform backward pass
            loss.backward()
            
            # Adding the predictions for this batch to prediction list
            pred_list = np.concatenate([pred_list, y_pred.detach().cpu().numpy()], axis=0)

            # Calculate the training loss
            train_loss += (loss * X_train_batch.size()[0]).detach().cpu().numpy()

            # Update Parameters
            self.optimiser.step()            
        
        # Saving the predictions
        self.train_predictions = pred_list
            
        return train_loss/len(train_loader.dataset.tensors[0])


    def validate(self, val_loader):
        """Evaluates the performance of the network
        on unseen validation data
        
        :param val_loader: the iterator that feeds in the validation data
        :type val_loader: torch.utils.data.dataloader.DataLoader

        :return: the error metric for that epoch
        :rtype: float
        """
        # Set the model to evaluate mode
        self.model.eval()

        val_loss = 0.

        # List of each batch predictions
        if self.MTL: 
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
                val_pred_list = np.concatenate([val_pred_list, y_pred.detach().cpu().numpy()], axis=0)
            
                # Calculate the validation loss
                val_loss += (loss * X_val_batch.size()[0]).detach().cpu().numpy()

        # Converting an array of batches of predictions to a list of predictions
        self.val_predictions = val_pred_list

        return val_loss/len(val_loader.dataset.tensors[0])


    def evaluate(self, model, test_loader):
        """Evaluates the performance of the network 
        on given data for a given model
        
        A lot of overlap of code with validation. Only kept separate
        due to inspection of attibutes made easier when running simulations
        if kept separate
        
        :param test_loader: the iterator that feeds in the data of choice
        :type test_loader: torch.utils.data.dataloader.DataLoader

        :return: the error metric for that dataset
        :rtype: float
        """
        # Set the model to evaluate mode
        model = model.eval()

        test_loss = 0.

        # Selecting the number of output features
        if self.MTL: 
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
                test_pred_list = np.concatenate([test_pred_list, y_pred.detach().cpu().numpy()], axis=0)

                # Calculate the validation loss
                test_loss += (loss * X_test_batch.size()[0]).detach().cpu().numpy()

        # Converting an array of batches of predictions to a list of predictions
        self.test_predictions = test_pred_list

        return test_loss/len(test_loader.dataset.tensors[0])


    def live_pred_plot(self):
        """Plots the training predictions, validation predictions
        and the live training/validation losses
        """        
        if self.MTL:
            fig, ax = plt.subplots(1, 5, figsize= (24, 5))

            ax[0].set_title("Training Predictions")
            ax[0].plot(self.train_predictions, label="Predicted")
            ax[0].plot(self.y_train.numpy(), '--',label="Observed")
            ax[0].legend()

            ax[1].set_title("Validation Predictions")
            ax[1].plot(self.val_predictions, label="Predicted")
            ax[1].plot(self.y_val.numpy(), '--', label="Observed")
            ax[1].legend()

            ax[2].set_title("Loss Plots")
            ax[2].plot(self.logs['Training Loss'], label="Training Loss")
            ax[2].plot(self.logs['Validation Loss'], label="Validation Loss")
            ax[2].legend()

            ax[3].set_title("Single Metal Inspection Train")
            ax[3].plot(self.train_predictions[:, 0], label="Predicted")
            ax[3].plot(self.y_train.numpy()[:, 0],label="Observed")
            ax[3].legend()
            
            ax[4].set_title("Single Metal Inspection Val")
            ax[4].plot(self.val_predictions[:, 0], label="Predicted")
            ax[4].plot(self.y_val.numpy()[:, 0],label="Observed")
            ax[4].legend()
            plt.show()
            
        else: 
            fig, ax = plt.subplots(1, 3, figsize= (20, 5))

            ax[0].set_title("Training Predictions")
            ax[0].plot(self.train_predictions, label="Predicted")
            ax[0].plot(self.y_train.numpy(), label="Observed")
            ax[0].legend()

            ax[1].set_title("Validation Predictions")
            ax[1].plot(self.val_predictions, label="Predicted")
            ax[1].plot(self.y_val.numpy(), label="Observed")
            ax[1].legend()

            ax[2].set_title("Loss Plots")
            ax[2].plot(self.logs['Training Loss'], label="Training Loss")
            ax[2].plot(self.logs['Validation Loss'], label="Validation Loss")
            ax[2].legend()
            plt.show()

    def training_wrapper(self):
        """The wrapper that performs the training and validation
        """
        # start timer
        start_time = time.time()

        # set seed
        set_seed(int(self.seed))

        # Create data loaders
        self.create_data_loaders()

        train_log = []
        val_log = []

        # The best validation score tracker
        self.best_val_score = np.inf

        # Begin training
        for epoch in range(self.n_epochs):

            live_logs = {}

            train_loss = self.train(self.train_loader)
            val_loss = self.validate(self.val_loader)

            # Saving the best model
            if val_loss.item() <= self.best_val_score:
                self.best_model = deepcopy(self.model)
                self.best_val_score = val_loss.item()

            train_log.append(train_loss.item())
            val_log.append(val_loss.item())
            self.logs["Training Loss"] = train_log
            self.logs["Validation Loss"] = val_log
            self.logs["Time"] = time.time() - start_time

            # Checking stopping criteria
            if self.early_stop:
                self.early(val_loss.item())
                if self.early.stop:
                    print("Early Stopping")
                    self.model = self.best_model
                    break
            
            #  Printing key metrics to screen
            if (epoch % self.disp_freq == 0):
                print("Epoch: %i Train: %.5f Val: %.5f  Time: %.3f  Best Val: %.5f" % (epoch, train_loss.item(),
                                                          val_loss.item(), (time.time() - start_time), self.best_val_score))

            # Plotting predictions and training metrics
            if (epoch % self.fig_disp_freq == 0):
                self.live_pred_plot()

        # Storing the best model
        self.model = self.best_model

        
def param_strip(param):
    """Strips the key text info out of certain parameters"""
    return str(param)[:str(param).find('(')]


def full_save(model, model_name, optimiser, num_epoch, learning_rate, momentum, weight_decay, use_lg_returns,
              PCA_used, data_X, train_loss, val_loss, test_loss, train_time, hidden_dim,
              mse, mae, mde, path="Models/CSVs/"):
    """Saves the models weights and hyperparameters to a pth file and csv file"""
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

    np.savetxt(path + model_name + '_' + str(val_loss).replace(".", "_")[:5] + ".csv", np.r_[ind, row], fmt='%s', delimiter=',')
    return True