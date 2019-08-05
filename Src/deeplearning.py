import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split


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
        
        
        
        
class DeepLearning():
    """Class to perform training and validation for a given model
    :param
    model: nn.Module the model to be trained
    df_X
    df_y
    n_epochs,
    optimiser, 
    window_size,
    loss_function
    device
    seed=42
    debug
    """
    def __init__(self, model, data_X, data_y,
                 n_epochs,
                 optimiser, 
                 window_size,
                 loss_function=torch.nn.MSELoss(size_average=False),
                 device="cpu", 
                 seed=42,
                 debug = True, 
                 disp_freq=20,
                 fig_disp_freq=50, 
                 early_stop=True,
                 patience=50,
                 tol=0):
        
        
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
        
        # The length of the time series window
        self.window_size = window_size
        
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
            self.early = early_stopping(patience=patience, rel_tol=tol)
        
        
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
        """Forms iterators to pipeline in the data"""
            
        # Create tensor datasets
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        
        if self.device =='cuda': self.pin_memory=True
          
        # Data loaders
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=8,  pin_memory=self.pin_memory)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=8,  pin_memory=self.pin_memory)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8,  pin_memory=self.pin_memory)
    
    
    def train(self, train_loader):
        """Performs a single training cycle and returns the
        mean squared error loss for the training dataset"""
        # Sets the model to train mode
        self.model.train()
        
        train_loss = 0.
        
        # List of each batch predictions
        pred_list = []
        
        # The data loader creates batches of data to train
        for i_batch, (X_train_batch, y_train_batch) in enumerate(train_loader):
            
            #print('Batch : ', i_batch)
            
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
            pred_list.append(y_pred)
            
            # Calculate the training loss
            train_loss += (loss * X_train_batch.size()[0]).detach().cpu().numpy()

            # Update Parameters
            self.optimiser.step()               
        
        # Converting an array of batches of predictions to a list of predictions
        self.train_predictions = [single_pred for batch in pred_list for single_pred in batch.detach().cpu().numpy()]
        
        return train_loss/len(train_loader.dataset.tensors[0])
    
    
    def validate(self, val_loader):
        """Evaluates the performance of the network
        on unseen validation data"""
        # Set the model to evaluate mode
        self.model.eval()
        
        val_loss = 0.
        
        # List of each batch predictions
        val_pred_list = []
        
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
                val_pred_list.append(y_pred)
                
                # Calculate the validation loss
                val_loss += (loss * X_val_batch.size()[0]).detach().cpu().numpy()
        
        # Converting an array of batches of predictions to a list of predictions
        self.val_predictions = [single_pred for batch in val_pred_list for single_pred in batch.detach().cpu().numpy()]
        
        return val_loss/len(val_loader.dataset.tensors[0])
    
    
    def evaluate(self, model, test_loader):
        """Evaluates the performance of the network on unseen test data"""
        # Set the model to evaluate mode
        model.eval()

        test_loss = 0.

        # List of each batch predictions
        test_pred_list = []

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
                test_pred_list.append(y_pred)

                # Calculate the validation loss
                test_loss += (loss * X_test_batch.size()[0]).detach().cpu().numpy()

        # Converting an array of batches of predictions to a list of predictions
        self.test_predictions = [single_pred for batch in test_pred_list for single_pred in batch.detach().cpu().numpy()]

        return test_loss/len(test_loader.dataset.tensors[0])
      
    
    def live_pred_plot(self):
        fig, ax = plt.subplots(1, 3, figsize= (20, 5))
  
        ax[0].plot(self.train_predictions, label="Predicted")
        ax[0].plot(self.y_train.numpy(), label="Observed")
        ax[0].legend()
        
        ax[1].plot(self.val_predictions, label="Val Predicted")
        ax[1].plot(self.y_val.numpy(), label="Val Observed")
        ax[1].legend()
        
        ax[2].set_title("Loss Plots")
        ax[2].plot(self.logs['Training Loss'], label="Training Loss")
        ax[2].plot(self.logs['Validation Loss'], label="Validation Loss")
        ax[2].legend()
        ax[2].set_ylim((0, median(self.logs['Validation Loss'])))
        
        plt.show()
    
    def training_wrapper(self):
        
        # start timer
        start_time = time.time()

        # set seed
        set_seed(int(self.seed))
        
        # Create data loaders
        learning.create_data_loaders()
        
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
            if val_loss <= self.best_val_score:
                self.best_model = self.model
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
                    break
                  
            if (epoch % self.disp_freq == 0): 
                print("Epoch: %i Train MSE: %.5f Val MSE: %.5f  Time: %.3f" % (epoch, train_loss.item(),
                                                          val_loss.item(), (time.time() - start_time)))
                
            if (epoch % self.fig_disp_freq == 0):
                self.live_pred_plot()