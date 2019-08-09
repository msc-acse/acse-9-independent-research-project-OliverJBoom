"""File for deep learning model architecture"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """A Long Short Term Memory network
    model with an additional dense layer"""
        
    def __init__(self, num_features, hidden_dim, dense_hidden, output_dim,
                 batch_size, series_length, device,
                 dropout=0.1, num_layers=2, debug=True):
        
        super(LSTM, self).__init__()
        
        # Number of features
        self.num_features = num_features
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of neurons in the dense layer
        self.dense_hidden = dense_hidden

        # Number of hidden layers
        self.num_layers = num_layers
        
        # The output dimensions
        self.output_dim = output_dim
        
        # Batch Size
        self.batch_size = batch_size
        
        # Length of sequence
        self.series_length = series_length
        
        # CPU or GPU
        self.device = device
        
        self.dropout = dropout
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size = self.num_features, 
            hidden_size =self.hidden_dim,
            dropout = self.dropout,
            num_layers =self.num_layers)
        
        
        # Defining the Dense Layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dense_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dense_hidden, self.output_dim)) 

        
    def init_hidden(self, batch_size):
        """Initialised the hidden state to be zeros"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
    
    
    def forward(self, x):
        """Forward pass through the neural network"""  
        # Adjust to a variable batch size 
        batch_size = x.size()[0]
        series_length = x.size()[1]

        assert (series_length == self.series_length)
        
        # Keeps the dimensions constant regardless of batchsize
        x = x.contiguous().view(series_length, batch_size, -1) 

        # Initialises the hidden states
        # Not a stateful LSTM
        h0, c0 = self.init_hidden(batch_size)
        
        # Pass through through lstm layer
        # Only the x is of interest
        x, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Output is seq to seq but only want seq to val
        # So only use the final value of the lstm outputted
        # sequence
        x = x[-1]  
        
        # Fully connected dense layers with dropout
        x = self.fc(x)

        return x

    
    
#     class LSTM(nn.Module):
#     """A Long Short Term Memory network
#     model

#     Specifically to load pickles of the univariate experiments
#     if so desired. As changed the class after this"""
        
#     def __init__(self, num_features, hidden_dim, output_dim,
#                  batch_size, series_length, device, 
#                  dropout=0.1, num_layers=2, debug=True):
        
#         super(LSTM, self).__init__()
        
#         # Number of features
#         self.num_features = num_features
        
#         # Hidden dimensions
#         self.hidden_dim = hidden_dim

#         # Number of hidden layers
#         self.num_layers = num_layers
        
#         # The output dimensions
#         self.output_dim = output_dim
        
#         # Batch Size
#         self.batch_size = batch_size
        
#         # Length of sequence
#         self.series_length = series_length
        
#         # Dropout Probability
#         self.dropout = dropout
        
#         # CPU or GPU
#         self.device = device
        
#         # Define the LSTM layer
#         self.lstm = nn.LSTM(
#             input_size = self.num_features, 
#             hidden_size =self.hidden_dim,
#             dropout = self.dropout,
#             num_layers =self.num_layers)

#         # Fully Connected Layer
#         self.fc1 = nn.Linear(in_features=self.hidden_dim, 
#                              out_features=self.hidden_dim)
        
#         # Activation function
#         self.act = nn.ReLU()
        
#         # Output layer
#         self.out = nn.Linear(in_features=self.hidden_dim, 
#                              out_features=self.output_dim)
        
        
#     def init_hidden(self, batch_size):
#         """Initialised the hidden state to be zeros"""
#         return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
#                 torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
    
    
#     def forward(self, x):
#         """Forward pass through the neural network"""
        
#         """TODO Directly switch these variables in 
#         the permute of the dataset"""
        
#         # Adjust to a variable batch size 
#         batch_size = x.size()[0]
#         series_length = x.size()[1]
        
#         #print("series_length, batch_size", series_length, batch_size)

#         assert (series_length == self.series_length)
        
#         """TODO Check output of contiguous and non 
#         contigious memory"""
        
#         # Keeps the dimensions constant regardless of batchsize
#         x = x.contiguous().view(series_length, batch_size, -1) 

#         # Initialises the hidden states
#         h0, c0 = self.init_hidden(batch_size)
        
#         # Pass through through lstm layer
#         # Only the x is of interest
#         x, (hn, cn) = self.lstm(x, (h0, c0))
        
#         # Output is seq to seq but only want seq to val
#         # So only use the final value of the lstm outputted
#         # sequence
#         x = x[-1]  
        
#         # Fully connected hidden layer
#         x = self.act(self.fc1(x))
        
#         return self.out(x)
   
