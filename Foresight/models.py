"""
This module contains the architectures for long short term memory
neural networks.

Author: Oliver Boom
Github Alias: OliverJBoom
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """A Long Short Term Memory network model with an additional dense layer

    :param num_features:      The number of features in the dataset
    :type  num_features:      int

    :param hidden_dim:        The number of neurons in the LSTMs hidden layer/s
    :type  hidden_dim:        int

    :param dense_hidden:      The number of neurons in the dense layers
    :type  dense_hidden:      int

    :param output_dim:        The number of neurons in the output layer
    :type  output_dim:        int

    :param batch_size:        The number of items in each batch
    :type  batch_size:        int

    :param series_length:     The length of the time series
    :type  series_length:     Int

    :param device:            The device to run on (Cpu or CUDA)
    :type  device:            string

    :param dropout:           The probability of dropout
    :type  dropout:           float

    :param num_layers:        The number of stacked LSTM layers
    :type  num_layers:        int
    """

    def __init__(self, num_features, hidden_dim, dense_hidden, output_dim,
                 batch_size, series_length, device,
                 dropout=0.1, num_layers=2):
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
            input_size=self.num_features,
            hidden_size=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.num_layers)

        # Defining the Dense Layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dense_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dense_hidden, self.output_dim))

    def init_hidden(self, batch_size):
        """Initialised the hidden state to be zeros. This clears the hidden
        state between batches. If you are running a stateful LSTM then this
        needs to be changed.

        To change to a stateful LSTM requires not detaching the backprop and
        storing the computational graph. This strongly increases runtime and
        shouldn't make a big difference. Hence a stateful LSTM was not used.

        :param batch_size:          The batch size to be zeroed
        :type  batch_size:          string
        """
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            self.device)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            self.device)
        return c0, h0

    def forward(self, x):
        """Forward pass through the neural network

        :param x:          The input into the network
        :type  x:          torch.Tensor
        """
        # Adjust to a variable batch size
        batch_size = x.size()[0]
        series_length = x.size()[1]

        # Making sure the series and batch haven't been mis-permuted
        assert series_length == self.series_length

        # Keeps the dimensions constant regardless of batch size
        x = x.contiguous().view(series_length, batch_size, -1)

        # Initialises the hidden states
        h0, c0 = self.init_hidden(batch_size)

        # Pass through through LSTM layer
        # Only the x is of interest
        x, (hn, cn) = self.lstm(x, (h0, c0))

        # Output is seq to seq but only want seq to val
        # So only use the final slice of the LSTM outputted sequence
        x = x[-1]

        # Fully connected dense layers with dropout
        x = self.fc(x)

        return x


class LSTM_shallow(nn.Module):
    """A Long Short Term Memory network model that passes staight
    from the LSTM layer to predictions

    :param num_features:      The number of features in the dataset
    :type  num_features:      int

    :param hidden_dim:        The number of neurons in the LSTMs hidden layer/s
    :type  hidden_dim:        int

    :param output_dim:        The number of neurons in the output layer
    :type  output_dim:        int

    :param batch_size:        The number of items in each batch
    :type  batch_size:        int

    :param series_length:     The length of the time series
    :type  series_length:     Int

    :param device:            The device to run on (Cpu or CUDA)
    :type  device:            string

    :param dropout:           The probability of dropout
    :type  dropout:           float

    :param num_layers:        The number of stacked LSTM layers
    :type  num_layers:        int
    """
    def __init__(self, num_features, hidden_dim, output_dim,
                 batch_size, series_length, device,
                 dropout=0.1, num_layers=2):
        super(LSTM_shallow, self).__init__()

        # Number of features
        self.num_features = num_features

        # Hidden dimensions
        self.hidden_dim = hidden_dim

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
            input_size=self.num_features,
            hidden_size=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.num_layers)

        # Defining the Dense Layers
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self, batch_size):
        """Initialised the hidden state to be zeros. This clears the hidden
        state between batches. If you are running a stateful LSTM then this
        needs to be changed.

        To change to a stateful LSTM requires not detaching the backprop and
        storing the computational graph. This strongly increases runtime and
        shouldn't make a big difference. Hence a stateful LSTM was not used.

        :param batch_size:          The batch size to be zeroed
        :type  batch_size:          string
        """
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            self.device)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            self.device)
        return c0, h0

    def forward(self, x):
        """Forward pass through the neural network

        :param x:          The input into the network
        :type  x:          torch.Tensor
        """
        # Adjust to a variable batch size
        batch_size = x.size()[0]
        series_length = x.size()[1]

        # Making sure the series and batch haven't been mis-permuted
        assert series_length == self.series_length

        # Keeps the dimensions constant regardless of batch size
        x = x.contiguous().view(series_length, batch_size, -1)

        # Initialises the hidden states
        h0, c0 = self.init_hidden(batch_size)

        # Pass through through LSTM layer
        # Only the x is of interest
        x, (hn, cn) = self.lstm(x, (h0, c0))

        # Output is seq to seq but only want seq to val
        # So only use the final slice of the LSTM outputted sequence
        x = x[-1]

        # Passes staight from LSTM through to prediction layer
        x = self.fc(x)

        return x


class LSTM_deeper(nn.Module):
    """A Long Short Term Memory network model with two additional dense layers

    :param num_features:      The number of features in the dataset
    :type  num_features:      int

    :param hidden_dim:        The number of neurons in the LSTMs hidden layer/s
    :type  hidden_dim:        int

    :param dense_hidden:      The number of neurons in the first dense layer
    :type  dense_hidden:      int

    :param dense_hidden_2:    The number of neurons in the second dense layer
    :type  dense_hidden_2:    int

    :param output_dim:        The number of neurons in the output layer
    :type  output_dim:        int

    :param batch_size:        The number of items in each batch
    :type  batch_size:        int

    :param series_length:     The length of the time series
    :type  series_length:     Int

    :param device:            The device to run on (Cpu or CUDA)
    :type  device:            string

    :param dropout:           The probability of dropout
    :type  dropout:           float

    :param num_layers:        The number of stacked LSTM layers
    :type  num_layers:        int
    """
    def __init__(self, num_features, hidden_dim, dense_hidden, dense_hidden_2, output_dim,
                 batch_size, series_length, device,
                 dropout=0.1, num_layers=2):
        super(LSTM_deeper, self).__init__()

        # Number of features
        self.num_features = num_features

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of neurons in the dense layer
        self.dense_hidden = dense_hidden

        # Number of neurons in the dense layer
        self.dense_hidden_2 = dense_hidden_2

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
            input_size=self.num_features,
            hidden_size=self.hidden_dim,
            dropout=self.dropout,
            num_layers=self.num_layers)

        # Defining the Dense Layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dense_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dense_hidden, self.dense_hidden_2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dense_hidden_2, self.output_dim),)

    def init_hidden(self, batch_size):
        """Initialised the hidden state to be zeros. This clears the hidden
        state between batches. If you are running a stateful LSTM then this
        needs to be changed.

        To change to a stateful LSTM requires not detaching the backprop and
        storing the computational graph. This strongly increases runtime and
        shouldn't make a big difference. Hence a stateful LSTM was not used.

        :param batch_size:          The batch size to be zeroed
        :type  batch_size:          string
        """
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            self.device)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(
            self.device)
        return c0, h0

    def forward(self, x):
        """Forward pass through the neural network

        :param x:          The input into the network
        :type  x:          torch.Tensor
        """
        # Adjust to a variable batch size
        batch_size = x.size()[0]
        series_length = x.size()[1]

        # Making sure the series and batch haven't been mis-permuted
        assert series_length == self.series_length

        # Keeps the dimensions constant regardless of batch size
        x = x.contiguous().view(series_length, batch_size, -1)

        # Initialises the hidden states
        h0, c0 = self.init_hidden(batch_size)

        # Pass through through LSTM layer
        # Only the x is of interest
        x, (hn, cn) = self.lstm(x, (h0, c0))

        # Output is seq to seq but only want seq to val
        # So only use the final slice of the LSTM outputted sequence
        x = x[-1]

        # Fully connected dense layers with dropout
        x = self.fc(x)

        return x
