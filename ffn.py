import torch
import torch.nn as nn


class FFNModel(nn.Module):
    """Feed Forward Neural Network Model.
    A simple feed-forward neural network implementation with 3 fully connected layers
    and ReLU/Sigmoid activation functions.
    Args:
        input_size (int): Size of the input features
        output_size (int): Size of the output layer
        hidden_sizes (list): List of hidden layer sizes. Default is [128, 128]
            First element is size of first hidden layer
            Second element is size of second hidden layer
    Attributes:
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Second fully connected layer 
        fc3 (nn.Linear): Output layer
        sigmoid (nn.Sigmoid): Sigmoid activation function
    Methods:
        forward(x): Forward pass of the network
    """


    def __init__(self, input_size=4542, output_size=2271, hidden_sizes=[1024, 1024]):
        super(FFNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x