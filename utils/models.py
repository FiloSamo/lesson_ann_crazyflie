from torch import nn

# Model
class NeuralNetwork(nn.Module):
    '''
        Simple Neural network for MNIST classification
    
    '''
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))      
        return out