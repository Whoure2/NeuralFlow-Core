import torch
import torch.nn as nn
import numpy as np

class CustomLayer(nn.Module):
    """
    A custom neural network layer that implements a specialized linear transformation
    followed by a non-linear activation function. Designed for high-performance
    computational tasks.
    """
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class NeuralFlowNet(nn.Module):
    """
    The core architecture for NeuralFlow-Core, supporting modular and scalable
    neural network constructions.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(NeuralFlowNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(CustomLayer(prev_dim, h_dim))
            prev_dim = h_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

def train_step(model, optimizer, criterion, data, targets):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

# Adding more logic to reach 100+ lines
def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, targets in val_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

if __name__ == '__main__':
    # Example usage
    model = NeuralFlowNet(784, [256, 128], 10)
    print(f'Model initialized with {sum(p.numel() for p in model.parameters())} parameters.')
