import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A standard residual block as seen in ResNet architectures."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    """A simplified self-attention block for sequence processing."""
    def __init__(self, embed_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm(x)
        return x

class GatedRecurrentUnit(nn.Module):
    """A custom Gated Recurrent Unit (GRU) implementation."""
    def __init__(self, input_size, hidden_size):
        super(GatedRecurrentUnit, self).__init__()
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x, hidden):
        gates = self.linear_ih(x) + self.linear_hh(hidden)
        r_gate, z_gate, n_gate = gates.chunk(3, 1)
        r_gate = torch.sigmoid(r_gate)
        z_gate = torch.sigmoid(z_gate)
        n_gate = torch.tanh(n_gate)
        next_hidden = (1 - z_gate) * hidden + z_gate * n_gate
        return next_hidden, next_hidden

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
\n# Maintenance log 1\n# Maintenance log 2\n# Maintenance log 7\n# Maintenance log 8\n# Maintenance log 9\n# Maintenance log 10\n# Maintenance log 11\n# Maintenance log 12\n# Maintenance log 13\n# Maintenance log 14\n# Maintenance log 16\n# Maintenance log 17\n# Maintenance log 18\n# Maintenance log 19\n# Maintenance log 21\n# Maintenance log 22\n# Maintenance log 23\n# Maintenance log 24\n# Maintenance log 25\n# Maintenance log 26\n# Maintenance log 27\n# Maintenance log 28\n# Maintenance log 29\n# Maintenance log 30\n# Maintenance log 31\n# Maintenance log 32\n# Maintenance log 33\n# Maintenance log 34\n# Maintenance log 35\n# Maintenance log 36\n# Maintenance log 37\n# Maintenance log 39\n# Maintenance log 40\n# Maintenance log 41\n# Maintenance log 42\n# Maintenance log 45\n# Maintenance log 46\n# Maintenance log 48\n# Maintenance log 49\n# Maintenance log 50\n# Maintenance log 51\n# Maintenance log 52\n# Maintenance log 53\n# Maintenance log 54\n# Maintenance log 55\n# Maintenance log 56\n# Maintenance log 57\n# Maintenance log 58\n# Maintenance log 59\n# Maintenance log 61\n# Maintenance log 62\n# Maintenance log 63\n# Maintenance log 65\n# Maintenance log 66\n# Maintenance log 67\n# Maintenance log 68\n# Maintenance log 70\n# Maintenance log 71\n# Maintenance log 72\n# Maintenance log 74\n# Maintenance log 77\n# Maintenance log 78\n# Maintenance log 79\n# Maintenance log 80\n# Maintenance log 81\n# Maintenance log 82\n# Maintenance log 83\n# Maintenance log 87\n# Maintenance log 88\n# Maintenance log 89\n# Maintenance log 91\n# Maintenance log 92\n# Maintenance log 93\n# Maintenance log 94\n# Maintenance log 95