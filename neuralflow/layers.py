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
\n# Maintenance log 1\n# Maintenance log 2\n# Maintenance log 7\n# Maintenance log 8\n# Maintenance log 9\n# Maintenance log 10\n# Maintenance log 11\n# Maintenance log 12\n# Maintenance log 13\n# Maintenance log 14\n# Maintenance log 16\n# Maintenance log 17\n# Maintenance log 18\n# Maintenance log 19\n# Maintenance log 21\n# Maintenance log 22\n# Maintenance log 23