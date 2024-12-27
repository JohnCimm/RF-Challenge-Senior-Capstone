import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=False, long_kernel=False):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5 if dropout else 0.25)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        return x

class UNet_py(nn.Module):
    def __init__(self, input_shape, k_sz=3, long_k_sz=101, lr=0.0003, k_neurons=32):
        super(UNet, self).__init__()
        
        self.k_sz = k_sz
        self.long_k_sz = long_k_sz
        self.k_neurons = k_neurons
        
        # Encoder
        self.enc_blocks = nn.ModuleList([
            ConvBlock(1, k_neurons * 8, long_k_sz, dropout=True),
            ConvBlock(k_neurons * 8, k_neurons * 8, k_sz, dropout=True),
            ConvBlock(k_neurons * 16, k_neurons * 8, k_sz, dropout=True),
            ConvBlock(k_neurons * 32, k_neurons * 4, k_sz, dropout=True),
            ConvBlock(k_neurons * 64, k_neurons * 1, k_sz, dropout=False),
        ])
        
        # MaxPooling and Middle block
        self.max_pool = nn.MaxPool1d(2)
        self.middle_block = ConvBlock(k_neurons * 8, k_neurons * 8, k_sz)
        
        # Decoder with ConvTranspose for upsampling
        self.dec_blocks = nn.ModuleList([
            ConvBlock(k_neurons * 16, k_neurons * 8, k_sz),
            ConvBlock(k_neurons * 16, k_neurons * 4, k_sz),
            ConvBlock(k_neurons * 8, k_neurons * 2, k_sz),
            ConvBlock(k_neurons * 4, k_neurons * 1, k_sz),
        ])
        
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose1d(k_neurons * 8, k_neurons * 8, k_sz, stride=2, padding=k_sz//2),
            nn.ConvTranspose1d(k_neurons * 8, k_neurons * 4, k_sz, stride=2, padding=k_sz//2),
            nn.ConvTranspose1d(k_neurons * 4, k_neurons * 2, k_sz, stride=2, padding=k_sz//2),
            nn.ConvTranspose1d(k_neurons * 2, k_neurons * 1, k_sz, stride=2, padding=k_sz//2),
        ])
        
        self.output_layer = nn.Conv1d(k_neurons * 1, 2, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # Adjust input for PyTorch (N, C, L)
        enc_outputs = []
        
        # Encoder path
        for enc_block in self.enc_blocks:
            x = enc_block(x)
            enc_outputs.append(x)
            x = self.max_pool(x)

        # Middle block
        x = self.middle_block(x)

        # Decoder path
        for i, dec_block in enumerate(self.dec_blocks):
            x = self.upsample_layers[i](x)
            x = torch.cat([x, enc_outputs[-(i + 1)]], dim=1)
            x = dec_block(x)
        
        x = self.output_layer(x)
        return x.transpose(1, 2)  # Switch back to (N, L, C)
