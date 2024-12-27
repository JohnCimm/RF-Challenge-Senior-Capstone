import os, sys

import numpy 
import random
import h5py
import argparse

import rfcutils
from torchvision import datasets as tfds

#from torchvision import datasets
#%import tensorflow_datasets as tfds
import torch as tf

import glob, h5py


from src import unet_model as UNet
import torch

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, model, metric_value):
        if not self.save_best_only or \
           (self.mode == 'min' and metric_value < self.best_value) or \
           (self.mode == 'max' and metric_value > self.best_value):
            self.best_value = metric_value
            torch.save(model.state_dict(), self.filepath)
            print(f"Model saved to {self.filepath} with {self.monitor}: {metric_value:.4f}")
class EarlyStopping:
    def __init__(self, monitor='val_loss', patience=5, mode='min', verbose=False):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric_value):
        if (self.mode == 'min' and metric_value < self.best_value) or \
           (self.mode == 'max' and metric_value > self.best_value):
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
#%mirrored_strategy = tf.distribute.MirroredStrategy(devices= ["/gpu:0","/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
bsz = 2


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import tensorflow_datasets as tfds  # Still used to load the dataset
#%from pytorch_unet import UNet  # Assuming you have a PyTorch implementation of the UNet model

all_datasets = ['QPSK_CommSignal2']

def train_script(idx, bsz=2):
    dataset_type = all_datasets[idx]

    # Load dataset using tfds
    dataset = tfds.load(dataset_type, split="train", shuffle_files=True, as_supervised=True,
                        data_dir='/scratch/general/vast/u1110463/tfds/')
    
    # Convert the dataset to PyTorch format
        # Convert the dataset to PyTorch format
    dataset = [(torch.tensor(mixture), torch.tensor(target)) for mixture, target in tfds.as_numpy(dataset)]
    
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bsz, shuffle=False)

    # Define model, loss, and optimizer
    window_len = 4896
    model = UNet(in_channels=2, out_channels=1, init_features=32)  # Customize input/output as needed
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Enable GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    epochs = 20
    patience = 100
    best_val_loss = float('inf')
    patience_counter = 0
    model_pathname = os.path.join('models', f'{dataset_type}_unet', 'checkpoint.pth')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for mixture, target in train_loader:
            mixture, target = mixture.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(mixture)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mixture, target in val_loader:
                mixture, target = mixture.to(device), target.to(device)
                output = model(mixture)
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping and checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_pathname)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
if __name__ == '__main__':
    train_script(int(sys.argv[1]))
