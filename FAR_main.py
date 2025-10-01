import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import h5py
from typing import Dict, List, Tuple, Optional
import warnings
import math
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.signal import cheby1, lfilter, lfilter_zi
from torch.nn import DataParallel
from sklearn.metrics import confusion_matrix, classification_report

# =============================================================================
# Configuration and Setup
# =============================================================================

class Config:
    """Configuration class for model training parameters"""
    INPUT_CHANNELS = 3 # IMU_ACC, IMU_GYRO, Bio-impedance
    OUTPUT_CHANNELS = 4 # Num of Fitness States

def setup_logger() -> logging.Logger:
    """Setup and configure logger"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler only
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def setup_cudnn():
    """Setup cuDNN optimization settings"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def create_directory_structure(model_type: str):
    """Create necessary directories if they don't exist"""
    directories = [
        'models/classification',
    ]
    for directory in directories:
        if model_type is not None:
            os.makedirs(os.path.join(directory, model_type), exist_ok=True)
        else:
            os.makedirs(directory, exist_ok=True)

# =============================================================================
# Dataset Classes
# =============================================================================

# =============================================================================
# Classification Model Architecture Classes
# =============================================================================

# TinyHAR
class Temporal_Attention(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, sensor_channel, hidden_dim):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.weighs_activation = nn.Tanh() 
        self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)
        self.sm = torch.nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):

        # batch  sensor_channel feature_dim
        #B C F

        out = self.weighs_activation(self.fc_1(x))

        out = self.fc_2(out).squeeze(2)

        weights_att = self.sm(out).unsqueeze(2)

        context = torch.sum(weights_att * x, 1)
        context = x[:, -1, :] + self.gamma * context # 마지막 시점에 전역 요약을 더해줌

        return context

class TinyHAR(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 128, 
                 num_layers: int = 2, output_size: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)

        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=dropout,
            batch_first=True)
        
        self.temporal_attention = Temporal_Attention(sensor_channel=input_size, hidden_dim=hidden_size // 2)

        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)

        x = x.permute(0, 2, 1)
        x = self.self_attention(x)

        x = self.flatten(x)
        x = self.fc1(x)

        x = self.lstm(x)

        x = self.temporal_attention(x)
        x = self.fc2(x)

        return x

# =============================================================================
# Utility Functions
# =============================================================================

def setup_device() -> torch.device:
    """Setup and return device (GPU/CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("💻 Using CPU")
    
    return device

def setup_multi_gpu(model: nn.Module, device: torch.device) -> Tuple[nn.Module, int]:
    """Setup multi-GPU training if available"""
    num_gpus = torch.cuda.device_count()
    
    if num_gpus > 1:
        logger.info(f"🚀 Multi-GPU detected: {num_gpus} GPUs available")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        model = DataParallel(model)
        effective_batch_size = Config.BATCH_SIZE * num_gpus
        logger.info(f"📊 Effective batch size: {effective_batch_size} (batch_size: {Config.BATCH_SIZE} × GPUs: {num_gpus})")
    else:
        effective_batch_size = Config.BATCH_SIZE
    
    return model, effective_batch_size

# =============================================================================
# Training and Testing
# =============================================================================

# =============================================================================
# Main Function
# =============================================================================
def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="FAR Training and Testing")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()