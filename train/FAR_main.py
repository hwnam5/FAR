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
"""
1. Individual CNN 1D Architecture

각 센서 채널별로 독립적인 1D CNN을 적용하여 채널별 특성을 추출합니다.

Tensor Shape 변화:
    Input:  [batch_size, num_channels, seq_length]
    Split:  [batch_size, 1, seq_length] (각 채널별로 분할)
    CNN:    [batch_size, num_filters, seq_length] (각 채널별 출력)
    Concat: [batch_size, num_channels * num_filters, seq_length] (채널 결합)
"""
class Individual_CNN_1D(nn.Module):
    def __init__(self, input_size: int = 1, filter_size: int = 32, kernel_size: int = 5,
                 stride: int = 2, num_layers: int = 4):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        self.filter_size = filter_size
        padding = kernel_size // 2
        layers = []
        
        layers += [
            nn.Conv1d(
                in_channels=self.input_size,
                out_channels=self.filter_size,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=padding
            ),
            nn.BatchNorm1d(self.filter_size),
            nn.ReLU()
        ]
        
        for _ in range(self.num_layers - 1):
            layers += [
                nn.Conv1d(
                    in_channels=self.filter_size,
                    out_channels=self.filter_size,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=padding
                ),
                nn.BatchNorm1d(self.filter_size),
                nn.ReLU()
            ]
        
        self.layers = nn.Sequential(*layers)
            
    def forward(self, one_channel_x):
        # one_channel_x : [batch_size, 1, seq_length]
        x = self.layers(one_channel_x)
        return x
        
class TinyHAR(nn.Module):
    def __init__(self, input_size: int = Config.INPUT_CHANNELS):
        super().__init__()
        
        for i in range(input_size):
            self.cnn4channels[i] = Individual_CNN_1D(input_size=1, filter_size=32, kernel_size=5, stride=2, num_layers=4)
            
    def forward(self, x):
        # x : [batch_size, num_channels, seq_length]
        cnn_outputs = []
        for i in range(Config.INPUT_CHANNELS):
            cnn_output = self.cnn4channels[i](x[:, i, :])
            cnn_outputs.append(cnn_output)
        cnn_outputs = torch.cat(cnn_outputs, dim=1)
        return cnn_outputs
        
        
        

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