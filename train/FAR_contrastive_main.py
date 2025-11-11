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
    SOURCE_INPUT_CHANNELS = 1 # Bio-impedance
    TARGET_INPUT_CHANNELS = 2 # IMU_ACC, IMU_GYRO
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
        'models/contrastive',
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

# encoder
class Encoder(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 128, 
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

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = self.self_attention(x, x, x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.lstm(x)
        return x

# translator
class Translator_block(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        x = self.self_attention(x, x, x)
        x = self.ffn(x)
        return x

class Translator(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        self.translator_blocks = nn.ModuleList([Translator_block(input_size, hidden_size, dropout) for _ in range(num_layers)])
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        for translator_block in self.translator_blocks:
            x = translator_block(x)

        x = self.ffn(x)

        return x

# classifier TinyHAR
class TinyHAR_classifier(nn.Module):
    """
    Temporal attention module
    """
    def __init__(self, sensor_channel, hidden_dim, output_size):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.weighs_activation = nn.Tanh() 
        self.fc_2 = nn.Linear(hidden_dim, 1, bias=False)
        self.sm = torch.nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

        self.fc_3 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        # batch  sensor_channel feature_dim
        #B C F

        out = self.weighs_activation(self.fc_1(x))

        out = self.fc_2(out).squeeze(2)

        weights_att = self.sm(out).unsqueeze(2)

        context = torch.sum(weights_att * x, 1)
        context = x[:, -1, :] + self.gamma * context # 마지막 시점에 전역 요약을 더해줌

        result = self.fc_3(context)

        return result

# =============================================================================
# contrastive loss
# =============================================================================

class Contrastive_loss(nn.Module):
    def __init__(self, alpha, tau):
        super().__init__()
        self.alpha = alpha
        self.tau = tau

    def forward(self, Rt, s2r):
        Rt = F.normalize(Rt, dim=-1)
        s2r = F.normalize(s2r, dim=-1)
        
        Batch_size = Rt.size(0)
        Label_size = s2r.size(1)

        logits = torch.matmul(Rt, s2r.transpose(1, 2)) / self.tau
        targets = torch.arange(Label_size).to(logits.device).unsqueeze(0).expand(Batch_size, -1)

        return F.cross_entropy(logits.reshape(Batch_size * Label_size, Label_size), targets.reshape(Batch_size * Label_size)) #(input, target)

class Classification_loss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, Ps, Pt, y):

        return self.ce(Ps, y) + self.alpha * self.ce(Pt, y)

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
        


