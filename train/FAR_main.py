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
    
""" 
2. Transformer encoder: Cross-channel info interaction

각 채널별로 추출된 특성을 통해 채널 간 정보 상호작용을 학습합니다.
Tensor Shape 변화:
    Input:  [batch_size, num_channels, num_filters, seq_length]
    Transformer: [batch_size * seq_length, num_channels, num_filters]
    Output: [batch_size, num_channels, num_filters, seq_length]
"""
class Cross_channel_Transformer_Encoder(nn.Module):
    def __init__(self, num_channels: int = Config.INPUT_CHANNELS, num_filters: int = 32,
                 seq_length: int = 100, num_encoder_layers: int = 1, num_heads: int = 4):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.seq_length = seq_length
        self.num_encoder_layers = num_encoder_layers
        self.num_heads = num_heads
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.num_filters,
            nhead=self.num_heads,
            dim_feedforward=self.num_filters * self.num_heads,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )
        
    def forward(self, x):
        # x : [B, C, F, T] -> [B * T, C, F]
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B * T, C, F)
        
        x = self.transformer_encoder(x)
        
        # [B * T, C, F] -> [B, T, C, F]
        x = x.reshape(B, T, C, F)
        return x

# 마지막에 각 채널의 독립적인 FC Layer를 적용하여 채널별 특성을 추출합니다.
class Cross_channel_FC_Layer(nn.Module):
    def __init__(self, num_channels: int = 1, num_filters: int = 32):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_filters, self.num_filters * 2),
            nn.BatchNorm1d(self.num_filters * 2),
            nn.ReLU(),
            nn.Linear(self.num_filters * 2, self.num_filters),
        )
        
    def forward(self, x):
        x = self.fc_layer(x)
        return x

"""
3. Fully Connected Layer: Cross-Channel Info Fusion
모든 센서 채널로부터 추출된 특성을 융합합니다. + Bottleneck 역할을 합니다.
Tensor Shape 변화:
    Input:  [batch_size, seq_length, num_channels, num_filters]
    FC:     [batch_size, seq_length, num_channels * num_filters]
    Output: [batch_size, seq_length, 2 * num_filters]
"""
class Channel_Fusion_FC_Layer(nn.Module):
    def __init__(self, num_channels: int = Config.INPUT_CHANNELS, num_filters: int = 32):
        super().__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_channels * self.num_filters, self.num_filters * 2),
            nn.BatchNorm1d(self.num_filters * 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc_layer(x)
        return x

"""
4. One-Layer LSTM: Global Temporal Info Extraction
LSTM을 사용하여 전역적인 시간 정보를 추출합니다.
Tensor Shape 변화:
    Input:  [batch_size, seq_length, 2 * num_filters]
    LSTM
    Output: [batch_size, seq_length, 2 * num_filters]
"""
class Global_Temporal_LSTM(nn.Module):
    def __init__(self, input_size: int = 2 * 32, output_size: int = 2 * 32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.output_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        x = self.lstm(x)

        return x

"""
5.Temporal Attention: Global Temporal Info Enhancement
LSTM 출력의 마지막 시간 축과 각 시간 축의 은닉 상태에 대해 가중 평균 합을 합치는 역할을 합니다.
이를 통해 전역 시간 정보 강화를 합니다.
"""
class Temporal_Attention(nn.Module):
    def __init__(self, input_size: int = 2 * 32):
        super().__init__()
        self.input_size = input_size

        self.attn_weights = nn.Linear(self.input_size, 1)

    def forward(self, x):

        last_hidden_state = x[:, -1, :] # [batch_size, 2 * 32]

        attn_weights = self.attn_weights(x)
        attn_weights = F.softmax(attn_weights, dim=1) # [batch_size, seq_length, 1]

        # 시간별 중요도를 각 시간 feature에 곱하여 합치는 역할
        context = torch.sum(attn_weights * x, dim=1) # [batch_size, 2 * 32]

        enhanced_context = last_hidden_state + context # [batch_size, 2 * 32]

        return enhanced_context



class TinyHAR(nn.Module):
    def __init__(self, input_size: int = Config.INPUT_CHANNELS):
        super().__init__()
        
        for i in range(input_size):
            self.cnn4channels[i] = Individual_CNN_1D(input_size=1, filter_size=32, kernel_size=5, stride=2, num_layers=4)
        
        self.cross_channel_transformer_encoder = Cross_channel_Transformer_Encoder(num_channels=Config.INPUT_CHANNELS, num_filters=32, seq_length=100, num_encoder_layers=1, num_heads=4)
        for i in range(input_size):
            self.cross_channel_fc_layer[i] = Cross_channel_FC_Layer(num_channels=1, num_filters=32)

        self.channel_fusion_fc_layer = Channel_Fusion_FC_Layer(num_channels=Config.INPUT_CHANNELS, num_filters=32)
        
        self.global_temporal_lstm = Global_Temporal_LSTM(input_size=2 * 32, output_size=2 * 32)

    def forward(self, x):
        # x : [batch_size, num_channels, seq_length]
        cnn_outputs = []
        for i in range(Config.INPUT_CHANNELS):
            cnn_output = self.cnn4channels[i](x[:, i, :])
            cnn_outputs.append(cnn_output)
        x = torch.cat(cnn_outputs, dim=1)
        
        x = self.cross_channel_transformer_encoder(x)
        B, T, C, F = x.shape
        
        fc_outputs = []
        for i in range(Config.INPUT_CHANNELS):
            fc_output = self.cross_channel_fc_layer[i](x[:, :, i, :])
            fc_outputs.append(fc_output)
        x = torch.cat(fc_outputs, dim=2)
        
        # [B, T, C, F] -> [B, T, C * F]
        x = x.reshape(B, T, C * F)
        x = self.channel_fusion_fc_layer(x)

        x = self.global_temporal_lstm(x)
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