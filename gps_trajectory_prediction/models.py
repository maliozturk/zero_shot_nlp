"""
Phase 3 & 4: Multi-head CNN Feature Extraction and SE-BiGRU Location Prediction

This module contains the deep learning models for feature extraction and location prediction:
- MultiHeadCNN: Parallel 1D convolutional layers for temporal feature extraction
- SEBiGRU: Bidirectional GRU with Squeeze-and-Excitation attention mechanism
- LocationPredictor: Main model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MultiHeadCNN(nn.Module):
    """
    Multi-head 1D CNN for temporal feature extraction from trajectory sequences.

    This module applies multiple parallel 1D convolutional layers with different
    kernel sizes to capture temporal patterns at various scales.
    """

    def __init__(self,
                 input_dim: int,
                 num_heads: int = 4,
                 kernel_sizes: Optional[List[int]] = None,
                 num_filters: int = 64,
                 dropout_rate: float = 0.2):
        """
        Initialize Multi-head CNN.

        Args:
            input_dim (int): Input feature dimension (embedding dimension)
            num_heads (int): Number of parallel CNN heads
            kernel_sizes (Optional[List[int]]): Kernel sizes for each head
            num_filters (int): Number of filters per head
            dropout_rate (float): Dropout rate for regularization
        """
        super(MultiHeadCNN, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        # Default kernel sizes if not provided
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9][:num_heads]

        if len(kernel_sizes) != num_heads:
            raise ValueError(f"Number of kernel sizes ({len(kernel_sizes)}) must match "
                             f"number of heads ({num_heads})")

        self.kernel_sizes = kernel_sizes

        # Create parallel CNN heads
        self.conv_heads = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for kernel_size in kernel_sizes:
            # 1D Convolutional layer
            conv = nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding
            )

            # Batch normalization
            batch_norm = nn.BatchNorm1d(num_filters)

            self.conv_heads.append(conv)
            self.batch_norms.append(batch_norm)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Output dimension after concatenation
        self.output_dim = num_heads * num_filters

        logger.info(f"MultiHeadCNN initialized: {num_heads} heads, "
                    f"kernel_sizes={kernel_sizes}, filters={num_filters}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, output_dim)
        """
        batch_size, seq_length, input_dim = x.shape

        # Transpose for 1D convolution: (batch_size, input_dim, seq_length)
        x = x.transpose(1, 2)

        # Apply each CNN head
        head_outputs = []
        for i, (conv, batch_norm) in enumerate(zip(self.conv_heads, self.batch_norms)):
            # Convolution
            conv_out = conv(x)  # (batch_size, num_filters, seq_length)

            # Batch normalization
            norm_out = batch_norm(conv_out)

            # ReLU activation
            activated = F.relu(norm_out)

            # Transpose back: (batch_size, seq_length, num_filters)
            head_output = activated.transpose(1, 2)
            head_outputs.append(head_output)

        # Concatenate all heads along feature dimension
        concatenated = torch.cat(head_outputs, dim=2)  # (batch_size, seq_length, output_dim)

        # Apply dropout
        output = self.dropout(concatenated)

        return output

    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation module for channel attention.

    This module applies global average pooling (squeeze) followed by
    two fully connected layers (excite) to learn channel-wise attention weights.
    """

    def __init__(self, input_dim: int, reduction_ratio: int = 16):
        """
        Initialize Squeeze-and-Excitation module.

        Args:
            input_dim (int): Input feature dimension
            reduction_ratio (int): Reduction ratio for bottleneck layer
        """
        super(SqueezeExcitation, self).__init__()

        self.input_dim = input_dim
        self.reduction_ratio = reduction_ratio
        self.reduced_dim = max(1, input_dim // reduction_ratio)

        # Squeeze: Global average pooling is applied in forward pass

        # Excite: Two fully connected layers
        self.fc1 = nn.Linear(input_dim, self.reduced_dim)
        self.fc2 = nn.Linear(self.reduced_dim, input_dim)

        logger.debug(f"SqueezeExcitation initialized: input_dim={input_dim}, "
                     f"reduced_dim={self.reduced_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Squeeze-and-Excitation module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        batch_size, seq_length, input_dim = x.shape

        # Squeeze: Global average pooling along sequence dimension
        squeezed = torch.mean(x, dim=1)  # (batch_size, input_dim)

        # Excite: Two FC layers with ReLU and Sigmoid
        excited = F.relu(self.fc1(squeezed))  # (batch_size, reduced_dim)
        attention_weights = torch.sigmoid(self.fc2(excited))  # (batch_size, input_dim)

        # Expand attention weights to match input shape
        attention_weights = attention_weights.unsqueeze(1)  # (batch_size, 1, input_dim)

        # Apply channel-wise rescaling
        output = x * attention_weights  # Broadcasting: (batch_size, seq_length, input_dim)

        return output


class SEBiGRU(nn.Module):
    """
    Bidirectional GRU with Squeeze-and-Excitation attention mechanism.

    This module combines bidirectional GRU for sequential modeling with
    SE attention for feature refinement.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 se_reduction_ratio: int = 16):
        """
        Initialize SE-BiGRU.

        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden dimension for GRU
            num_layers (int): Number of GRU layers
            dropout_rate (float): Dropout rate for regularization
            se_reduction_ratio (int): Reduction ratio for SE module
        """
        super(SEBiGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output dimension after bidirectional GRU
        self.gru_output_dim = 2 * hidden_dim

        # Squeeze-and-Excitation module
        self.se_module = SqueezeExcitation(
            input_dim=self.gru_output_dim,
            reduction_ratio=se_reduction_ratio
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        logger.info(f"SEBiGRU initialized: hidden_dim={hidden_dim}, "
                    f"num_layers={num_layers}, output_dim={self.gru_output_dim}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SE-BiGRU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Sequence output: (batch_size, seq_length, 2*hidden_dim)
                - Final hidden state: (batch_size, 2*hidden_dim)
        """
        batch_size, seq_length, input_dim = x.shape

        # Bidirectional GRU
        gru_output, gru_hidden = self.bigru(x)
        # gru_output: (batch_size, seq_length, 2*hidden_dim)
        # gru_hidden: (2*num_layers, batch_size, hidden_dim)

        # Apply Squeeze-and-Excitation attention
        se_output = self.se_module(gru_output)

        # Apply dropout
        se_output = self.dropout(se_output)

        # Get final hidden state (concatenate forward and backward)
        # Take the last layer's hidden states
        forward_hidden = gru_hidden[-2, :, :]  # (batch_size, hidden_dim)
        backward_hidden = gru_hidden[-1, :, :]  # (batch_size, hidden_dim)
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch_size, 2*hidden_dim)

        return se_output, final_hidden

    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.gru_output_dim


class LocationPredictor(nn.Module):
    """
    Main location prediction model combining all components.

    This model integrates:
    1. Multi-head CNN for temporal feature extraction
    2. SE-BiGRU for sequential modeling with attention
    3. Final prediction layers with softmax activation
    """

    def __init__(self,
                 embedding_dim: int,
                 num_locations: int,
                 cnn_config: Optional[Dict[str, Any]] = None,
                 gru_config: Optional[Dict[str, Any]] = None,
                 dropout_rate: float = 0.3):
        """
        Initialize Location Predictor.

        Args:
            embedding_dim (int): Input embedding dimension
            num_locations (int): Number of possible locations (output classes)
            cnn_config (Optional[Dict[str, Any]]): Configuration for CNN component
            gru_config (Optional[Dict[str, Any]]): Configuration for GRU component
            dropout_rate (float): Dropout rate for final layers
        """
        super(LocationPredictor, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_locations = num_locations
        self.dropout_rate = dropout_rate

        # Default configurations
        if cnn_config is None:
            cnn_config = {
                'num_heads': 4,
                'kernel_sizes': [3, 5, 7, 9],
                'num_filters': 64,
                'dropout_rate': 0.2
            }

        if gru_config is None:
            gru_config = {
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout_rate': 0.2,
                'se_reduction_ratio': 16
            }

        # Multi-head CNN for feature extraction
        self.cnn = MultiHeadCNN(
            input_dim=embedding_dim,
            **cnn_config
        )

        # SE-BiGRU for sequential modeling
        self.se_bigru = SEBiGRU(
            input_dim=self.cnn.get_output_dim(),
            **gru_config
        )

        # Final prediction layers
        gru_output_dim = self.se_bigru.get_output_dim()

        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, gru_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gru_output_dim // 2, gru_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gru_output_dim // 4, num_locations)
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(f"LocationPredictor initialized: embedding_dim={embedding_dim}, "
                    f"num_locations={num_locations}")

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim)

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'logits': Raw prediction logits (batch_size, num_locations)
                - 'probabilities': Softmax probabilities (batch_size, num_locations)
                - 'cnn_features': CNN features (batch_size, seq_length, cnn_output_dim)
                - 'gru_features': GRU features (batch_size, seq_length, gru_output_dim)
                - 'final_hidden': Final hidden state (batch_size, gru_output_dim)
        """
        # Multi-head CNN feature extraction
        cnn_features = self.cnn(x)  # (batch_size, seq_length, cnn_output_dim)

        # SE-BiGRU sequential modeling
        gru_features, final_hidden = self.se_bigru(cnn_features)
        # gru_features: (batch_size, seq_length, gru_output_dim)
        # final_hidden: (batch_size, gru_output_dim)

        # Final prediction using the last hidden state
        logits = self.classifier(final_hidden)  # (batch_size, num_locations)

        # Apply softmax for probabilities
        probabilities = F.softmax(logits, dim=1)

        return {
            'logits': logits,
            'probabilities': probabilities,
            'cnn_features': cnn_features,
            'gru_features': gru_features,
            'final_hidden': final_hidden
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim)

        Returns:
            torch.Tensor: Predicted class indices (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs['probabilities'], dim=1)
        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model architecture information.

        Returns:
            Dict[str, Any]: Model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_dim': self.embedding_dim,
            'num_locations': self.num_locations,
            'cnn_output_dim': self.cnn.get_output_dim(),
            'gru_output_dim': self.se_bigru.get_output_dim()
        }