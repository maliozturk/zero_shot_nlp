"""
Utility Functions and Classes

This module provides utility functions for data loading, visualization,
configuration management, and other helper functions.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import json
import yaml
import logging
from pathlib import Path
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory sequences.

    Handles loading and preprocessing of trajectory data for training.
    """

    def __init__(self,
                 sequences: np.ndarray,
                 labels: np.ndarray,
                 transform: Optional[callable] = None):
        """
        Initialize trajectory dataset.

        Args:
            sequences (np.ndarray): Trajectory sequences of shape (n_samples, seq_length, feature_dim)
            labels (np.ndarray): Target labels of shape (n_samples,)
            transform (Optional[callable]): Optional transform to apply to sequences
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

        logger.info(f"TrajectoryDataset initialized: {len(self.sequences)} samples, "
                    f"sequence shape: {self.sequences.shape[1:]}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.

        Args:
            idx (int): Sample index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (sequence, label)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]

        if self.transform:
            sequence = self.transform(sequence)

        return sequence, label


class DataLoader:
    """
    Data loading utilities for the Geolife dataset and trajectory processing.
    """

    @staticmethod
    def load_geolife_data(data_path: str,
                          user_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load Geolife dataset from file.

        Args:
            data_path (str): Path to Geolife dataset file
            user_limit (Optional[int]): Limit number of users to load

        Returns:
            pd.DataFrame: Loaded dataset with columns ['user_id', 'latitude', 'longitude', 'timestamp']

        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If required columns are missing
        """
        try:
            logger.info(f"Loading Geolife data from {data_path}")

            # Determine file format and load accordingly
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.pkl'):
                df = pd.read_pickle(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")

            # Validate required columns
            required_columns = ['user_id', 'latitude', 'longitude', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Convert timestamp to datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Limit users if specified
            if user_limit is not None:
                unique_users = df['user_id'].unique()[:user_limit]
                df = df[df['user_id'].isin(unique_users)]

            logger.info(f"Loaded {len(df)} GPS points from {df['user_id'].nunique()} users")
            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Geolife data file not found: {data_path}")
        except Exception as e:
            raise ValueError(f"Error loading Geolife data: {str(e)}")

    @staticmethod
    def load_poi_data(poi_path: str) -> pd.DataFrame:
        """
        Load POI (Points of Interest) data.

        Args:
            poi_path (str): Path to POI dataset file

        Returns:
            pd.DataFrame: POI data with columns ['latitude', 'longitude', 'category', 'name']

        Raises:
            FileNotFoundError: If POI file is not found
            ValueError: If required columns are missing
        """
        try:
            logger.info(f"Loading POI data from {poi_path}")

            if poi_path.endswith('.csv'):
                df = pd.read_csv(poi_path)
            elif poi_path.endswith('.json'):
                df = pd.read_json(poi_path)
            else:
                raise ValueError(f"Unsupported POI file format: {poi_path}")

            # Validate required columns
            required_columns = ['latitude', 'longitude', 'category', 'name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required POI columns: {missing_columns}")

            logger.info(f"Loaded {len(df)} POI locations with {df['category'].nunique()} categories")
            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"POI data file not found: {poi_path}")
        except Exception as e:
            raise ValueError(f"Error loading POI data: {str(e)}")

    @staticmethod
    def create_data_loaders(sequences: np.ndarray,
                            labels: np.ndarray,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            batch_size: int = 32,
                            shuffle: bool = True,
                            random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.

        Args:
            sequences (np.ndarray): Trajectory sequences
            labels (np.ndarray): Target labels
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
            test_ratio (float): Ratio of data for testing
            batch_size (int): Batch size for data loaders
            shuffle (bool): Whether to shuffle training data
            random_seed (int): Random seed for reproducibility

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

        # Set random seed
        np.random.seed(random_seed)

        # Create indices and shuffle
        n_samples = len(sequences)
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        # Split indices
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # Create datasets
        train_dataset = TrajectoryDataset(sequences[train_indices], labels[train_indices])
        val_dataset = TrajectoryDataset(sequences[val_indices], labels[val_indices])
        test_dataset = TrajectoryDataset(sequences[test_indices], labels[test_indices])

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        logger.info(f"Created data loaders: train={len(train_dataset)}, "
                    f"val={len(val_dataset)}, test={len(test_dataset)}")

        return train_loader, val_loader, test_loader


class Visualizer:
    """
    Visualization utilities for trajectory data and model results.
    """

    @staticmethod
    def plot_trajectory_on_map(trajectory_sequence,
                               center_lat: Optional[float] = None,
                               center_lon: Optional[float] = None,
                               zoom_start: int = 12) -> folium.Map:
        """
        Plot trajectory sequence on an interactive map.

        Args:
            trajectory_sequence: TrajectorySequence object
            center_lat (Optional[float]): Center latitude for map
            center_lon (Optional[float]): Center longitude for map
            zoom_start (int): Initial zoom level

        Returns:
            folium.Map: Interactive map with trajectory
        """
        coordinates = trajectory_sequence.get_coordinates()

        # Calculate center if not provided
        if center_lat is None or center_lon is None:
            center_lat = coordinates[:, 0].mean()
            center_lon = coordinates[:, 1].mean()

        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )

        # Add trajectory points
        for i, (stay_point, label) in enumerate(zip(trajectory_sequence.stay_points,
                                                    trajectory_sequence.semantic_labels)):
            # Color based on POI category
            color = Visualizer._get_poi_color(label)

            folium.CircleMarker(
                location=[stay_point.latitude, stay_point.longitude],
                radius=8,
                popup=f"Point {i + 1}: {label}<br>Duration: {stay_point.duration:.1f} min",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)

        # Add trajectory line
        trajectory_coords = [[lat, lon] for lat, lon in coordinates]
        folium.PolyLine(
            trajectory_coords,
            color='blue',
            weight=3,
            opacity=0.8
        ).add_to(m)

        return m

    @staticmethod
    def _get_poi_color(poi_label: str) -> str:
        """
        Get color for POI category.

        Args:
            poi_label (str): POI label

        Returns:
            str: Color string
        """
        color_map = {
            'restaurant': 'red',
            'shopping': 'blue',
            'entertainment': 'green',
            'transport': 'purple',
            'education': 'orange',
            'healthcare': 'pink',
            'unknown': 'gray'
        }

        # Extract category from label
        category = poi_label.split('_')[0].lower() if '_' in poi_label else poi_label.lower()
        return color_map.get(category, 'gray')

    @staticmethod
    def plot_stay_point_distribution(trajectory_sequences: List,
                                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot distribution of stay points across users.

        Args:
            trajectory_sequences (List): List of TrajectorySequence objects
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Matplotlib figure
        """
        stay_point_counts = [len(seq) for seq in trajectory_sequences]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        ax1.hist(stay_point_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Number of Stay Points')
        ax1.set_ylabel('Number of Users')
        ax1.set_title('Distribution of Stay Points per User')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(stay_point_counts)
        ax2.set_ylabel('Number of Stay Points')
        ax2.set_title('Stay Points Distribution (Box Plot)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_poi_category_distribution(trajectory_sequences: List,
                                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot distribution of POI categories.

        Args:
            trajectory_sequences (List): List of TrajectorySequence objects
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Matplotlib figure
        """
        all_labels = []
        for seq in trajectory_sequences:
            all_labels.extend(seq.semantic_labels)

        # Extract categories
        categories = [label.split('_')[0] if '_' in label else label for label in all_labels]
        category_counts = pd.Series(categories).value_counts()

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(range(len(category_counts)), category_counts.values,
                      color='lightcoral', alpha=0.8)
        ax.set_xlabel('POI Category')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of POI Categories')
        ax.set_xticks(range(len(category_counts)))
        ax.set_xticklabels(category_counts.index, rotation=45, ha='right')

        # Add value labels on bars
        for bar, value in zip(bars, category_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(category_counts),
                    str(value), ha='center', va='bottom')

        plt.tight_layout()
        return fig


class ConfigManager:
    """
    Configuration management for the GPS trajectory prediction system.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path (Optional[str]): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_default_config()

        if config_path is not None:
            self.load_config(config_path)

    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'data': {
                'geolife_path': 'data/geolife.csv',
                'poi_path': 'data/poi.csv',
                'glove_path': 'data/glove.6B.200d.txt',
                'max_sequence_length': 100,
                'min_stay_points': 200,
                'user_limit': None
            },
            'preprocessing': {
                'stay_point': {
                    'time_threshold': 5.0,
                    'distance_threshold': 200.0
                },
                'poi_matching': {
                    'k': 1,
                    'max_distance': 500.0
                }
            },
            'embedding': {
                'embedding_dim': 200,
                'tfidf': {
                    'min_df': 1,
                    'max_df': 1.0
                }
            },
            'model': {
                'cnn': {
                    'num_heads': 4,
                    'kernel_sizes': [3, 5, 7, 9],
                    'num_filters': 64,
                    'dropout_rate': 0.2
                },
                'gru': {
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'dropout_rate': 0.2,
                    'se_reduction_ratio': 16
                },
                'classifier': {
                    'dropout_rate': 0.3
                }
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'epochs': 100,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'early_stopping': {
                    'patience': 10,
                    'min_delta': 0.001
                },
                'scheduler': {
                    'type': 'ReduceLROnPlateau',
                    'factor': 0.5,
                    'patience': 5
                }
            },
            'evaluation': {
                'checkpoint_dir': 'checkpoints',
                'results_dir': 'results'
            }
        }

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.

        Args:
            config_path (str): Path to configuration file

        Raises:
            FileNotFoundError: If config file is not found
            ValueError: If config file format is invalid
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    user_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")

            # Merge with default config
            self._merge_config(self.config, user_config)
            logger.info(f"Configuration loaded from {config_path}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")

    def _merge_config(self, default: Dict[str, Any], user: Dict[str, Any]) -> None:
        """
        Recursively merge user configuration with default.

        Args:
            default (Dict[str, Any]): Default configuration
            user (Dict[str, Any]): User configuration
        """
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value

    def save_config(self, save_path: str) -> None:
        """
        Save current configuration to file.

        Args:
            save_path (str): Path to save configuration
        """
        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif save_path.endswith('.json'):
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported save format: {save_path}")

        logger.info(f"Configuration saved to {save_path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path (str): Dot-separated key path (e.g., 'model.cnn.num_heads')
            default (Any): Default value if key not found

        Returns:
            Any: Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key_path (str): Dot-separated key path
            value (Any): Value to set
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value