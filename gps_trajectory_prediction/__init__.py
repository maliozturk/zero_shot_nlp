"""
GPS Trajectory Location Prediction System

A comprehensive deep learning pipeline for predicting locations based on GPS trajectory data.
This system implements a four-phase approach:
1. Data Preprocessing and Semantic Trajectory Generation
2. Word Embedding with TF-IDF and GloVe
3. Multi-head CNN Feature Extraction
4. SE-BiGRU Location Prediction

Author: AI Assistant
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .data_preprocessing import TrajectoryPreprocessor, StayPointDetector, POIMatcher
from .embedding import SemanticEmbedding, TFIDFWeighter, GloVeIntegrator
from .models import MultiHeadCNN, SEBiGRU, LocationPredictor
from .utils import DataLoader, Visualizer, ConfigManager
from .training import Trainer, Evaluator

__all__ = [
    'TrajectoryPreprocessor',
    'StayPointDetector', 
    'POIMatcher',
    'SemanticEmbedding',
    'TFIDFWeighter',
    'GloVeIntegrator',
    'MultiHeadCNN',
    'SEBiGRU',
    'LocationPredictor',
    'DataLoader',
    'Visualizer',
    'ConfigManager',
    'Trainer',
    'Evaluator'
]