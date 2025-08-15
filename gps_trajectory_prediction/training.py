"""
Training and Evaluation Components

This module provides comprehensive training and evaluation functionality for the
GPS trajectory location prediction model, including:
- Training loop with validation
- Model evaluation metrics
- Learning rate scheduling
- Early stopping
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
import time
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.

    Monitors validation loss and stops training when no improvement
    is observed for a specified number of epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.

        Args:
            patience (int): Number of epochs to wait for improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

        logger.info(f"EarlyStopping initialized: patience={patience}, min_delta={min_delta}")

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.

        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to potentially save weights from

        Returns:
            bool: True if training should be stopped
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best model weights")

        return self.early_stop


class MetricsCalculator:
    """
    Calculates various evaluation metrics for location prediction.
    """

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (Optional[List[str]]): Names of classes for detailed metrics

        Returns:
            Dict[str, Any]: Dictionary containing various metrics
        """
        # Basic accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support
        }

        if class_names is not None:
            metrics['class_names'] = class_names

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, Any]) -> None:
        """
        Print formatted metrics.

        Args:
            metrics (Dict[str, Any]): Metrics dictionary from calculate_metrics
        """
        print("\n" + "=" * 50)
        print("EVALUATION METRICS")
        print("=" * 50)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")

        if 'class_names' in metrics:
            print("\nPer-class metrics:")
            print("-" * 70)
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<10}")
            print("-" * 70)

            for i, class_name in enumerate(metrics['class_names']):
                print(f"{class_name:<20} {metrics['per_class_precision'][i]:<12.4f} "
                      f"{metrics['per_class_recall'][i]:<12.4f} "
                      f"{metrics['per_class_f1'][i]:<12.4f} "
                      f"{metrics['per_class_support'][i]:<10}")


class Trainer:
    """
    Main training class for the location prediction model.

    Handles the complete training process including validation,
    metrics calculation, and model checkpointing.
    """

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 scheduler_config: Optional[Dict[str, Any]] = None):
        """
        Initialize trainer.

        Args:
            model (nn.Module): Model to train
            device (torch.device): Device to train on
            learning_rate (float): Initial learning rate
            weight_decay (float): Weight decay for regularization
            scheduler_config (Optional[Dict[str, Any]]): Learning rate scheduler configuration
        """
        self.model = model.to(device)
        self.device = device

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = None
        if scheduler_config is not None:
            scheduler_type = scheduler_config.get('type', 'StepLR')
            if scheduler_type == 'StepLR':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 10),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=scheduler_config.get('factor', 0.5),
                    patience=scheduler_config.get('patience', 5),
                    verbose=True
                )

        # Training history
        self.history = defaultdict(list)

        logger.info(f"Trainer initialized: lr={learning_rate}, weight_decay={weight_decay}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): Training data loader

        Returns:
            Dict[str, float]: Training metrics for the epoch
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs['logits'], targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            predictions = torch.argmax(outputs['probabilities'], dim=1)
            correct_predictions += (predictions == targets).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            Dict[str, float]: Validation metrics for the epoch
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs['logits'], targets)

                # Update metrics
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

                predictions = torch.argmax(outputs['probabilities'], dim=1)
                correct_predictions += (predictions == targets).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 100,
              early_stopping: Optional[EarlyStopping] = None,
              checkpoint_dir: Optional[str] = None,
              save_best_only: bool = True) -> Dict[str, List[float]]:
        """
        Complete training loop.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (Optional[DataLoader]): Validation data loader
            epochs (int): Number of epochs to train
            early_stopping (Optional[EarlyStopping]): Early stopping callback
            checkpoint_dir (Optional[str]): Directory to save checkpoints
            save_best_only (bool): Whether to save only the best model

        Returns:
            Dict[str, List[float]]: Training history
        """
        logger.info(f"Starting training for {epochs} epochs")

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader)

            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])

            if val_loader is not None:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_accuracy'].append(val_metrics['accuracy'])

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader is not None:
                        self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Logging
            epoch_time = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch + 1}/{epochs} ({epoch_time:.2f}s) - "
            log_msg += f"train_loss: {train_metrics['loss']:.4f}, "
            log_msg += f"train_acc: {train_metrics['accuracy']:.4f}"

            if val_loader is not None:
                log_msg += f", val_loss: {val_metrics['loss']:.4f}, "
                log_msg += f"val_acc: {val_metrics['accuracy']:.4f}"

            logger.info(log_msg)

            # Model checkpointing
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)

                current_val_loss = val_metrics.get('loss', train_metrics['loss'])

                if not save_best_only or current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': current_val_loss,
                        'history': dict(self.history)
                    }, checkpoint_path)
                    logger.info(f"Saved best model checkpoint: {checkpoint_path}")

            # Early stopping
            if early_stopping is not None and val_loader is not None:
                if early_stopping(val_metrics['loss'], self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        return dict(self.history)

    def save_model(self, filepath: str) -> None:
        """
        Save model state.

        Args:
            filepath (str): Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': dict(self.history)
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load model state.

        Args:
            filepath (str): Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'history' in checkpoint:
            self.history = defaultdict(list, checkpoint['history'])

        logger.info(f"Model loaded from {filepath}")


class Evaluator:
    """
    Model evaluation class with comprehensive metrics and visualization.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize evaluator.

        Args:
            model (nn.Module): Trained model to evaluate
            device (torch.device): Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.metrics_calculator = MetricsCalculator()

        logger.info("Evaluator initialized")

    def evaluate(self, test_loader: DataLoader,
                 class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate model on test data.

        Args:
            test_loader (DataLoader): Test data loader
            class_names (Optional[List[str]]): Names of classes

        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                predictions = torch.argmax(outputs['probabilities'], dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(outputs['probabilities'].cpu().numpy())

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            class_names
        )

        # Add probability information
        metrics['probabilities'] = np.array(all_probabilities)
        metrics['predictions'] = np.array(all_predictions)
        metrics['targets'] = np.array(all_targets)

        logger.info(f"Evaluation completed: accuracy={metrics['accuracy']:.4f}")

        return metrics

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            confusion_matrix (np.ndarray): Confusion matrix
            class_names (Optional[List[str]]): Names of classes
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()
        return fig

    def plot_training_history(self, history: Dict[str, List[float]],
                              figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
        """
        Plot training history.

        Args:
            history (Dict[str, List[float]]): Training history
            figsize (Tuple[int, int]): Figure size

        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Loss plot
        ax1.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(history['train_accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig