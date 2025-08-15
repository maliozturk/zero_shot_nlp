"""
Example Usage of GPS Trajectory Location Prediction System

This script demonstrates how to use the complete GPS trajectory location prediction
pipeline from data loading to model training and evaluation.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any

# Import our modules
from gps_trajectory_prediction import (
    TrajectoryPreprocessor, StayPointDetector, POIMatcher,
    SemanticEmbedding, TFIDFWeighter, GloVeIntegrator,
    LocationPredictor, Trainer, Evaluator,
    DataLoader, Visualizer, ConfigManager
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data() -> tuple:
    """
    Generate sample GPS trajectory data for demonstration.

    Returns:
        tuple: (geolife_data, poi_data) - Sample datasets
    """
    logger.info("Generating sample data...")

    # Generate sample GPS trajectory data
    np.random.seed(42)
    n_users = 50
    n_points_per_user = 500

    geolife_data = []

    # Beijing area coordinates (approximate)
    base_lat, base_lon = 39.9042, 116.4074

    for user_id in range(n_users):
        # Generate trajectory for each user
        user_lat = base_lat + np.random.normal(0, 0.1)
        user_lon = base_lon + np.random.normal(0, 0.1)

        for i in range(n_points_per_user):
            # Random walk around user's base location
            lat = user_lat + np.random.normal(0, 0.01)
            lon = user_lon + np.random.normal(0, 0.01)
            timestamp = pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=i * 5)

            geolife_data.append({
                'user_id': f'user_{user_id:03d}',
                'latitude': lat,
                'longitude': lon,
                'timestamp': timestamp
            })

    geolife_df = pd.DataFrame(geolife_data)

    # Generate sample POI data
    poi_categories = ['restaurant', 'shopping', 'entertainment', 'transport', 'education']
    poi_data = []

    for i in range(200):
        category = np.random.choice(poi_categories)
        lat = base_lat + np.random.normal(0, 0.05)
        lon = base_lon + np.random.normal(0, 0.05)

        poi_data.append({
            'latitude': lat,
            'longitude': lon,
            'category': category,
            'name': f'{category}_{i:03d}'
        })

    poi_df = pd.DataFrame(poi_data)

    logger.info(f"Generated {len(geolife_df)} GPS points and {len(poi_df)} POI locations")
    return geolife_df, poi_df


def create_sample_glove_embeddings(vocab_size: int = 1000, embedding_dim: int = 200) -> str:
    """
    Create sample GloVe embeddings file for demonstration.

    Args:
        vocab_size (int): Number of vocabulary items
        embedding_dim (int): Embedding dimension

    Returns:
        str: Path to created embeddings file
    """
    logger.info("Creating sample GloVe embeddings...")

    embeddings_path = "sample_glove.txt"

    # Generate sample vocabulary
    poi_categories = ['restaurant', 'shopping', 'entertainment', 'transport', 'education', 'unknown']
    vocab = []

    # Add POI-related words
    for category in poi_categories:
        for i in range(50):
            vocab.append(f"{category}_{i:03d}")

    # Add some common words
    common_words = ['location', 'place', 'area', 'building', 'center', 'station', 'mall', 'park']
    vocab.extend(common_words)

    # Pad to vocab_size
    while len(vocab) < vocab_size:
        vocab.append(f"word_{len(vocab)}")

    # Generate random embeddings
    np.random.seed(42)

    with open(embeddings_path, 'w') as f:
        for word in vocab[:vocab_size]:
            embedding = np.random.normal(0, 0.1, embedding_dim)
            embedding_str = ' '.join([f'{x:.6f}' for x in embedding])
            f.write(f"{word} {embedding_str}\n")

    logger.info(f"Created sample GloVe embeddings: {embeddings_path}")
    return embeddings_path


def main():
    """
    Main function demonstrating the complete pipeline.
    """
    logger.info("Starting GPS Trajectory Location Prediction Demo")

    # 1. Configuration Management
    logger.info("=" * 60)
    logger.info("1. CONFIGURATION SETUP")
    logger.info("=" * 60)

    config_manager = ConfigManager()

    # Update some configuration values
    config_manager.set('training.epochs', 20)  # Reduced for demo
    config_manager.set('data.max_sequence_length', 50)  # Reduced for demo
    config_manager.set('data.min_stay_points', 20)  # Reduced for demo

    logger.info("Configuration loaded and updated")

    # 2. Data Generation and Loading
    logger.info("=" * 60)
    logger.info("2. DATA GENERATION AND LOADING")
    logger.info("=" * 60)

    # Generate sample data
    geolife_data, poi_data = generate_sample_data()
    glove_path = create_sample_glove_embeddings()

    # 3. Data Preprocessing
    logger.info("=" * 60)
    logger.info("3. DATA PREPROCESSING")
    logger.info("=" * 60)

    # Initialize preprocessing components
    stay_point_detector = StayPointDetector(
        time_threshold=config_manager.get('preprocessing.stay_point.time_threshold'),
        distance_threshold=config_manager.get('preprocessing.stay_point.distance_threshold')
    )

    poi_matcher = POIMatcher(
        poi_data=poi_data,
        k=config_manager.get('preprocessing.poi_matching.k'),
        max_distance=config_manager.get('preprocessing.poi_matching.max_distance')
    )

    trajectory_preprocessor = TrajectoryPreprocessor(
        stay_point_detector=stay_point_detector,
        poi_matcher=poi_matcher,
        min_stay_points=config_manager.get('data.min_stay_points')
    )

    # Process trajectories
    trajectory_sequences = trajectory_preprocessor.process_dataset(geolife_data)
    vocabulary = trajectory_preprocessor.get_vocabulary(trajectory_sequences)

    logger.info(f"Processed {len(trajectory_sequences)} trajectory sequences")
    logger.info(f"Vocabulary size: {len(vocabulary)}")

    # 4. Semantic Embedding
    logger.info("=" * 60)
    logger.info("4. SEMANTIC EMBEDDING")
    logger.info("=" * 60)

    # Initialize embedding components
    tfidf_weighter = TFIDFWeighter(
        min_df=config_manager.get('embedding.tfidf.min_df'),
        max_df=config_manager.get('embedding.tfidf.max_df')
    )

    glove_integrator = GloVeIntegrator(
        glove_path=glove_path,
        embedding_dim=config_manager.get('embedding.embedding_dim')
    )
    glove_integrator.load_glove_embeddings()

    semantic_embedding = SemanticEmbedding(
        tfidf_weighter=tfidf_weighter,
        glove_integrator=glove_integrator,
        max_sequence_length=config_manager.get('data.max_sequence_length')
    )

    # Prepare sequences for embedding
    semantic_sequences = [seq.semantic_labels for seq in trajectory_sequences]

    # Fit and transform sequences
    embedded_sequences = semantic_embedding.fit_transform(semantic_sequences)

    logger.info(f"Embedded sequences shape: {embedded_sequences.shape}")

    # 5. Prepare Labels and Data Loaders
    logger.info("=" * 60)
    logger.info("5. DATA PREPARATION")
    logger.info("=" * 60)

    # Create labels (next location prediction)
    # For demonstration, we'll predict the last location in each sequence
    labels = []
    sequences_for_training = []

    for i, seq in enumerate(semantic_sequences):
        if len(seq) > 1:
            # Use all but last location as input, last location as target
            input_seq = seq[:-1]
            target_location = seq[-1]

            if target_location in vocabulary:
                labels.append(vocabulary[target_location])
                sequences_for_training.append(embedded_sequences[i])

    # Convert to numpy arrays
    X = np.array(sequences_for_training)
    y = np.array(labels)

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Number of classes: {len(vocabulary)}")

    # Create data loaders
    data_loader = DataLoader()
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(
        sequences=X,
        labels=y,
        train_ratio=config_manager.get('training.train_ratio'),
        val_ratio=config_manager.get('training.val_ratio'),
        test_ratio=config_manager.get('training.test_ratio'),
        batch_size=config_manager.get('training.batch_size')
    )

    # 6. Model Creation
    logger.info("=" * 60)
    logger.info("6. MODEL CREATION")
    logger.info("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = LocationPredictor(
        embedding_dim=config_manager.get('embedding.embedding_dim'),
        num_locations=len(vocabulary),
        cnn_config=config_manager.get('model.cnn'),
        gru_config=config_manager.get('model.gru'),
        dropout_rate=config_manager.get('model.classifier.dropout_rate')
    )

    # Print model info
    model_info = model.get_model_info()
    logger.info(f"Model created with {model_info['total_parameters']} parameters")

    # 7. Training
    logger.info("=" * 60)
    logger.info("7. MODEL TRAINING")
    logger.info("=" * 60)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config_manager.get('training.learning_rate'),
        weight_decay=config_manager.get('training.weight_decay'),
        scheduler_config=config_manager.get('training.scheduler')
    )

    # Early stopping
    from gps_trajectory_prediction.training import EarlyStopping
    early_stopping = EarlyStopping(
        patience=config_manager.get('training.early_stopping.patience'),
        min_delta=config_manager.get('training.early_stopping.min_delta')
    )

    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config_manager.get('training.epochs'),
        early_stopping=early_stopping,
        checkpoint_dir=config_manager.get('evaluation.checkpoint_dir')
    )

    # 8. Evaluation
    logger.info("=" * 60)
    logger.info("8. MODEL EVALUATION")
    logger.info("=" * 60)

    # Initialize evaluator
    evaluator = Evaluator(model=model, device=device)

    # Evaluate on test set
    class_names = list(vocabulary.keys())
    test_metrics = evaluator.evaluate(test_loader, class_names=class_names)

    # Print metrics
    evaluator.metrics_calculator.print_metrics(test_metrics)

    # 9. Visualization
    logger.info("=" * 60)
    logger.info("9. VISUALIZATION")
    logger.info("=" * 60)

    # Plot training history
    fig_history = evaluator.plot_training_history(history)
    fig_history.savefig('training_history.png', dpi=300, bbox_inches='tight')
    logger.info("Training history plot saved: training_history.png")

    # Plot confusion matrix
    fig_cm = evaluator.plot_confusion_matrix(
        test_metrics['confusion_matrix'],
        class_names=class_names[:20]  # Show only first 20 classes for readability
    )
    fig_cm.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    logger.info("Confusion matrix plot saved: confusion_matrix.png")

    # Visualize sample trajectories
    visualizer = Visualizer()

    # Plot stay point distribution
    fig_dist = visualizer.plot_stay_point_distribution(trajectory_sequences)
    fig_dist.savefig('stay_point_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("Stay point distribution plot saved: stay_point_distribution.png")

    # Plot POI category distribution
    fig_poi = visualizer.plot_poi_category_distribution(trajectory_sequences)
    fig_poi.savefig('poi_category_distribution.png', dpi=300, bbox_inches='tight')
    logger.info("POI category distribution plot saved: poi_category_distribution.png")

    # 10. Model Saving
    logger.info("=" * 60)
    logger.info("10. MODEL SAVING")
    logger.info("=" * 60)

    # Save model
    trainer.save_model('final_model.pth')

    # Save embedding
    semantic_embedding.save('semantic_embedding.pkl')

    # Save configuration
    config_manager.save_config('final_config.yaml')

    logger.info("Model, embedding, and configuration saved")

    # 11. Sample Prediction
    logger.info("=" * 60)
    logger.info("11. SAMPLE PREDICTION")
    logger.info("=" * 60)

    # Make prediction on a sample
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        sample_sequences, sample_labels = sample_batch
        sample_sequences = sample_sequences.to(device)

        predictions = model.predict(sample_sequences)
        probabilities = model(sample_sequences)['probabilities']

        # Show first few predictions
        for i in range(min(5, len(predictions))):
            pred_idx = predictions[i].item()
            true_idx = sample_labels[i].item()
            confidence = probabilities[i, pred_idx].item()

            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
            true_label = class_names[true_idx] if true_idx < len(class_names) else f"class_{true_idx}"

            logger.info(f"Sample {i + 1}: Predicted={pred_label} (confidence={confidence:.3f}), "
                        f"True={true_label}")

    logger.info("=" * 60)
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    # Clean up sample files
    import os
    try:
        os.remove(glove_path)
        logger.info("Cleaned up sample files")
    except:
        pass


if __name__ == "__main__":
    main()