# GPS Trajectory Location Prediction System

A comprehensive deep learning pipeline for predicting locations based on GPS trajectory data using PyTorch. This system implements a four-phase approach combining data preprocessing, semantic embeddings, multi-head CNN feature extraction, and SE-BiGRU prediction.

## 🚀 Features

- **Complete Pipeline**: End-to-end solution from raw GPS data to location predictions
- **Advanced Architecture**: Multi-head CNN + SE-BiGRU with attention mechanisms
- **Semantic Processing**: TF-IDF weighted GloVe embeddings for location semantics
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and model analysis
- **Production Ready**: Modular design with proper error handling and documentation

## 📋 System Architecture

### Phase 1: Data Preprocessing
- **Stay Point Detection**: Identifies significant locations using time (5min) and distance (200m) thresholds
- **POI Matching**: Maps GPS coordinates to semantic labels using k-nearest neighbors
- **User Filtering**: Ensures data quality by filtering users with minimum 200 stay points

### Phase 2: Semantic Embedding
- **TF-IDF Weighting**: Computes location importance using formula: `Sl,t = Tf × log(lf/Q)`
- **GloVe Integration**: Combines TF-IDF weights with 200-dimensional GloVe vectors
- **Sequence Processing**: Handles variable-length trajectories with padding/truncation

### Phase 3: Multi-head CNN Feature Extraction
- **Parallel Processing**: Multiple 1D convolutional heads with different kernel sizes [3,5,7,9]
- **Temporal Features**: Captures patterns at various time scales
- **Batch Normalization**: Improves training stability and convergence

### Phase 4: SE-BiGRU Location Prediction
- **Bidirectional GRU**: Processes sequences in both forward and backward directions
- **Squeeze-and-Excitation**: Channel attention mechanism for feature refinement
- **Final Prediction**: Softmax classification over location vocabulary

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/gps-trajectory-prediction.git
cd gps-trajectory-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data Requirements

### Geolife Dataset Format
```csv
user_id,latitude,longitude,timestamp
user_001,39.9042,116.4074,2023-01-01 08:00:00
user_001,39.9045,116.4078,2023-01-01 08:05:00
...
```

### POI Dataset Format
```csv
latitude,longitude,category,name
39.9040,116.4070,restaurant,restaurant_001
39.9050,116.4080,shopping,mall_001
...
```

### GloVe Embeddings
Download pre-trained GloVe embeddings (e.g., `glove.6B.200d.txt`) from [Stanford NLP](https://nlp.stanford.edu/projects/glove/).

## 🚀 Quick Start

```python
from gps_trajectory_prediction import *
import torch
import pandas as pd

# 1. Load configuration
config = ConfigManager('config.yaml')

# 2. Load data
data_loader = DataLoader()
geolife_data = data_loader.load_geolife_data('data/geolife.csv')
poi_data = data_loader.load_poi_data('data/poi.csv')

# 3. Preprocessing
stay_detector = StayPointDetector(time_threshold=5.0, distance_threshold=200.0)
poi_matcher = POIMatcher(poi_data, k=1, max_distance=500.0)
preprocessor = TrajectoryPreprocessor(stay_detector, poi_matcher, min_stay_points=200)

trajectory_sequences = preprocessor.process_dataset(geolife_data)
vocabulary = preprocessor.get_vocabulary(trajectory_sequences)

# 4. Semantic embedding
tfidf_weighter = TFIDFWeighter()
glove_integrator = GloVeIntegrator('data/glove.6B.200d.txt', embedding_dim=200)
glove_integrator.load_glove_embeddings()

semantic_embedding = SemanticEmbedding(tfidf_weighter, glove_integrator, max_sequence_length=100)
embedded_sequences = semantic_embedding.fit_transform([seq.semantic_labels for seq in trajectory_sequences])

# 5. Model creation and training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LocationPredictor(embedding_dim=200, num_locations=len(vocabulary))

trainer = Trainer(model, device, learning_rate=0.001)
history = trainer.train(train_loader, val_loader, epochs=100)

# 6. Evaluation
evaluator = Evaluator(model, device)
metrics = evaluator.evaluate(test_loader, class_names=list(vocabulary.keys()))
evaluator.metrics_calculator.print_metrics(metrics)
```

## 📈 Example Usage

Run the complete demonstration:

```bash
python example_usage.py
```

This script will:
1. Generate sample GPS trajectory and POI data
2. Process the data through all pipeline phases
3. Train the location prediction model
4. Evaluate performance with comprehensive metrics
5. Generate visualizations and save results

## 🔧 Configuration

The system uses YAML configuration files for easy customization:

```yaml
data:
  max_sequence_length: 100
  min_stay_points: 200
  
preprocessing:
  stay_point:
    time_threshold: 5.0
    distance_threshold: 200.0
  poi_matching:
    k: 1
    max_distance: 500.0

model:
  cnn:
    num_heads: 4
    kernel_sizes: [3, 5, 7, 9]
    num_filters: 64
  gru:
    hidden_dim: 128
    num_layers: 2

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
```

## 📊 Model Performance

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision/Recall/F1**: Per-class and macro/weighted averages
- **Confusion Matrix**: Detailed classification results
- **Training Curves**: Loss and accuracy progression

## 🎯 Key Components

### Core Classes

- `TrajectoryPreprocessor`: Main preprocessing pipeline
- `SemanticEmbedding`: TF-IDF + GloVe embedding integration
- `LocationPredictor`: Complete neural network model
- `Trainer`: Training loop with validation and checkpointing
- `Evaluator`: Comprehensive model evaluation

### Utility Classes

- `DataLoader`: Data loading and preprocessing utilities
- `Visualizer`: Trajectory and results visualization
- `ConfigManager`: Configuration management
- `MetricsCalculator`: Evaluation metrics computation

## 🔍 Model Architecture Details

### Multi-head CNN
- **Input**: Embedded trajectory sequences (batch_size, seq_length, embedding_dim)
- **Processing**: Parallel 1D convolutions with different kernel sizes
- **Output**: Concatenated features from all heads

### SE-BiGRU
- **Bidirectional GRU**: Captures temporal dependencies in both directions
- **Squeeze-and-Excitation**: 
  - Squeeze: Global average pooling
  - Excite: Two FC layers with ReLU and Sigmoid
  - Rescale: Channel-wise attention weighting

### Final Prediction
- **Input**: Final hidden state from SE-BiGRU
- **Processing**: Multi-layer classifier with dropout
- **Output**: Softmax probabilities over location vocabulary

## 📝 Research Applications

This system is suitable for:

- **Location Prediction**: Next location prediction in GPS trajectories
- **Mobility Analysis**: Understanding human movement patterns
- **Urban Planning**: Analyzing traffic flows and popular destinations
- **Recommendation Systems**: Location-based recommendations
- **Anomaly Detection**: Identifying unusual movement patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Zheng, Y., et al. "GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory." IEEE Data Engineering Bulletin, 2010.
2. Pennington, J., et al. "GloVe: Global Vectors for Word Representation." EMNLP, 2014.
3. Hu, J., et al. "Squeeze-and-Excitation Networks." CVPR, 2018.
4. Cho, K., et al. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." EMNLP, 2014.

## 🆘 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the example usage script

## 🔄 Version History

- **v1.0.0**: Initial release with complete pipeline implementation
- **v1.1.0**: Added visualization utilities and improved documentation
- **v1.2.0**: Enhanced model architecture with attention mechanisms

---

**Note**: This is a research implementation. For production use, consider additional optimizations for scalability and performance.