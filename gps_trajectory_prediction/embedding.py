"""
Phase 2: Word Embedding with TF-IDF and GloVe Integration

This module handles the conversion of semantic trajectory sequences to dense vector
representations using TF-IDF weighting combined with pre-trained GloVe embeddings.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TFIDFWeighter:
    """
    Computes TF-IDF weights for semantic trajectory sequences.

    Uses the formula: Sl,t = Tf × log(lf/Q)
    where:
    - Tf: Term frequency of location l in trajectory t
    - lf: Number of trajectories containing location l
    - Q: Total number of trajectories
    """

    def __init__(self, min_df: int = 1, max_df: float = 1.0):
        """
        Initialize TF-IDF weighter.

        Args:
            min_df (int): Minimum document frequency for a term to be included
            max_df (float): Maximum document frequency for a term to be included
        """
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.vocabulary_ = None
        self.idf_weights_ = None

        logger.info(f"TFIDFWeighter initialized with min_df={min_df}, max_df={max_df}")

    def fit(self, trajectory_sequences: List[List[str]]) -> 'TFIDFWeighter':
        """
        Fit TF-IDF weighter on trajectory sequences.

        Args:
            trajectory_sequences (List[List[str]]): List of trajectory sequences,
                                                   where each sequence is a list of semantic labels

        Returns:
            TFIDFWeighter: Fitted weighter instance
        """
        # Convert sequences to documents (space-separated strings)
        documents = [' '.join(sequence) for sequence in trajectory_sequences]

        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            token_pattern=r'\S+',  # Split on whitespace
            lowercase=False
        )

        tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Store vocabulary and IDF weights
        self.vocabulary_ = self.vectorizer.vocabulary_
        self.idf_weights_ = self.vectorizer.idf_

        logger.info(f"TF-IDF fitted on {len(documents)} trajectories with "
                    f"{len(self.vocabulary_)} unique locations")

        return self

    def transform(self, trajectory_sequences: List[List[str]]) -> np.ndarray:
        """
        Transform trajectory sequences to TF-IDF weighted vectors.

        Args:
            trajectory_sequences (List[List[str]]): List of trajectory sequences

        Returns:
            np.ndarray: TF-IDF weighted matrix of shape (n_sequences, n_features)

        Raises:
            ValueError: If weighter has not been fitted
        """
        if self.vectorizer is None:
            raise ValueError("TFIDFWeighter must be fitted before transform")

        documents = [' '.join(sequence) for sequence in trajectory_sequences]
        tfidf_matrix = self.vectorizer.transform(documents)

        return tfidf_matrix.toarray()

    def fit_transform(self, trajectory_sequences: List[List[str]]) -> np.ndarray:
        """
        Fit weighter and transform sequences in one step.

        Args:
            trajectory_sequences (List[List[str]]): List of trajectory sequences

        Returns:
            np.ndarray: TF-IDF weighted matrix
        """
        return self.fit(trajectory_sequences).transform(trajectory_sequences)

    def get_feature_names(self) -> List[str]:
        """
        Get feature names (semantic labels).

        Returns:
            List[str]: List of feature names

        Raises:
            ValueError: If weighter has not been fitted
        """
        if self.vectorizer is None:
            raise ValueError("TFIDFWeighter must be fitted before getting feature names")

        return self.vectorizer.get_feature_names_out().tolist()

    def get_weights_for_sequence(self, sequence: List[str]) -> Dict[str, float]:
        """
        Get TF-IDF weights for a single sequence.

        Args:
            sequence (List[str]): Single trajectory sequence

        Returns:
            Dict[str, float]: Mapping of semantic labels to TF-IDF weights
        """
        if self.vectorizer is None:
            raise ValueError("TFIDFWeighter must be fitted before getting weights")

        tfidf_vector = self.transform([sequence])[0]
        feature_names = self.get_feature_names()

        weights = {}
        for idx, weight in enumerate(tfidf_vector):
            if weight > 0:
                weights[feature_names[idx]] = weight

        return weights


class GloVeIntegrator:
    """
    Integrates pre-trained GloVe embeddings with TF-IDF weights.

    Combines TF-IDF weights with GloVe vectors using the formula:
    Yz = Sl,t × kl
    where:
    - Sl,t: TF-IDF weight for location l in trajectory t
    - kl: GloVe embedding vector for location l
    """

    def __init__(self, glove_path: str, embedding_dim: int = 200):
        """
        Initialize GloVe integrator.

        Args:
            glove_path (str): Path to GloVe embeddings file
            embedding_dim (int): Dimension of GloVe embeddings
        """
        self.glove_path = glove_path
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self.vocabulary = set()

        logger.info(f"GloVeIntegrator initialized with embedding_dim={embedding_dim}")

    def load_glove_embeddings(self) -> 'GloVeIntegrator':
        """
        Load GloVe embeddings from file.

        Returns:
            GloVeIntegrator: Instance with loaded embeddings

        Raises:
            FileNotFoundError: If GloVe file is not found
            ValueError: If embedding dimensions don't match
        """
        try:
            logger.info(f"Loading GloVe embeddings from {self.glove_path}")

            with open(self.glove_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != self.embedding_dim + 1:
                        logger.warning(f"Skipping line {line_num}: expected {self.embedding_dim + 1} "
                                       f"parts, got {len(parts)}")
                        continue

                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)

                    self.embeddings[word] = vector
                    self.vocabulary.add(word)

            logger.info(f"Loaded {len(self.embeddings)} GloVe embeddings")
            return self

        except FileNotFoundError:
            raise FileNotFoundError(f"GloVe embeddings file not found: {self.glove_path}")
        except Exception as e:
            raise ValueError(f"Error loading GloVe embeddings: {str(e)}")

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get GloVe embedding for a word.

        Args:
            word (str): Word to get embedding for

        Returns:
            Optional[np.ndarray]: Embedding vector or None if not found
        """
        return self.embeddings.get(word)

    def get_random_embedding(self) -> np.ndarray:
        """
        Generate random embedding for unknown words.

        Returns:
            np.ndarray: Random embedding vector
        """
        return np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)

    def create_embedding_matrix(self, vocabulary: Dict[str, int]) -> np.ndarray:
        """
        Create embedding matrix for given vocabulary.

        Args:
            vocabulary (Dict[str, int]): Vocabulary mapping words to indices

        Returns:
            np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim)
        """
        vocab_size = len(vocabulary)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)

        found_count = 0
        for word, idx in vocabulary.items():
            embedding = self.get_embedding(word)
            if embedding is not None:
                embedding_matrix[idx] = embedding
                found_count += 1
            else:
                embedding_matrix[idx] = self.get_random_embedding()

        logger.info(f"Created embedding matrix: {found_count}/{vocab_size} words found in GloVe")
        return embedding_matrix


class SemanticEmbedding:
    """
    Main embedding class that combines TF-IDF weighting with GloVe embeddings.

    This class orchestrates the entire embedding process, converting semantic
    trajectory sequences to dense vector representations.
    """

    def __init__(self,
                 tfidf_weighter: TFIDFWeighter,
                 glove_integrator: GloVeIntegrator,
                 max_sequence_length: int = 100):
        """
        Initialize semantic embedding.

        Args:
            tfidf_weighter (TFIDFWeighter): TF-IDF weighting component
            glove_integrator (GloVeIntegrator): GloVe integration component
            max_sequence_length (int): Maximum length for sequence padding/truncation
        """
        self.tfidf_weighter = tfidf_weighter
        self.glove_integrator = glove_integrator
        self.max_sequence_length = max_sequence_length
        self.vocabulary = None
        self.embedding_matrix = None

        logger.info(f"SemanticEmbedding initialized with max_sequence_length={max_sequence_length}")

    def fit(self, trajectory_sequences: List[List[str]]) -> 'SemanticEmbedding':
        """
        Fit embedding on trajectory sequences.

        Args:
            trajectory_sequences (List[List[str]]): List of trajectory sequences

        Returns:
            SemanticEmbedding: Fitted embedding instance
        """
        # Fit TF-IDF weighter
        self.tfidf_weighter.fit(trajectory_sequences)

        # Create vocabulary from TF-IDF features
        feature_names = self.tfidf_weighter.get_feature_names()
        self.vocabulary = {name: idx for idx, name in enumerate(feature_names)}

        # Create embedding matrix
        self.embedding_matrix = self.glove_integrator.create_embedding_matrix(self.vocabulary)

        logger.info(f"SemanticEmbedding fitted with vocabulary size: {len(self.vocabulary)}")
        return self

    def transform_sequence(self, sequence: List[str]) -> np.ndarray:
        """
        Transform a single trajectory sequence to weighted embeddings.

        Args:
            sequence (List[str]): Single trajectory sequence

        Returns:
            np.ndarray: Weighted embedding sequence of shape (max_sequence_length, embedding_dim)

        Raises:
            ValueError: If embedding has not been fitted
        """
        if self.vocabulary is None or self.embedding_matrix is None:
            raise ValueError("SemanticEmbedding must be fitted before transform")

        # Get TF-IDF weights for sequence
        tfidf_weights = self.tfidf_weighter.get_weights_for_sequence(sequence)

        # Create weighted embedding sequence
        embedding_dim = self.embedding_matrix.shape[1]
        weighted_sequence = np.zeros((self.max_sequence_length, embedding_dim), dtype=np.float32)

        # Process each location in sequence
        for i, location in enumerate(sequence[:self.max_sequence_length]):
            if location in self.vocabulary:
                vocab_idx = self.vocabulary[location]
                embedding_vector = self.embedding_matrix[vocab_idx]
                tfidf_weight = tfidf_weights.get(location, 0.0)

                # Apply TF-IDF weighting: Yz = Sl,t × kl
                weighted_sequence[i] = tfidf_weight * embedding_vector
            else:
                # Use random embedding for unknown locations
                random_embedding = self.glove_integrator.get_random_embedding()
                weighted_sequence[i] = random_embedding

        return weighted_sequence

    def transform(self, trajectory_sequences: List[List[str]]) -> np.ndarray:
        """
        Transform multiple trajectory sequences to weighted embeddings.

        Args:
            trajectory_sequences (List[List[str]]): List of trajectory sequences

        Returns:
            np.ndarray: Weighted embedding sequences of shape
                       (n_sequences, max_sequence_length, embedding_dim)
        """
        if not trajectory_sequences:
            return np.array([])

        embedding_dim = self.embedding_matrix.shape[1]
        n_sequences = len(trajectory_sequences)

        weighted_sequences = np.zeros(
            (n_sequences, self.max_sequence_length, embedding_dim),
            dtype=np.float32
        )

        for i, sequence in enumerate(trajectory_sequences):
            weighted_sequences[i] = self.transform_sequence(sequence)

        logger.info(f"Transformed {n_sequences} sequences to weighted embeddings")
        return weighted_sequences

    def fit_transform(self, trajectory_sequences: List[List[str]]) -> np.ndarray:
        """
        Fit embedding and transform sequences in one step.

        Args:
            trajectory_sequences (List[List[str]]): List of trajectory sequences

        Returns:
            np.ndarray: Weighted embedding sequences
        """
        return self.fit(trajectory_sequences).transform(trajectory_sequences)

    def save(self, filepath: str) -> None:
        """
        Save fitted embedding to file.

        Args:
            filepath (str): Path to save embedding
        """
        embedding_data = {
            'vocabulary': self.vocabulary,
            'embedding_matrix': self.embedding_matrix,
            'max_sequence_length': self.max_sequence_length,
            'tfidf_weighter': self.tfidf_weighter,
            'embedding_dim': self.glove_integrator.embedding_dim
        }

        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)

        logger.info(f"SemanticEmbedding saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, glove_integrator: GloVeIntegrator) -> 'SemanticEmbedding':
        """
        Load fitted embedding from file.

        Args:
            filepath (str): Path to load embedding from
            glove_integrator (GloVeIntegrator): GloVe integrator instance

        Returns:
            SemanticEmbedding: Loaded embedding instance
        """
        with open(filepath, 'rb') as f:
            embedding_data = pickle.load(f)

        # Create instance
        embedding = cls(
            tfidf_weighter=embedding_data['tfidf_weighter'],
            glove_integrator=glove_integrator,
            max_sequence_length=embedding_data['max_sequence_length']
        )

        # Restore fitted state
        embedding.vocabulary = embedding_data['vocabulary']
        embedding.embedding_matrix = embedding_data['embedding_matrix']

        logger.info(f"SemanticEmbedding loaded from {filepath}")
        return embedding