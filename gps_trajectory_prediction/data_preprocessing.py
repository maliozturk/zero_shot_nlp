"""
Phase 1: Data Preprocessing and Semantic Trajectory Generation

This module handles the preprocessing of raw GPS trajectory data, including:
- Stay point detection using time and distance thresholds
- User filtering based on minimum stay points
- POI matching using k-nearest neighbors
- Trajectory sequence organization
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from geopy.distance import geodesic
import logging

logger = logging.getLogger(__name__)


@dataclass
class StayPoint:
    """
    Represents a stay point in a GPS trajectory.

    Attributes:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        arrival_time (pd.Timestamp): Time of arrival at this location
        departure_time (pd.Timestamp): Time of departure from this location
        duration (float): Duration of stay in minutes
        poi_label (Optional[str]): Semantic label from POI matching
    """
    latitude: float
    longitude: float
    arrival_time: pd.Timestamp
    departure_time: pd.Timestamp
    duration: float
    poi_label: Optional[str] = None

    @property
    def coordinates(self) -> Tuple[float, float]:
        """Return coordinates as (lat, lon) tuple."""
        return (self.latitude, self.longitude)


@dataclass
class TrajectorySequence:
    """
    Represents a sequence of stay points for a user.

    Attributes:
        user_id (str): Unique identifier for the user
        stay_points (List[StayPoint]): Ordered list of stay points
        semantic_labels (List[str]): Semantic labels for each stay point
    """
    user_id: str
    stay_points: List[StayPoint]
    semantic_labels: List[str]

    def __len__(self) -> int:
        """Return number of stay points in sequence."""
        return len(self.stay_points)

    def get_coordinates(self) -> np.ndarray:
        """Return coordinates as numpy array of shape (n_points, 2)."""
        return np.array([sp.coordinates for sp in self.stay_points])


class StayPointDetector:
    """
    Detects stay points from raw GPS trajectory data using time and distance thresholds.

    A stay point is identified when a user remains within a certain distance threshold
    for a minimum duration of time.
    """

    def __init__(self,
                 time_threshold: float = 5.0,
                 distance_threshold: float = 200.0):
        """
        Initialize stay point detector.

        Args:
            time_threshold (float): Minimum duration in minutes to consider a stay point
            distance_threshold (float): Maximum distance in meters to consider same location
        """
        self.time_threshold = time_threshold
        self.distance_threshold = distance_threshold
        logger.info(f"StayPointDetector initialized with time_threshold={time_threshold}min, "
                    f"distance_threshold={distance_threshold}m")

    def detect_stay_points(self, trajectory_data: pd.DataFrame) -> List[StayPoint]:
        """
        Detect stay points from trajectory data.

        Args:
            trajectory_data (pd.DataFrame): DataFrame with columns ['latitude', 'longitude', 'timestamp']

        Returns:
            List[StayPoint]: List of detected stay points

        Raises:
            ValueError: If required columns are missing from trajectory_data
        """
        required_columns = ['latitude', 'longitude', 'timestamp']
        if not all(col in trajectory_data.columns for col in required_columns):
            raise ValueError(f"trajectory_data must contain columns: {required_columns}")

        if len(trajectory_data) < 2:
            logger.warning("Insufficient data points for stay point detection")
            return []

        stay_points = []
        trajectory_data = trajectory_data.sort_values('timestamp').reset_index(drop=True)

        i = 0
        while i < len(trajectory_data) - 1:
            j = i + 1

            # Find points within distance threshold
            while j < len(trajectory_data):
                distance = self._calculate_distance(
                    (trajectory_data.iloc[i]['latitude'], trajectory_data.iloc[i]['longitude']),
                    (trajectory_data.iloc[j]['latitude'], trajectory_data.iloc[j]['longitude'])
                )

                if distance > self.distance_threshold:
                    break
                j += 1

            # Check if duration exceeds time threshold
            if j > i + 1:  # At least 2 points
                duration = self._calculate_duration(
                    trajectory_data.iloc[i]['timestamp'],
                    trajectory_data.iloc[j - 1]['timestamp']
                )

                if duration >= self.time_threshold:
                    # Calculate centroid of stay points
                    lat_mean = trajectory_data.iloc[i:j]['latitude'].mean()
                    lon_mean = trajectory_data.iloc[i:j]['longitude'].mean()

                    stay_point = StayPoint(
                        latitude=lat_mean,
                        longitude=lon_mean,
                        arrival_time=trajectory_data.iloc[i]['timestamp'],
                        departure_time=trajectory_data.iloc[j - 1]['timestamp'],
                        duration=duration
                    )
                    stay_points.append(stay_point)

                    logger.debug(f"Stay point detected: {lat_mean:.6f}, {lon_mean:.6f}, "
                                 f"duration={duration:.1f}min")

            i = max(i + 1, j - 1)

        logger.info(f"Detected {len(stay_points)} stay points from {len(trajectory_data)} GPS points")
        return stay_points

    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate geodesic distance between two coordinates in meters."""
        return geodesic(coord1, coord2).meters

    def _calculate_duration(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
        """Calculate duration between two timestamps in minutes."""
        return (end_time - start_time).total_seconds() / 60.0


class POIMatcher:
    """
    Matches stay points to Points of Interest (POI) using k-nearest neighbors.

    This class converts GPS coordinates to semantic labels by finding the nearest
    POI for each stay point.
    """

    def __init__(self, poi_data: pd.DataFrame, k: int = 1, max_distance: float = 500.0):
        """
        Initialize POI matcher.

        Args:
            poi_data (pd.DataFrame): DataFrame with columns ['latitude', 'longitude', 'category', 'name']
            k (int): Number of nearest neighbors to consider
            max_distance (float): Maximum distance in meters to consider a match
        """
        required_columns = ['latitude', 'longitude', 'category', 'name']
        if not all(col in poi_data.columns for col in required_columns):
            raise ValueError(f"poi_data must contain columns: {required_columns}")

        self.poi_data = poi_data.copy()
        self.k = k
        self.max_distance = max_distance

        # Initialize k-NN model
        coordinates = poi_data[['latitude', 'longitude']].values
        self.knn_model = NearestNeighbors(n_neighbors=k, metric='haversine')
        self.knn_model.fit(np.radians(coordinates))

        logger.info(f"POIMatcher initialized with {len(poi_data)} POIs, k={k}, "
                    f"max_distance={max_distance}m")

    def match_stay_points(self, stay_points: List[StayPoint]) -> List[StayPoint]:
        """
        Match stay points to POIs and assign semantic labels.

        Args:
            stay_points (List[StayPoint]): List of stay points to match

        Returns:
            List[StayPoint]: Stay points with assigned POI labels
        """
        if not stay_points:
            return stay_points

        # Extract coordinates
        coordinates = np.array([sp.coordinates for sp in stay_points])
        coordinates_rad = np.radians(coordinates)

        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(coordinates_rad)

        # Convert distances from radians to meters (approximate)
        distances_meters = distances * 6371000  # Earth radius in meters

        matched_stay_points = []
        for i, stay_point in enumerate(stay_points):
            if distances_meters[i, 0] <= self.max_distance:
                # Assign POI label
                poi_idx = indices[i, 0]
                poi_category = self.poi_data.iloc[poi_idx]['category']
                poi_name = self.poi_data.iloc[poi_idx]['name']

                stay_point.poi_label = f"{poi_category}_{poi_name}"
                logger.debug(f"Matched stay point to POI: {stay_point.poi_label}")
            else:
                stay_point.poi_label = "unknown"
                logger.debug(f"No POI match found within {self.max_distance}m")

            matched_stay_points.append(stay_point)

        matched_count = sum(1 for sp in matched_stay_points if sp.poi_label != "unknown")
        logger.info(f"Matched {matched_count}/{len(stay_points)} stay points to POIs")

        return matched_stay_points

    def get_poi_categories(self) -> List[str]:
        """Return list of unique POI categories."""
        return self.poi_data['category'].unique().tolist()


class TrajectoryPreprocessor:
    """
    Main preprocessing class that orchestrates the entire preprocessing pipeline.

    This class combines stay point detection, POI matching, and user filtering
    to generate semantic trajectory sequences.
    """

    def __init__(self,
                 stay_point_detector: StayPointDetector,
                 poi_matcher: POIMatcher,
                 min_stay_points: int = 200):
        """
        Initialize trajectory preprocessor.

        Args:
            stay_point_detector (StayPointDetector): Stay point detection component
            poi_matcher (POIMatcher): POI matching component
            min_stay_points (int): Minimum number of stay points required per user
        """
        self.stay_point_detector = stay_point_detector
        self.poi_matcher = poi_matcher
        self.min_stay_points = min_stay_points

        logger.info(f"TrajectoryPreprocessor initialized with min_stay_points={min_stay_points}")

    def process_user_trajectory(self, user_data: pd.DataFrame, user_id: str) -> Optional[TrajectorySequence]:
        """
        Process trajectory data for a single user.

        Args:
            user_data (pd.DataFrame): GPS trajectory data for one user
            user_id (str): Unique identifier for the user

        Returns:
            Optional[TrajectorySequence]: Processed trajectory sequence or None if insufficient data
        """
        try:
            # Detect stay points
            stay_points = self.stay_point_detector.detect_stay_points(user_data)

            # Filter users with insufficient stay points
            if len(stay_points) < self.min_stay_points:
                logger.debug(f"User {user_id} filtered out: {len(stay_points)} < {self.min_stay_points} stay points")
                return None

            # Match stay points to POIs
            matched_stay_points = self.poi_matcher.match_stay_points(stay_points)

            # Extract semantic labels
            semantic_labels = [sp.poi_label for sp in matched_stay_points]

            trajectory_sequence = TrajectorySequence(
                user_id=user_id,
                stay_points=matched_stay_points,
                semantic_labels=semantic_labels
            )

            logger.info(f"Processed trajectory for user {user_id}: {len(matched_stay_points)} stay points")
            return trajectory_sequence

        except Exception as e:
            logger.error(f"Error processing trajectory for user {user_id}: {str(e)}")
            return None

    def process_dataset(self, geolife_data: pd.DataFrame) -> List[TrajectorySequence]:
        """
        Process entire Geolife dataset.

        Args:
            geolife_data (pd.DataFrame): Complete Geolife dataset with columns
                                       ['user_id', 'latitude', 'longitude', 'timestamp']

        Returns:
            List[TrajectorySequence]: List of processed trajectory sequences

        Raises:
            ValueError: If required columns are missing from geolife_data
        """
        required_columns = ['user_id', 'latitude', 'longitude', 'timestamp']
        if not all(col in geolife_data.columns for col in required_columns):
            raise ValueError(f"geolife_data must contain columns: {required_columns}")

        logger.info(f"Processing dataset with {len(geolife_data)} GPS points from "
                    f"{geolife_data['user_id'].nunique()} users")

        trajectory_sequences = []

        # Process each user separately
        for user_id in geolife_data['user_id'].unique():
            user_data = geolife_data[geolife_data['user_id'] == user_id].copy()

            trajectory_sequence = self.process_user_trajectory(user_data, user_id)
            if trajectory_sequence is not None:
                trajectory_sequences.append(trajectory_sequence)

        logger.info(f"Successfully processed {len(trajectory_sequences)} users")
        return trajectory_sequences

    def get_vocabulary(self, trajectory_sequences: List[TrajectorySequence]) -> Dict[str, int]:
        """
        Build vocabulary from semantic labels.

        Args:
            trajectory_sequences (List[TrajectorySequence]): Processed trajectory sequences

        Returns:
            Dict[str, int]: Vocabulary mapping labels to indices
        """
        all_labels = []
        for seq in trajectory_sequences:
            all_labels.extend(seq.semantic_labels)

        unique_labels = sorted(set(all_labels))
        vocabulary = {label: idx for idx, label in enumerate(unique_labels)}

        logger.info(f"Built vocabulary with {len(vocabulary)} unique semantic labels")
        return vocabulary