import numpy as np
import matplotlib.pyplot as plt
from typing import *
import collections
import pandas as pd

class Vector:
    '''Vector Maths Class for statistical analysis and visualization.'''

    def __init__(self, label: Union[int, str] = 0, data_points: Union[np.ndarray, pd.Series] = None):
        self.label = label
        if isinstance(data_points, pd.Series):
            data_points = data_points.dropna().values  # Convert to clean np.ndarray

        if data_points is not None:
            self.v = data_points
            self.n = len(data_points)

            if data_points.ndim == 2 and data_points.shape[1] > 1:
                self.x = data_points[:, 0]
                self.y = data_points[:, 1]
            else:
                self.x = data_points
                self.y = None
        else:
            self.x = None
            self.y = None

    def count(self, array: np.ndarray, value: float) -> int:
        return np.count_nonzero(array == value)

    def linear_scale(self):
        if self.x is None:
            raise ValueError("No data available for linear scaling.")
        histo_gram = collections.Counter(self.x)
        val, cnt = zip(*sorted(histo_gram.items()))
        n = sum(cnt)
        prob_vector = [x / n for x in cnt]
        plt.plot(val, prob_vector, 'x')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.title('Linear Scale of Vector')
        plt.grid(True)
        plt.show()

    def log_binning(self) -> Tuple[float, float]:
        if self.x is None:
            raise ValueError("No data available for log binning.")

        hist = collections.Counter(self.x)
        val, cnt = zip(*sorted(hist.items()))
        total = sum(cnt)
        prob_vector = [c / total for c in cnt]

        nonzero_probs = [p for p in prob_vector if p > 0]
        in_min, in_max = min(nonzero_probs), max(nonzero_probs)
        log_bins = np.logspace(np.log10(in_min), np.log10(in_max), num=20)

        deg_hist, bin_edges = np.histogram(nonzero_probs, bins=log_bins, density=True)

        plt.title("Log Binning & Scaling")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(bin_edges[:-1], deg_hist, 'o')
        plt.grid(True)
        plt.show()
        return in_min, in_max

    def get_prob_vector(self, axis: int = 0, rounding: int = None) -> Dict[float, float]:
        vector = self.y if axis == 1 and self.y is not None else self.x
        if vector is None:
            raise ValueError("No data available for probability vector.")
        if rounding is not None:
            vector = np.round(vector, rounding)
        total = len(vector)
        values, counts = np.unique(vector, return_counts=True)
        return dict(zip(values, counts / total))

    def plot_pdf(self, bins: Union[int, str] = 'auto'):
        data = self.y if self.y is not None else self.x
        if data is None:
            raise ValueError("No data for PDF plot.")
        plt.hist(data, bins=bins, density=True, alpha=0.5, label='PDF')
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title('Probability Density Function')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_basic_stats(self):
        data = self.y if self.y is not None else self.x
        if data is None:
            raise ValueError("No data available.")
        mean = np.mean(data)
        std = np.std(data)
        plt.hist(data, bins='auto', alpha=0.5, label='Data')
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
        plt.axvline(mean + std, color='g', linestyle='dashed', linewidth=1, label=f'Std: {std:.2f}')
        plt.axvline(mean - std, color='g', linestyle='dashed', linewidth=1)
        plt.legend()
        plt.grid(True)
        plt.show()

    def rolling_average(self, window_size: int = 3) -> np.ndarray:
        data = self.y if self.y is not None else self.x
        if data is None or window_size > len(data):
            raise ValueError("Insufficient data for rolling average.")
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

    @staticmethod
    def calculate_aligned_entropy(vector1: 'Vector', vector2: 'Vector') -> float:
        keys = set(vector1.get_prob_vector().keys()).union(vector2.get_prob_vector().keys())
        p = np.array([vector1.get_prob_vector().get(k, 0.0) for k in keys])
        q = np.array([vector2.get_prob_vector().get(k, 0.0) for k in keys])
        joint_probs = p * q
        joint_probs = joint_probs[joint_probs > 0]
        return -np.sum(joint_probs * np.log2(joint_probs))

    @staticmethod
    def set_operations(v1: 'Vector', v2: 'Vector') -> Tuple[Set[float], Set[float], float]:
        set1 = set(v1.x if v1.x is not None else [])
        set2 = set(v2.y if v2.y is not None else v2.x if v2.x is not None else [])
        union = set1.union(set2)
        intersection = set1.intersection(set2)
        jaccard_index = len(intersection) / len(union) if union else 0.0
        return union, intersection, jaccard_index

    @staticmethod
    def generate_noisy_sin(start: float = 0, points: int = 100) -> np.ndarray:
        x = np.linspace(start, 2 * np.pi, points)
        y = np.sin(x) + np.random.normal(0, 0.2, points)
        return np.column_stack((x, y))

    def transform_to_euclidean(self, projection_vector: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise ValueError("No data for transformation.")
        projection_matrix = np.outer(projection_vector, projection_vector)
        return np.dot(self.x, projection_matrix)

    def normalize(self) -> np.ndarray:
        if self.x is None:
            raise ValueError("No data to normalize.")
        return (self.x - np.mean(self.x)) / np.std(self.x)

    def calculate_distance(self, other_vector: 'Vector') -> float:
        if self.x is None or other_vector.x is None:
            raise ValueError("Missing data for distance calculation.")
        return np.linalg.norm(self.x - other_vector.x)

    def resample(self, size: int) -> np.ndarray:
        if self.x is None:
            raise ValueError("No data to resample.")
        return np.random.choice(self.x, size=size, replace=True)

    def get_median(self) -> float:
        return float(np.median(self.x)) if self.x is not None else None

    def get_mean(self) -> float:
        return float(np.mean(self.x)) if self.x is not None else None

    def get_std(self) -> float:
        return float(np.std(self.x)) if self.x is not None else None

    # Vector Algebra

    def add(self, other: 'Vector') -> 'Vector':
        if self.x is None or other.x is None:
            raise ValueError("Missing data for addition.")
        return Vector(label=f'{self.label}+{other.label}', data_points=self.x + other.x)

    def subtract(self, other: 'Vector') -> 'Vector':
        if self.x is None or other.x is None:
            raise ValueError("Missing data for subtraction.")
        return Vector(label=f'{self.label}-{other.label}', data_points=self.x - other.x)

    def dot(self, other: 'Vector') -> float:
        if self.x is None or other.x is None:
            raise ValueError("Missing data for dot product.")
        return float(np.dot(self.x, other.x))

    def cross(self, other: 'Vector') -> np.ndarray:
        if self.x is None or other.x is None:
            raise ValueError("Missing data for cross product.")
        if len(self.x) != 3 or len(other.x) != 3:
            raise ValueError("Cross product only defined for 3D vectors.")
        return np.cross(self.x, other.x)
