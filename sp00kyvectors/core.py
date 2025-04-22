import numpy as np
import matplotlib.pyplot as plt
from typing import *
import collections

class Vector:
    '''Vector Maths Class for statistical analysis and visualization.'''

    def __init__(self, label: int = 0, data_points: np.array = None):
        self.label = label
        if data_points is not None:
            self.v = data_points
            self.n = len(data_points)

            if data_points.ndim == 2 and data_points.shape[1] > 1:
                self.x = data_points[:, 0]
                self.y = data_points[:, 1] if data_points.shape[1] > 1 else None
            else:
                self.x = data_points
                self.y = None
        else:
            self.x = None
            self.y = None

    def count(self, array: np.array, value: float) -> int:
        '''Returns count of value in an array.'''
        return np.count_nonzero(array == value)

    def linear_scale(self):
        '''Visualize the linear scaling of the data.'''
        if self.x is None:
            raise ValueError("No data available for linear scaling.")
        
        histo_gram = collections.Counter(self.x)
        val, cnt = zip(*histo_gram.items())

        n = len(cnt)
        prob_vector = [x / n for x in cnt]
        plt.plot(val, prob_vector, 'x')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.title('Linear Scale of Vector')
        plt.show()

    def log_binning(self) -> Tuple[float, float]:
        '''Plot the degree distribution with log binning.'''
        if self.x is None:
            raise ValueError("No data available for log binning.")
        
        histo_gram = collections.Counter(self.x)
        val, cnt = zip(*histo_gram.items())

        n = len(cnt)
        prob_vector = [x / n for x in cnt]
        in_max, in_min = max(prob_vector), min(prob_vector)
        log_bins = np.logspace(np.log10(in_min), np.log10(in_max), num=20)
        
        deg_hist, log_bin_edges = np.histogram(
            prob_vector, bins=log_bins, density=True, range=(in_min, in_max))
        
        plt.title(f"Log Binning & Scaling")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.plot(log_bin_edges[:-1], deg_hist, 'o')
        plt.show()

        return in_min, in_max

    def get_prob_vector(self, axis: int = 0, rounding: int = None) -> Dict[float, float]:
        '''Return probability vector for a given axis.'''
        if axis == 0:
            vector = self.x
        elif self.y is not None:
            vector = self.y
        else:
            raise ValueError("Invalid axis for probability vector.")

        if rounding is not None:
            vector = np.round(vector, rounding)

        unique_values = np.unique(vector)
        prob_dict = {value: self.count(
            vector, value) / self.n for value in unique_values}
        return prob_dict

    def plot_pdf(self, bins: int = 'auto'):
        '''Plots the Probability Density Function (PDF) of the vector.'''
        if self.y is not None:
            data = self.y
        else:
            data = self.x

        density, bins, _ = plt.hist(
            data, bins=bins, density=True, alpha=0.5, label='PDF')
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.title('Probability Density Function')
        plt.legend()
        plt.show()

    def plot_basic_stats(self):
        '''Plots basic statistics: mean and standard deviation.'''
        if self.y is not None:
            data = self.y
        else:
            data = self.x

        mean = np.mean(data)
        std = np.std(data)

        plt.hist(data, bins='auto', alpha=0.5, label='Data')
        plt.axvline(mean, color='r', linestyle='dashed',
                    linewidth=1, label=f'Mean: {mean:.2f}')
        plt.axvline(mean + std, color='g', linestyle='dashed',
                    linewidth=1, label=f'Std: {std:.2f}')
        plt.axvline(mean - std, color='g', linestyle='dashed', linewidth=1)
        plt.legend()
        plt.show()

    def rolling_average(self, window_size: int = 3) -> np.array:
        '''Calculates and returns the rolling average of the vector using numpy.'''
        if self.y is not None:
            data = self.y
        else:
            data = self.x

        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

    @staticmethod
    def calculate_aligned_entropy(vector1, vector2) -> float:
        """
        Calculates the entropy between two aligned probability distributions from Vector instances.
        """
        prob_dist1 = np.array(list(vector1.get_prob_vector().values()))
        prob_dist2 = np.array(list(vector2.get_prob_vector().values()))

        # Calculate joint probabilities
        joint_probs = prob_dist1 * prob_dist2

        # Filter out zero probabilities to avoid NaNs in the logarithm
        joint_probs = joint_probs[joint_probs != 0]

        # Calculate entropy
        entropy = -np.sum(joint_probs * np.log2(joint_probs))

        return entropy

    @staticmethod
    def set_operations(v1: 'Vector', v2: 'Vector') -> Tuple[Set[float], Set[float], float]:
        '''Performs set operations: union, intersection, and calculates Jaccard index.'''
        set1 = set(v1.x if v1.x is not None else [])
        set2 = set(v2.y if v2.y is not None else v2.x if v2.x is not None else [])

        if not set1 or not set2:
            raise ValueError("Both vectors must have data for set operations.")

        union = set1.union(set2)
        intersection = set1.intersection(set2)
        jaccard_index = len(intersection) / len(union) if union else 0.0

        return union, intersection, jaccard_index

    @staticmethod
    def generate_noisy_sin(start: float = 0, points: int = 100) -> Tuple[np.array, np.array]:
        '''Creates a noisy sine wave for testing.'''
        x = np.linspace(start, 2 * np.pi, points)
        y = np.sin(x) + np.random.normal(0, 0.2, points)
        return np.column_stack((x, y))

    def transform_to_euclidean(self, projection_vector: np.array) -> np.array:
        '''Transforms data to a new coordinate system using a projection vector.'''
        if self.x is None:
            raise ValueError("No data available for transformation.")

        projection_matrix = np.outer(projection_vector, projection_vector)
        transformed_data = np.dot(self.x, projection_matrix)
        return transformed_data

    def normalize(self) -> np.array:
        '''Normalizes data to have a mean of 0 and standard deviation of 1.'''
        if self.x is None:
            raise ValueError("No data available for normalization.")
        
        normalized_data = (self.x - np.mean(self.x)) / np.std(self.x)
        return normalized_data

    def calculate_distance(self, other_vector: 'Vector') -> float:
        '''Calculates Euclidean distance between this vector and another.'''
        if self.x is None or other_vector.x is None:
            raise ValueError("Data is missing from one or both vectors.")
        
        distance = np.linalg.norm(self.x - other_vector.x)
        return distance

    def resample(self, size: int) -> np.array:
        '''Resamples the data (with replacement) to the specified size.'''
        if self.x is None:
            raise ValueError("No data available for resampling.")
        
        return np.random.choice(self.x, size=size, replace=True)

    def get_median(self) -> float:
        '''Returns the median of the data.'''
        if self.x is None:
            raise ValueError("No data available for median calculation.")
        
        return np.median(self.x)

    def get_mean(self) -> float:
        '''Returns the mean of the data.'''
        if self.x is None:
            raise ValueError("No data available for mean calculation.")
        
        return np.mean(self.x)

    def get_std(self) -> float:
        '''Returns the standard deviation of the data.'''
        if self.x is None:
            raise ValueError("No data available for std calculation.")
        
        return np.std(self.x)

    # Vector Algebra Operations

    def add(self, other: 'Vector') -> 'Vector':
        '''Adds two vectors element-wise.'''
        if self.x is None or other.x is None:
            raise ValueError("Data missing in one of the vectors.")
        
        added = np.add(self.x, other.x)
        return Vector(label=f'{self.label} + {other.label}', data_points=added)

    def subtract(self, other: 'Vector') -> 'Vector':
        '''Subtracts another vector from this vector element-wise.'''
        if self.x is None or other.x is None:
            raise ValueError("Data missing in one of the vectors.")
        
        subtracted = np.subtract(self.x, other.x)
        return Vector(label=f'{self.label} - {other.label}', data_points=subtracted)

    def dot(self, other: 'Vector') -> float:
        '''Calculates the dot product between two vectors.'''
        if self.x is None or other.x is None:
            raise ValueError("Data missing in one of the vectors.")
        
        return np.dot(self.x, other.x)

    def cross(self, other: 'Vector') -> np.array:
        '''Calculates the cross product between two vectors.'''
        if self.x is None or other.x is None or len(self.x) != 3 or len(other.x) != 3:
            raise ValueError("Cross product requires 3-dimensional vectors.")
        
        return np.cross(self.x, other.x)

