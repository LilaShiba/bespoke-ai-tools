import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Set
import glob 
from pathlib import Path

class Sp00kyUtils():
    def __init__(self, lable):
        self.lable = lable 
    # 1. General Utilities
    def safe_load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Safely load a CSV file into a pandas DataFrame.
        
        Args:
        - filepath (str): Path to the CSV file.
        
        Returns:
        - pd.DataFrame: Data loaded from the CSV.
        """
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        raise FileNotFoundError(f"File not found: {filepath}")

    def remove_nans(self, data: List[float]) -> List[float]:
        """
        Removes NaN values from a list.
        
        Args:
        - data (List[float]): List of numerical data.
        
        Returns:
        - List[float]: List with NaN values removed.
        """
        return [x for x in data if not np.isnan(x)]

    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Normalize a numpy array to a range [0, 1].
        
        Args:
        - arr (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Normalized array.
        """
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def standardize_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Standardize a numpy array (zero mean, unit variance).
        
        Args:
        - arr (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Standardized array.
        """
        return (arr - np.mean(arr)) / np.std(arr)

    # 2. Plotting Helpers
    def show_plot(self, title: str = "", xlabel: str = "", ylabel: str = ""):
        """
        Display a plot with customizable title, x and y labels.
        
        Args:
        - title (str): The title of the plot.
        - xlabel (str): The label for the x-axis.
        - ylabel (str): The label for the y-axis.
        """
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 3. Statistical Tools
    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate the Kullback-Leibler (KL) divergence between two probability distributions.
        
        Args:
        - p (np.ndarray): The first distribution (P).
        - q (np.ndarray): The second distribution (Q).
        
        Returns:
        - float: The KL divergence between P and Q.
        """
        p = p[p > 0]
        q = q[q > 0]
        return np.sum(p * np.log(p / q))

    def entropy(self, p: np.ndarray) -> float:
        """
        Calculate the entropy of a probability distribution.
        
        Args:
        - p (np.ndarray): The probability distribution.
        
        Returns:
        - float: The entropy of the distribution.
        """
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    # 4. Distance Functions (for custom k-NN or clustering)
    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two vectors.
        
        Args:
        - a (np.ndarray): First vector.
        - b (np.ndarray): Second vector.
        
        Returns:
        - float: The Euclidean distance between a and b.
        """
        return np.linalg.norm(a - b)

    def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate the Manhattan distance between two vectors.
        
        Args:
        - a (np.ndarray): First vector.
        - b (np.ndarray): Second vector.
        
        Returns:
        - float: The Manhattan distance between a and b.
        """
        return np.sum(np.abs(a - b))

    @staticmethod
    def load_folder(path):
        folder_path = Path("/Users/lila/data_viz/data")
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))
        all_dfs = []
        # Append the rest of the CSVs without headers
        for file in tqdm(all_files, desc="✨💖 Loading CSVs ✨💖"):
            df = pd.read_csv(file)
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df

    @staticmethod
    def calculate_aligned_entropy(vector1: 'Vector', vector2: 'Vector') -> float:
        keys = set(vector1.get_prob_vector()).union(vector2.get_prob_vector())
        p = np.array([vector1.get_prob_vector().get(k, 0.0) for k in keys])
        q = np.array([vector2.get_prob_vector().get(k, 0.0) for k in keys])
        jp = p * q
        jp = jp[jp > 0]
        return -np.sum(jp * np.log2(jp))

    @staticmethod
    def set_operations(v1: 'Vector', v2: 'Vector') -> Tuple[Set[float], Set[float], float]:
        s1 = set(v1.x)
        s2 = set(v2.x)
        union = s1.union(s2)
        inter = s1.intersection(s2)
        j = len(inter) / len(union) if union else 0.0
        return union, inter, j

    @staticmethod
    def generate_noisy_sin(start: float = 0, points: int = 100) -> np.ndarray:
        x = np.linspace(start, 2*np.pi, points)
        y = np.sin(x) + np.random.normal(0, 0.2, points)
        return np.column_stack((x, y))
    
    @staticmethod
    def normalize(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        data = np.concatenate([arr1, arr2])
        mask = ~(np.isnan(data))
        data = data[mask]
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data