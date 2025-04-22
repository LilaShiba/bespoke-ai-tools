import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple


# 1. General Utilities
def safe_load_csv(filepath: str) -> pd.DataFrame:
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

def remove_nans(data: List[float]) -> List[float]:
    """
    Removes NaN values from a list.
    
    Args:
    - data (List[float]): List of numerical data.
    
    Returns:
    - List[float]: List with NaN values removed.
    """
    return [x for x in data if not np.isnan(x)]

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy array to a range [0, 1].
    
    Args:
    - arr (np.ndarray): Input array.
    
    Returns:
    - np.ndarray: Normalized array.
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def standardize_array(arr: np.ndarray) -> np.ndarray:
    """
    Standardize a numpy array (zero mean, unit variance).
    
    Args:
    - arr (np.ndarray): Input array.
    
    Returns:
    - np.ndarray: Standardized array.
    """
    return (arr - np.mean(arr)) / np.std(arr)


# 2. Plotting Helpers
def show_plot(title: str = "", xlabel: str = "", ylabel: str = ""):
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
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
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

def entropy(p: np.ndarray) -> float:
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
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
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
