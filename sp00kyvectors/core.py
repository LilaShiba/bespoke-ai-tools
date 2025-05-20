import numpy as np
import matplotlib.pyplot as plt
from typing import *
import collections
import pandas as pd
from scipy import stats
import glob 
import os
from tqdm import tqdm
from pathlib import Path
import warnings
from typing import *

class Vector:
    '''Vector Maths Class for statistical analysis and visualization.'''

    def __init__(self, label: Union[int, str] = 0,
                 data_points: Union[np.ndarray, pd.Series, pd.DataFrame, list] = None):
        self.label = label
        self.t = None  # original index for Series inputs

        # --- ingest data_points ---
        if isinstance(data_points, pd.Series):
            # preserve timestamp/index
            self.t = data_points.index.to_numpy()
            data = data_points.values
        elif isinstance(data_points, pd.DataFrame):
            data = data_points.values
        else:
            data = np.asarray(data_points) if data_points is not None else None

        if data is not None:
            self.v = data
            self.n = len(data)
            if data.ndim == 2 and data.shape[1] > 1:
                self.x = data[:, 0]
                self.y = data[:, 1]
            else:
                self.x = data
                self.y = np.array([x for x in range(len(self.x))])
        else:
            self.v = None
            self.n = 0
            self.x = None
            self.y = None

    def drop_na(self):
        """Remove entries where x or y is NaN, syncing t, v, and n."""
        if self.x is None:
            return

       
        mask = ~(np.isnan(self.x) | np.isnan(self.y))
       

        self.x = self.x[mask]
        if self.y is not None:
            self.y = self.y[mask]
        if self.t is not None:
            self.t = self.t[mask]
        if self.v is not None:
            self.v = self.v[mask]
        self.n = len(self.x)

    def drop_outliers(self, method: str = "iqr", thresh: float = 1.5):
        """
        Remove outliers from x (and y) using:
         - 'iqr'   : interquartile range rule with multiplier `thresh`
         - 'zscore': standard Z‑score filter at |z| < thresh
        """
        if self.x is None:
            return

        # build pandas DataFrame for filtering
        df = pd.DataFrame({"x": self.x})
        if self.y is not None:
            df["y"] = self.y

        if method.lower() == "iqr":
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - thresh * IQR
            upper = Q3 + thresh * IQR
            mask = df.apply(lambda col: col.between(lower[col.name], upper[col.name]))
            keep = mask.all(axis=1)

        elif method.lower() == "zscore":
            z = df.apply(lambda col: stats.zscore(col, nan_policy="omit"))
            keep = (z.abs() < thresh).all(axis=1)

        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        # apply mask
        df_clean = df[keep]
        self.x = df_clean["x"].to_numpy()
        if self.y is not None:
            self.y = df_clean["y"].to_numpy()
        if self.t is not None:
            self.t = self.t[keep.to_numpy()]

        self.v = np.column_stack((self.x, self.y)) if self.y is not None else self.x
        self.n = len(self.x)

    def count(self, array: np.ndarray, value: float) -> int:
        return np.count_nonzero(array == value)

    def linear_scale(self):
        if self.x is None:
            raise ValueError("No data available for linear scaling.")
        histo = collections.Counter(self.x)
        val, cnt = zip(*sorted(histo.items()))
        total = sum(cnt)
        prob = [c/total for c in cnt]
        plt.plot(val, prob, 'x')
        plt.xlabel('Value'); plt.ylabel('Probability')
        plt.title('Linear Scale of Vector')
        plt.grid(True); plt.show()

    def log_binning(self) -> Tuple[float, float]:
        if self.x is None:
            raise ValueError("No data available for log binning.")
        hist = collections.Counter(self.x)
        _, cnt = zip(*sorted(hist.items()))
        total = sum(cnt)
        prob = [c/total for c in cnt]
        nonzero = [p for p in prob if p > 0]
        mn, mx = min(nonzero), max(nonzero)
        bins = np.logspace(np.log10(mn), np.log10(mx), num=20)
        hist_vals, edges = np.histogram(nonzero, bins=bins, density=True)
        plt.title("Log Binning & Scaling")
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel('K'); plt.ylabel('P(K)')
        plt.plot(edges[:-1], hist_vals, 'o')
        plt.grid(True); plt.show()
        return mn, mx

    def get_prob_vector(self, axis: int = 0, rounding: int = None) -> Dict[float, float]:
        vec = self.x
        if vec is None:
            raise ValueError("No data available for probability vector.")
        if rounding is not None:
            vec = np.round(vec, rounding)
        vals, cnts = np.unique(vec, return_counts=True)
        return dict(zip(vals, cnts / len(vec)))

    def plot_pdf(self, bins: Union[int, str] = 'auto'):
        data = self.x
        if data is None:
            raise ValueError("No data for PDF plot.")
        plt.hist(data, bins=bins, density=True, alpha=0.5, label='PDF')
        plt.ylabel('Probability'); plt.xlabel('Data')
        plt.title('Probability Density Function')
        plt.legend(); plt.grid(True); plt.show()

    def plot_basic_stats(self):
        data = self.x
        if data is None:
            raise ValueError("No data available.")
        m, s = np.mean(data), np.std(data)
        plt.hist(data, bins='auto', alpha=0.5, label='Data')
        plt.axvline(m, color='r', linestyle='--', linewidth=1, label=f'Mean: {m:.2f}')
        plt.axvline(m+s, color='g', linestyle='--', linewidth=1, label=f'Std: {s:.2f}')
        plt.axvline(m-s, color='g', linestyle='--', linewidth=1)
        plt.legend(); plt.grid(True); plt.show()

    def rolling_average(self, window_size: int = 3) -> np.ndarray:
        data = self.x
        if data is None or window_size > len(data):
            raise ValueError("Insufficient data for rolling average.")
        csum = np.cumsum(np.insert(data, 0, 0))
        return (csum[window_size:] - csum[:-window_size]) / float(window_size)

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

    def transform_to_euclidean(self, projection_vector: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise ValueError("No data for transformation.")
        P = np.outer(projection_vector, projection_vector)
        return np.dot(self.x, P)
    
    @staticmethod
    def normalize(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        data = np.concatenate([arr1, arr2])
        mask = ~(np.isnan(data))
        data = data[mask]
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data


    def calculate_distance(self, other_vector: 'Vector') -> float:
        if self.x is None or other_vector.x is None:
            raise ValueError("Missing data for distance calculation.")
        return float(np.linalg.norm(self.x - other_vector.x))

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

    def add(self, other: 'Vector') -> 'Vector':
        if self.x is None or other.x is None:
            raise ValueError("Missing data for addition.")
        return Vector(label=f'{self.label}+{other.label}', data_points=[self.x + other.x])

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

    def roll(self, shift: int = 1):
        """Shift x, y, and t (if present)."""
        if self.x is not None:
            self.x = np.roll(self.x, shift)
        if self.y is not None:
            self.y = np.roll(self.y, shift)
        if self.t is not None:
            self.t = np.roll(self.t, shift)

    def __repr__(self):
        return f"<Vector '{self.label}' (n={self.n})>"


    @staticmethod
    def plot_vector_op(v1, v2=None, op: str = "add", scale: str = "unit") -> None:
        """
        Plot 2D linear‑algebra ops on Vector instances.

        Parameters
        ----------
        v1 : Vector
            First Vector (must have .x and .y)
        v2 : Vector, optional
            Second Vector for binary ops
        op : str
            One of: 'plot', 'add', 'subtract', 'dot', 'cross',
            'normalize', 'project'
        scale : str
            'unit' for unit vector normalization, '01' for [0,1] scaling
        """

        def clean_and_scale(x, y):
            arr = np.stack([x, y], axis=1)
            arr = arr[~np.isnan(arr).any(axis=1)]

            # Drop extreme outliers using z-score
            z = np.abs((arr - arr.mean(axis=0)) / arr.std(axis=0))
            arr = arr[(z < 3).all(axis=1)]

            # Scale
            if scale == "01":
                min_vals = arr.min(axis=0)
                max_vals = arr.max(axis=0)
                arr = (arr - min_vals) / (max_vals - min_vals)
            elif scale == "unit":
                norm = np.linalg.norm(arr, axis=1, keepdims=True)
                norm[norm == 0] = 1
                arr = arr / norm

            return np.mean(arr, axis=0)

        u = clean_and_scale(v1.x, v1.y)
        w = clean_and_scale(v2.x, v2.y) if v2 else None

        # --- handle ops ---
        if op == "plot":
            results = []
        elif op == "add":
            results = [u + w]
        elif op == "subtract":
            results = [u - w]
        elif op == "dot":
            print(f"Dot({v1.label},{v2.label}) =", float(np.dot(u, w)))
            return
        elif op == "cross":
            z = u[0] * w[1] - u[1] * w[0]
            print(f"Cross({v1.label},{v2.label}) =", z)
            return
        elif op == "normalize":
            norm = np.linalg.norm(u)
            results = [u / norm]
        elif op == "project":
            proj = (np.dot(u, w) / np.dot(w, w)) * w
            results = [proj]
        else:
            raise ValueError(f"Unknown op {op!r}")

        # --- plotting ---
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.axhline(0, color='gray', lw=1)
        ax.axvline(0, color='gray', lw=1)
        ax.set_aspect('equal', 'box')
        ax.quiver(0, 0, *u, angles='xy', scale_units='xy', scale=1, label=v1.label, color='pink')
        if w is not None:
            ax.quiver(0, 0, *w, angles='xy', scale_units='xy', scale=1, label=v2.label, color='blue')
        for r in results:
            ax.quiver(0, 0, *r, angles='xy', scale_units='xy', scale=1, label=f"{op} result")
        ax.legend()
        ax.grid(True)
        plt.title(f"{op.capitalize()} of {v1.label}" + (f" & {v2.label}" if v2 else ""))
        plt.show()


    @staticmethod
    def load_folder(path):
        """
        Load all CSVs in the given folder, align columns by name (union of all headers),
        drop stray index columns or repeated header rows, and return one combined DataFrame.
        """
        folder_path = Path(path)
        csv_paths = glob.glob(str(folder_path / "*.csv"))
        if not csv_paths:
            return pd.DataFrame()

        warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
        # 1) Build the full set of columns
        all_cols = set()
        for fp in tqdm(csv_paths, desc="🔍 Gathering columns"):
            tmp = pd.read_csv(fp, nrows=0, index_col=False)
            tmp = tmp.loc[:, ~tmp.columns.str.contains(r'^Unnamed')]                     # drop stray unnamed cols
            all_cols.update(tmp.columns.tolist())
        all_cols = sorted(all_cols)

        # 2) Read, clean, align each file
        dfs = []
        for fp in tqdm(csv_paths, desc="📥 Reading & aligning"):
            df = pd.read_csv(fp, index_col=False)                                        # no inferred index
            df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]                         # drop unnamed cols
            df = df[[col for col in df.columns                                         
                    if not (df[col].astype(str) == col).all()]]                         # drop repeated header rows
            df = df.reindex(columns=all_cols)                                             # align + fill NaN
            dfs.append(df)

        # 3) Concatenate everything
        combined = pd.concat(dfs, ignore_index=True, sort=False)
        return combined

