# 🎃 Sp00kyVectors: Vector Analysis wrapper for Python

Welcome to **Sp00kyVectors**, the eerily intuitive Python class for **vector analysis**, **statistical computation**, and **visualizations** 🪄👻

Whether you’re chasing spectral patterns or just need stats from dirty data, this library’s got your back (from beyond).

---

## 🧼 Examples

| Before | After |
|--------|-------|
| ![Before Cleaning](https://github.com/LilaShiba/sp00kyvectors/raw/main/imgs/temp_before_clean.png) | ![After Cleaning](https://github.com/LilaShiba/sp00kyvectors/raw/main/imgs/temp_after_clean.png) |
| ![Before Bin](https://github.com/LilaShiba/sp00kyvectors/raw/main/imgs/beforebin.png) | ![After Bin](https://github.com/LilaShiba/sp00kyvectors/raw/main/imgs/afterbin.png) |
| ![Vector Projections](https://github.com/LilaShiba/sp00kyvectors/raw/main/imgs/output_add.png) | ![Normalize](https://github.com/LilaShiba/sp00kyvectors/raw/main/imgs/output.png) |

## 🧹 Dirty Data? No Problem!
#### Load without worry
Easily load and align mismatched CSV files. This utility intelligently collects, normalizes, and organizes messy datasets — so you can focus on the analysis, not the cleanup. 🚀

``` Vector.load_folder(path) ``` loads a folder of CSV files with potentially mismatched or missing columns,  
aligns all columns based on their headers, and combines them into a single clean DataFrame.  
Missing columns in any file are automatically filled with `NaN` values to maintain consistency.

Perfect for messy datasets where CSVs don't share the exact same structure!

Cleaning is done one layer up with `sp00kyDF.get_clean_df()` ✨🧹

This method returns a cleaned version of the DataFrame by performing the following steps:

1. 🧩 Removes duplicate rows (performed twice to ensure thorough cleaning)  
2. 🚫📊 Clips outlier values based on the Z-score method *(an Interquartile Range (IQR) method is also available)*  
3. 🏷️ Standardizes column names for consistency  
4. ❌🕳️ *(Optionally drops null values — currently commented out)*

Finally, it returns the cleaned DataFrame ready for analysis. 🎯


# 🎛️⚙️✨ Granular Control? No Problem!
## 🧠 Features

- 🧮 **Vector Magic**:
  - Load 1D or 2D arrays into `Vector` objects
  - X/Y decomposition for 2D data
  - Linear algebra methods like magnitude, angle, dot, and projection

- 📊 **Statistical Potions**:
  - Mean, median, standard deviation 💀  
  - Probability vectors and PDFs 🧪  
  - Z-score normalization 🧼  
  - Entropy between aligned vectors 🌀  
  - Internal entropy of a vector  

- 🖼️ **Visualizations**:
  - Linear and log-scale histogramming  
  - Vector plots with tails, heads, and haunted trails  
  - Optional "entropy mode" that colors plots based on mysterious disorder 👀  

- 🔧 **Tools of the Craft**:
  - Gaussian kernel smoothing for smoothing out your nightmares  
  - Elementwise operations: `.normalize()`, `.project()`, `.difference()`, and more  
  - Pretty `__repr__` so your print statements conjure elegant summaries

---

## 🧪 Example

<pre><code>
from sp00kyvectors import Vector

v = Vector([1, 2, 3, 4, 5])
print(v.mean())  # Output: 3.0

v2 = Vector([1, 1, 1, 1, 6])
print(v.entropy(v2))  # Output: spooky entropy value
</code></pre>

---

## 📦 Installation

<pre><code>
pip install sp00kyvectors
</code></pre>

Or summon it from your own local clone:

<pre><code>
git clone https://github.com/LilaShiba/sp00kyvectors.git
cd sp00kyvectors
pip install .
</code></pre>

---

## 📚 Documentation

### 🧪 Class: `Vector`

#### ✨ Initialization

Create a new `Vector` from a list or numpy array.

<pre><code>
from sp00kyvectors import Vector

v = Vector([1, 2, 3, 4, 5])
</code></pre>

If you're working with 2D data:

<pre><code>
v2d = Vector([[1, 2], [3, 4], [5, 6]])
</code></pre>

---

## 📊 Methods

### `.mean()`

Returns the mean of the vector.

<pre><code>
v.mean()  # ➜ 3.0
</code></pre>

---

### `.median()`

Returns the median.

<pre><code>
v.median()  # ➜ 3
</code></pre>

---

### `.std()`

Returns the standard deviation.

<pre><code>
v.std()  # ➜ 1.5811...
</code></pre>

---

### `.normalize()`

Normalizes the vector using Z-score (zero mean, unit variance).

<pre><code>
v_norm = v.normalize()
</code></pre>

---

### `.entropy(other: Vector)`

Computes Shannon entropy between this vector and another.

<pre><code>
v2 = Vector([1, 1, 1, 1, 6])
v.entropy(v2)  # ➜ ~0.72 (varies based on normalization)
</code></pre>

---

### `.difference(other: Vector)`

Returns a new Vector representing the difference between this vector and another.

<pre><code>
v3 = v.difference(v2)
</code></pre>

---

### `.project(dim: int)`

Projects a 2D vector onto a specific dimension (0 = x, 1 = y).

<pre><code>
vx = v2d.project(0)
vy = v2d.project(1)
</code></pre>

---

## 🔢 Linear Algebra Methods

### `.magnitude()`

Returns the magnitude (length) of the vector.

<pre><code>
v.magnitude()  # ➜ 7.416
</code></pre>

---

### `.angle(other: Vector)`

Returns the angle between this vector and another, in radians.

<pre><code>
v.angle(v2)  # ➜ 0.225 (radians)
</code></pre>

---

### `.dot(other: Vector)`

Computes the dot product of this vector and another.

<pre><code>
v.dot(v2)  # ➜ 24
</code></pre>

---

### `.cross(other: Vector)`

Computes the cross product (only for 3D vectors).

<pre><code>
v3d = Vector([1, 2, 3])
v3d2 = Vector([4, 5, 6])
v3d.cross(v3d2)  # ➜ [-3, 6, -3]
</code></pre>

---

### `.projection(other: Vector)`

Returns the projection of this vector onto another.

<pre><code>
v.projection(v2)  # ➜ Vector with projected values
</code></pre>

---

## 📈 Plotting

### `.histogram(log=False)`

Plots a histogram of the vector values. Set `log=True` for logarithmic scale.

<pre><code>
v.histogram()
v.histogram(log=True)
</code></pre>

---

### `.plot_vectors(mode="line", entropy=False)`

Plots 2D vectors.

- `mode`: `"line"`, `"arrow"`, or `"trail"`
- `entropy`: if `True`, colorizes vectors by entropy

<pre><code>
v2d.plot_vectors(mode="arrow", entropy=True)
</code></pre>

---

## 🔮 Utilities

### `.gaussian_smooth(sigma=1.0)`

Applies Gaussian smoothing to the vector.

<pre><code>
v_smooth = v.gaussian_smooth(sigma=2.0)
</code></pre>

---

## 💀 Dunder Methods

### `__repr__()`

Pretty string representation.

<pre><code>
print(v)  # Vector(mean=3.0, std=1.58, ...)
</code></pre>

---

## 🛠 Developer Notes

- Internal data is stored as `numpy.ndarray`
- Methods use `scipy.stats`, `numpy`, and `matplotlib`
- Entropy assumes aligned distributions (normalized first)

---

## 🧛 License

MIT — haunt and hack as you please.

---

## 🕸️ Coming Soon

- 3D support
- More spooky plots
- CLI interface: `spookify file.csv --plot`

---

## 👻 Contributing

Spirits and sorcerers of all levels are welcome. Open an issue, fork the repo, or summon a pull request.

---

## 🧛 License

MIT — you’re free to haunt this code as you wish as long as money is never involved! 

---

✨ Stay spooky, and may your vectors always point toward the unknown. 🕸️

# Student Opportunities 🎓💻

Learning to code, using GitHub, or just curious? Reach out and join the team!  
We’re currently looking for volunteers of all skill levels. Everyone’s welcome!
