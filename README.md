# 🎃 Sp00kyVectors: Haunted Vector Analysis for the Living and the Dead 📈💀

Welcome to **Sp00kyVectors**, the eerily intuitive Python class for **vector analysis**, **statistical computation**, and **sinister visualizations** — all coded to thrill and analyze! 🪄👻

Whether you’re chasing spectral patterns or just need basic stats with dramatic flair, this library’s got your back (from beyond).

---

## 🧠 Features

- 🧮 **Vector Magic**:
  - Load 1D or 2D arrays into Vector objects
  - X/Y decomposition for 2D data

- 📊 **Statistical Potions**:
  - Mean, median, standard deviation 💀
  - Probability vectors and PDFs 🧪
  - Z-score normalization 🧼
  - Entropy between aligned vectors 🌀
  - Entropy within vector

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

```python
from sp00kyvectors import Vector

v = Vector([1, 2, 3, 4, 5])
print(v.mean())  # Output: 3.0

v2 = Vector([1, 1, 1, 1, 6])
print(v.entropy(v2))  # Output: spooky entropy value
```

---

## 📦 Installation

```bash
pip install sp00kyvectors
```

Or summon it from your own local clone:

```bash
git clone https://github.com/yourname/sp00kyvectors.git
cd sp00kyvectors
pip install .
```

---

## 📚 Documentation

Coming soon! For now, check out the code and let your curiosity guide you through the crypts.

---

## 👻 Contributing

Spirits and sorcerers of all levels are welcome. Open an issue, fork the repo, or summon a pull request.

---

## 🧛 License

MIT — you’re free to haunt this code as you wish as long as money is never involved! 

---

✨ Stay spooky, and may your vectors always point toward the unknown. 🕸️
