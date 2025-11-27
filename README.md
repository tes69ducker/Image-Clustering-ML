# 🎯 Image Clustering ML

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

ML-based image clustering using **unsupervised learning**. Custom **K-Means implementation** with **Cosine Similarity** for high-dimensional feature spaces.

**Demo:** 603 flower images | **Accuracy:** 82%+ Rand Score | **Speed:** ~3 seconds

---

## 🎯 Overview

Unsupervised clustering algorithm that **dynamically discovers** optimal cluster count based on feature similarity. Unlike traditional K-Means (requires predefined K), this implementation adapts to data structure automatically.

**Key Innovation:** Cosine similarity in high-dimensional space + dynamic threshold-based clustering

---

## ✨ Features

- 🎨 **Dynamic Clustering** - Automatic cluster discovery (no predefined K needed)
- 🧮 **Cosine Similarity** - Robust for high-dimensional image features  
- 🔄 **Iterative Refinement** - Converges in 3-5 iterations typically
- 🧹 **Noise Filtering** - Removes clusters below minimum size
- 📊 **Rand Score: 0.82+** - Strong agreement with ground truth
- ⚙️ **Configurable** - JSON-based parameter tuning

---

## 🧠 Algorithm

```python
1. Load normalized feature vectors
2. For each iteration (max 10):
   • Assign to clusters (similarity > 0.60 threshold)
   • Create new cluster if no match
   • Recalculate centroids (mean + normalize)
   • Check convergence (atol=1e-4)
3. Filter small clusters (min size = 10)
4. Evaluate with Rand Score
```

**Why Cosine Similarity?**
- Measures vector direction, not magnitude
- Superior for high-dimensional spaces
- Invariant to vector length

---

## 📊 Demo Dataset

**603 Flower Images** across 20 species

- Format: PNG images + pre-extracted features (`.pkl`)
- Ground truth labels for evaluation
- **Algorithm-agnostic:** Works with any feature vectors

---

## 🚀 Quick Start

```bash
# Clone & Install
git clone https://github.com/Dan-Ofri/Image-Clustering-ML.git
cd Image-Clustering-ML
pip install -r requirements.txt

# Run
python main.py
```

### Configuration (`config.json`)

```json
{
  "features_file": "data/flowers/image-features.pkl",
  "labels_file": "data/flowers/flowers-solution.csv",
  "min_cluster_size": 10,
  "max_iterations": 10
}
```


## 📈 Results

| Metric | Value |
|--------|-------|
| **Rand Score** | 0.82+ |
| **Clusters Found** | ~18 (from 20 actual) |
| **Execution Time** | ~3 seconds |
| **Convergence** | 3-5 iterations |

---

## 📁 Project Structure

```
Image-Clustering-ML/
├── main.py              # Clustering algorithm
├── utils.py             # Evaluation functions
├── config.json          # Parameters
├── requirements.txt     # Dependencies
└── data/flowers/
    ├── image-features.pkl
    ├── flowers-solution.csv
    └── images/          # 603 PNGs
```
```

---

## 🛠️ Tech Stack

**Python 3.8+** | **NumPy** | **scikit-learn** | **Pandas**

```python
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import rand_score
```

---

## 👨‍💻 Author

**Dan Ofri** • [@Dan-Ofri](https://github.com/Dan-Ofri) • ofridan@gmail.com

**Course:** Computational Learning with Python | **Year:** 2025

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🔮 Potential Extensions

- Additional datasets (animals, vehicles, faces)
- t-SNE/PCA visualization
- Alternative metrics (Euclidean, Manhattan)
- CNN-based feature extraction
- GPU acceleration
- Web interface

---

⭐ **Star this repo if you found it useful!**
