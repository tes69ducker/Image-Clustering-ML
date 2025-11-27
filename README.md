# 🎯 Image Clustering ML

```
  ___                              ____ _           _            _             
 |_ _|_ __ ___   __ _  __ _  ___  / ___| |_   _ ___| |_ ___ _ __(_)_ __   __ _ 
  | || '_ ` _ \ / _` |/ _` |/ _ \| |   | | | | / __| __/ _ \ '__| | '_ \ / _` |
  | || | | | | | (_| | (_| |  __/| |___| | |_| \__ \ ||  __/ |  | | | | | (_| |
 |___|_| |_| |_|\__,_|\__, |\___| \____|_|\__,_|___/\__\___|_|  |_|_| |_|\__, |
                      |___/                                              |___/ 
        __  __ _     
       |  \/  | |    
       | |\/| | |    
       | |  | | |___ 
       |_|  |_|_____|
```

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**ML-based image clustering** using unsupervised learning algorithms. Features a custom **K-Means-style implementation** with **Cosine Similarity** for high-dimensional feature spaces.

**Demonstrated on 603 flower images** achieving 82%+ clustering accuracy (Rand Score).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Algorithm](#algorithm)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Author](#author)
- [License](#license)

---

## 🎯 Overview

This project implements an **unsupervised clustering algorithm** for grouping images based on visual feature similarity. Instead of using pre-trained models or standard K-Means, it implements a **custom clustering approach** that:

- Uses **Cosine Similarity** to measure feature vector similarity
- Dynamically creates clusters based on a minimum similarity threshold
- Iteratively refines centroids until convergence
- Filters out small clusters to reduce noise
- Evaluates results using the **Rand Score** metric

### Why This Approach?

Traditional K-Means requires knowing the number of clusters in advance. This implementation **dynamically discovers** the optimal number of clusters based on feature similarity, making it more flexible for datasets where the number of categories is unknown.

### Current Demonstration

The algorithm is **demonstrated on a dataset of 603 flower images**, showcasing its effectiveness in Computer Vision applications. The modular design allows for easy adaptation to other image datasets.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎨 **Image Dataset** | Demonstrated on 603 flower images with diverse visual features |
| 🧮 **Cosine Similarity** | Robust similarity metric for high-dimensional feature vectors |
| 🔄 **Iterative Refinement** | K-Means-style centroid recalculation until convergence |
| 🎯 **Dynamic Clustering** | Automatic cluster creation based on similarity threshold |
| 🧹 **Noise Filtering** | Removes small clusters below minimum size threshold |
| 📊 **Performance Metrics** | Rand Score evaluation against ground truth labels |
| ⚙️ **Configurable** | JSON-based configuration for easy parameter tuning |

---

## 🧠 Algorithm

### Clustering Process

```
1. Load normalized feature vectors from pickle file
2. Initialize empty centroids list
3. For each iteration (max 10):
   a. Assign images to clusters:
      - Compare each image vector to all centroids using cosine similarity
      - If similarity > threshold (0.60), assign to best matching cluster
      - Otherwise, create new cluster with image as centroid
   b. Recalculate centroids:
      - For each cluster, compute mean of all member vectors
      - Normalize centroid vectors (unit length)
   c. Check convergence:
      - If centroids haven't changed significantly, stop
4. Filter out clusters smaller than minimum size (10 images)
5. Evaluate using Rand Score
```

### Cosine Similarity

The algorithm uses **cosine similarity** instead of Euclidean distance:

```python
def cosine(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))
```

**Why Cosine Similarity?**
- Measures angle between vectors, not magnitude
- Better for high-dimensional feature spaces (common in image features)
- Invariant to vector length, focusing on direction

### Convergence Criteria

The algorithm stops when:
- Centroids don't change significantly between iterations (`atol=1e-4`)
- Maximum iterations reached (10)

---

## 📊 Dataset

### Demo: Flower Images

The current implementation is demonstrated on a flower image dataset:

- **Total Images**: 603 flower photographs  
- **Categories**: Multiple flower species (dynamically discovered by algorithm)
- **Format**: PNG images organized by category (00-19)
- **Features**: Pre-extracted feature vectors (stored in `image-features.pkl`)

The algorithm is **dataset-agnostic** and can be applied to any image collection with pre-extracted features.

### Data Structure

```
data/
└── flowers/
    ├── image-features.pkl      # Pre-extracted feature vectors
    ├── flowers-solution.csv    # Ground truth labels for evaluation
    └── images/                 # 603 flower images
        ├── 00_001.png ... 00_046.png
        ├── 01_001.png ... 01_029.png
        ├── ...
        └── 19_001.png ... 19_035.png
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Dan-Ofri/Image-Clustering-ML.git
cd Image-Clustering-ML
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Quick Start

Run the clustering algorithm with default configuration:

```bash
python main.py
```

### Configuration

Edit `config.json` to adjust parameters:

```json
{
  "features_file": "data/flowers/image-features.pkl",
  "labels_file": "data/flowers/flowers-solution.csv",
  "min_cluster_size": 10,
  "max_iterations": 10
}
```

**Parameters:**
- `features_file`: Path to pre-extracted feature vectors
- `labels_file`: Path to ground truth labels (for evaluation)
- `min_cluster_size`: Minimum images per cluster (filters noise)
- `max_iterations`: Maximum clustering iterations

### Example Output

```
starting clustering images in file data/flowers/image-features.pkl
clustered images in labeled data: 603
clusters in solution: 18 and actual: 20
clustered in solution: 603 and actual: 603, members in common: 603
rand score for 603 members: 0.8234
total time: 3.0 sec
```

---

## 📈 Results

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Rand Score** | 0.82+ | Agreement with ground truth clustering |
| **Clusters Found** | ~18 | Dynamically discovered clusters |
| **Execution Time** | ~3s | Total clustering time |
| **Convergence** | 3-5 iterations | Typical iterations until convergence |

### Interpretation

- **Rand Score of 0.82+** indicates strong agreement with ground truth
- Algorithm successfully groups similar flowers together
- Dynamic cluster discovery finds optimal number of categories
- Fast execution suitable for real-time applications

---

## 📁 Project Structure

```
Image-Clustering-ML/
│
├── main.py              # Main clustering algorithm
├── utils.py             # Evaluation and utility functions
├── config.json          # Configuration parameters
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
└── data/
    └── flowers/                      # Demo dataset
        ├── image-features.pkl        # Pre-extracted feature vectors
        ├── flowers-solution.csv      # Ground truth labels
        └── images/                   # 603 flower images
            ├── 00_*.png
            ├── 01_*.png
            └── ...
```

---

## 🛠️ Technologies

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Programming language |
| **NumPy** | Numerical computing, vector operations |
| **Pickle** | Feature vector serialization |
| **scikit-learn** | Rand Score evaluation metric |
| **Pandas** | CSV data handling |

### Key Libraries

```python
import numpy as np                    # Vector operations
from numpy.linalg import norm         # Vector normalization
from sklearn.metrics import rand_score # Clustering evaluation
import pickle                         # Data loading
import pandas as pd                   # CSV handling
```

---

## 👨‍💻 Author

**Dan Ofri**

- GitHub: [@Dan-Ofri](https://github.com/Dan-Ofri)
- Email: ofridan@gmail.com

**Course**: Computational Learning with Python  
**Institution**: Computer Science Department  
**Year**: 2025

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎓 Academic Context

This project was developed as part of the **Computational Learning with Python** course (Assignment 4), demonstrating understanding of:

- Unsupervised learning algorithms
- Feature-based image clustering
- Similarity metrics (Cosine Similarity)
- Algorithm convergence and optimization
- Performance evaluation (Rand Score)

---

## 🔮 Potential Extensions

- [ ] Support for additional image datasets (animals, vehicles, faces)
- [ ] Visualization of clusters (t-SNE/PCA projection)
- [ ] Alternative similarity metrics (Euclidean, Manhattan)
- [ ] Real-time feature extraction (CNN-based)
- [ ] Interactive cluster exploration tool
- [ ] Comparison with standard K-Means and DBSCAN
- [ ] GPU acceleration for larger datasets
- [ ] Web interface for uploading custom datasets

---

**⭐ If you found this project helpful, please give it a star!**
