
# Session 02 — Page 04

## Eigenvalues, Norms, and Transformations

 
**Learning outcomes:**
- Explain what eigenvalues and eigenvectors represent
- Describe how PCA uses eigenvalues for dimensionality reduction
- Compute norms and distances between vectors
- Recognise linear transformations in ML contexts
- Understand how these concepts connect to agri-food applications

---

## 1. Why This Matters

Eigenvalues and eigenvectors power Principal Component Analysis (PCA), one of the most widely used dimensionality reduction methods in ML. Norms define distance metrics used in clustering and anomaly detection. Linear transformations explain how neural networks process information.

These concepts are not abstract mathematics — they are the tools that make modern ML possible.

> **Key idea:** Eigenvalues tell you which directions in your data contain 
> the most information. Norms tell you how similar two samples are.

---

## 2. Eigenvalues and Eigenvectors

For a square matrix $\mathbf{A}$, an **eigenvector** $\mathbf{v}$ and **eigenvalue** $\lambda$ satisfy:

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
$$

**Interpretation:** When $\mathbf{A}$ is applied to $\mathbf{v}$, the 
vector is stretched by $\lambda$ without rotating.

### 2.1 Worked Example

$$
\mathbf{A} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}
$$

The eigenvalues are 2 and 3. The eigenvectors are:

$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix},\quad
\mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

Verification: $\mathbf{A}\mathbf{v}_1 = [2, 0]^\top = 2\mathbf{v}_1$.

### 2.2 PCA: Dimensionality Reduction with Eigenvalues

Principal Component Analysis (PCA) finds the eigenvectors of the covariance 
matrix of $\mathbf{X}$. The eigenvalues tell you how much variance each component explains.

**Agri-food example:** You measure 10 soil properties for 500 fields. They 
are highly correlated. PCA finds new axes (components) that capture most of 
the variation with fewer numbers.

| Component | Eigenvalue | % Variance | Cumulative % |
|-----------|------------|------------|--------------|
| PC1 | 5.4 | 54% | 54% |
| PC2 | 2.1 | 21% | 75% |
| PC3 | 1.2 | 12% | 87% |
| PC4 | 0.6 | 6% | 93% |
| ... | ... | ... | ... |

We can keep only 3 components (87% of variance) instead of 10 variables. 
This is used for clustering fields into soil-management zones.

**Additional agri-food applications of PCA:**
- Reduce multi-spectral satellite bands to a few informative dimensions.
- Visualise high-dimensional genotype data.
- Compress sensor data from IoT networks.

---

## 3. Norms and Distance Metrics

A **norm** measures the length of a vector. The most common is the Euclidean 
norm:

$$
\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_p^2}
$$

### 3.1 Worked Example

$$
\mathbf{x} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
\quad\Longrightarrow\quad
\|\mathbf{x}\|_2 = \sqrt{9 + 16} = 5
$$

### 3.2 Agri-Food Applications

- **Similarity of two soil samples:** $\|\mathbf{x}_i - \mathbf{x}_j\|_2$
- **Anomaly detection:** Fields whose feature vector is far from the average are candidates for review
- **Loss functions:** MSE is the squared Euclidean norm of residuals
- **Regularisation:** L1 (Lasso) and L2 (Ridge) use $\ell_1$ and $\ell_2$ norms

---

## 4. Vector Spaces and Basis

A **vector space** is a set of vectors closed under addition and scalar 
multiplication. A **basis** is a minimal set of vectors that spans the space.

> **ML connection:** Feature vectors live in $\mathbb{R}^p$. Models find 
> a useful *basis* for prediction. Neural network layers compute new bases 
> through learned weights.

**Agri-food example:** Raw spectral bands (B2, B3, B4, ...) form one basis. 
Vegetation indices (NDVI, EVI, SAVI) form another, more informative basis for crop health prediction.

---

## 5. Linear Transformations

A matrix is represents a **linear transformation**: multiply a vector by a matrix, it stretches, rotates, or projects vectors.

### 5.1 Agri-Food Examples

- **Rotation:** Aligning drone imagery with a map.
- **Projection:** PCA projects high-dimensional soil data onto a 2D plane.
- **Scaling:** Standardising features across fields.
- **Convolution:** A CNN layer applies many small matrix transformations to 
  an image.

> **Key insight:** A deep neural network is a composition of linear 
> transformations interleaved with non-linearities. The linear part is all 
> matrices.

---

## 6. Python Snippet: Eigenvalues, PCA, and Norms

```python
import numpy as np

# 1. Eigenvalues of a simple matrix
A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)  # [5., 2.]
print("Eigenvectors:\n", eigenvectors)

# 2. PCA on synthetic soil data
np.random.seed(42)
X = np.random.randn(100, 5)  # 100 samples, 5 features

# Centre the data
X_centered = X - X.mean(axis=0)

# Covariance matrix
cov = np.cov(X_centered.T)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov)
print("Eigenvalues (sorted):", eigenvalues)

# Explained variance
explained = eigenvalues / eigenvalues.sum()
print("Explained variance:", explained)
print("Cumulative:", np.cumsum(explained))

# 3. Norms and distances
x = np.array([3, 4])
print("Euclidean norm:", np.linalg.norm(x))  # 5.0

# Distance between two soil samples
sample1 = np.array([6.8, 2.3, 1.5])
sample2 = np.array([6.2, 1.8, 1.2])
distance = np.linalg.norm(sample1 - sample2)
print("Distance between samples:", distance)
```

## 7. Key Ideas on Special Matrices
1. Eigenvalues and eigenvectors describe how a matrix stretches or rotates vectors.
2. PCA uses eigenvalues to find the directions of maximum variance in data.
3. Norms measure vector length; they are used in distance metrics, loss functions, and regularisation.
4. Vector spaces and bases provide the language for understanding feature spaces.
5. A neural network is a composition of linear transformations and non-linearities.



---

## References (Interactive Resources)
- https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab 
- https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF
- https://youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&si=i6fibUdgSR2nUy48
- https://www.youtube.com/@statquest
- https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
- https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/ 
- Deisenroth, Faisal, Ong (2020) Mathematics for Machine Learning, _(Free PDF at https://mml-book.github.io)_, Chapters 2–4, 10
- Gilbert Strang (2023), Introduction to Linear Algebra (6th ed.) Chapters 1–5, _Wellesley-Cambridge Press_
- Gilbert Strang (2019)  Linear Algebra and Learning from Data, _Wellesley-Cambridge Press_
- Zhang et al. Dive into Deep Learning, _(Free at https://d2l.ai)_ Sections 2.3, 22.1
- Md. Jalil Piran (2026), Linear Algebra with Applications in Machine Learning, _Springer_ (Free on https://github.com/jalil-piran/Linear-Algebra-with-Applications-in-Machine-Learning )







