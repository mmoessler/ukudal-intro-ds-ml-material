# Session 02 — page 01

## Matrices and Vectors

**Learning outcomes:**
- Explain what a matrix is and how it represents agricultural data
- Identify the dimensions of a dataset as a matrix
- Distinguish between row and column vectors
- Read and write ML notation for matrices and vectors
- Index into matrices and vectors correctly in Python

---

## 1. Why This Matters

Every dataset you have ever worked with is a matrix. A soil survey with 
50 samples and 12 measured properties is a 50 × 12 matrix. A yield trial 
with 200 plots and 5 management variables is a 200 × 5 matrix. A time series 
of daily weather for 3 years is a 1,095 × 6 matrix (if you have 6 variables).

Machine learning does not see "fields" or "plots" or "days." It sees matrices. 
Understanding this representation is the first step toward understanding how 
ML algorithms work.

> **Key idea:** Every ML algorithm expects data in matrix form: rows are 
> samples, columns are features.

---

## 2. What is a Matrix?

A **matrix** is a two-dimensional array of numbers arranged in rows and columns.

### 2.1 Agri-Food Example: A Soil Dataset

Consider a soil survey with four sampling points. At each point, you measure 
four properties: pH, organic carbon (OC %), nitrogen (N %), and clay content (%).

| Sample | pH | OC | N | Clay |
|--------|-----|-----|-----|------|
| Field 1 | 6.8 | 2.3 | 1.5 | 45 |
| Field 2 | 6.2 | 1.8 | 1.2 | 52 |
| Field 3 | 7.1 | 2.7 | 1.9 | 38 |
| Field 4 | 5.9 | 1.4 | 0.9 | 61 |

This table is a **4 × 4 matrix**: 4 samples, 4 features.

In mathematical notation:

$$
\mathbf{X} = \begin{bmatrix}
6.8 & 2.3 & 1.5 & 45 \\
6.2 & 1.8 & 1.2 & 52 \\
7.1 & 2.7 & 1.9 & 38 \\
5.9 & 1.4 & 0.9 & 61
\end{bmatrix}
$$

### 2.2 Notation and Conventions

- **Capital letters** (X, A, W) denote matrices.
- **Dimensions** are written as **rows × columns**.
- A matrix with $m$ rows and $n$ columns is written as $\mathbf{A} \in \mathbb{R}^{m \times n}$.

For the soil dataset: $\mathbf{X} \in \mathbb{R}^{4 \times 4}$.

### 2.3 The Universal ML Convention

In machine learning, the feature matrix is nearly always denoted $\mathbf{X}$:

$$
\mathbf{X} \in \mathbb{R}^{n \times p}
$$

where:
- \(n\) = number of samples (rows)
- \(p\) = number of features (columns)

The target vector (what we want to predict) is denoted $\mathbf{y}$:

$$
\mathbf{y} \in \mathbb{R}^{n}
$$

> **Key idea:** The feature matrix X and the target vector y are the two 
> fundamental objects in supervised learning.

---

## 3. What is a Vector?

A **vector** is a matrix with only one column (by convention). Vectors are 
named with lowercase bold letters: $\mathbf{x}$, $\mathbf{w}$, $\mathbf{y}$.

### 3.1 Two Views of a Vector

**View 1: A row of X — one sample's features.**

For Field 3:
$$
\mathbf{x}_3 = \begin{bmatrix} 7.1 \\ 2.7 \\ 1.9 \\ 38 \end{bmatrix}
$$

This vector describes all features of a single field.

**View 2: A column of X — one feature across all samples.**

The pH values across all four fields:
$$
\mathbf{x}_{\text{pH}} = \begin{bmatrix} 6.8 \\ 6.2 \\ 7.1 \\ 5.9 \end{bmatrix}
$$

This vector describes one property across all samples.

### 3.2 Dimensions of a Vector

A vector with \(n\) elements has dimensions $n \times 1$.

- $\mathbf{y} \in \mathbb{R}^4$ means a vector with 4 elements.
- $\mathbf{w} \in \mathbb{R}^3$ means a weight vector with 3 weights.

---

## 4. Indexing: Accessing Individual Elements

### 4.1 Matrix Indexing

Individual matrix entries are accessed by two indices: $x_{ij}$ is the value 
in row $i$, column $j$.

For our soil matrix:
- $x_{1,1} = 6.8$ — pH of Field 1
- $x_{2,3} = 1.2$ — nitrogen of Field 2
- $x_{4,4} = 61$ — clay content of Field 4
- $x_{3,5}$ — does not exist (only 4 columns)

> **Warning:** Mathematical notation is **1-indexed**. Python is **0-indexed**. \
> In Python, `X[0, 0]` gives $x_{1,1}$, and `X[3, 3]` gives $x_{4,4}$. \
> Getting this wrong is a common source of bugs.

### 4.2 Vector Indexing

Vectors are indexed by a single number: $x_i$ is the $i$-th element.

For the yield vector $\mathbf{y} = [4.5, 3.8, 5.1, 4.2]$:
- $y_1 = 4.5$ — yield of Field 1
- $y_3 = 5.1$ — yield of Field 3
- $y_5$ — does not exist (only 4 elements)

---

## 5. Python Snippet: Matrices in NumPy

```python
import numpy as np

# Create a matrix: 4 soil samples, 4 features
X = np.array([
    [6.8, 2.3, 1.5, 45],
    [6.2, 1.8, 1.2, 52],
    [7.1, 2.7, 1.9, 38],
    [5.9, 1.4, 0.9, 61]
])

print("Shape of X:", X.shape)          # (4, 4)
print("Number of samples:", X.shape[0])  # 4
print("Number of features:", X.shape[1]) # 4

# Access individual elements
print("pH of Field 1:", X[0, 0])         # 6.8
print("Nitrogen of Field 2:", X[1, 2])   # 1.2
print("Clay of Field 4:", X[3, 3])       # 61

# Create a vector
y = np.array([4.5, 3.8, 5.1, 4.2])      # yield in t/ha
print("Shape of y:", y.shape)            # (4,)

# Access vector elements
print("Yield of Field 1:", y[0])         # 4.5
print("Yield of Field 3:", y[2])         # 5.1
```

**Note:** 
- $X.shape$ returns $(4, 4)$ — a tuple of (rows, columns)
- $X[0, 0]$ accesses the first element (zero-indexed)
- $y.shape$ returns (4,) — a 1D array


## 6. Worked Example: Framing a Yield Prediction Problem
Suppose you have 50 maize fields. For each field, you have:

- Rainfall (mm)

- Mean temperature (°C)

- Nitrogen applied (kg/ha)

- Soil pH

- Yield (t/ha)

**Question:** How would you write this as a matrix and a target vector?

**Answer:**

- Feature matrix: $\mathbf{y} \in \mathbb{R}^{50 \times 4}$ (50 samples, 4 features: rainfall, temperature, nitrogen, pH)
- Target vector: $\mathbf{y} \in \mathbb{R}^{50}$  (50 yields)

**In python**
```python
# Load your dataset
data = pd.read_csv("maize_fields.csv")

# Feature matrix (all columns except yield)
X = data[["rainfall", "temperature", "nitrogen", "ph"]].values

# Target vector
y = data["yield"].values

print("X shape:", X.shape)  # (50, 4)
print("y shape:", y.shape)  # (50,)
```
## 7. Key Ideas on Matrices

1. Every agri-food dataset is a matrix: rows are samples, columns are features.
2. A vector is a matrix with one column — either one sample's features or one feature across samples.
3. In ML notation, $\mathbf{X} \in \mathbb{R}^{n \times p}$ means $n$ samples and $p$ features.
4. Mathematical indexing is 1-based; Python indexing is 0-based. Always check.
5. The feature matrix $\mathbf{X}$ and target vector $\mathbf{y}$ are the fundamental objects in supervised learning.