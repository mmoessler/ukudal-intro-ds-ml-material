
# Session 02 — Page 03

## Special Matrices and Linear Systems
 **Learning outcomes:**
- Explain what an identity matrix is and why it matters
- Determine when a matrix has an inverse
- Explain collinearity and why it breaks linear regression
- Use the pseudo-inverse to solve least-squares problems
- Recognise systems of linear equations in ML contexts

---

## 1. Why This Matters

Linear regression — the foundation of much of machine learning — is a 
linear system. Solving it requires understanding when matrices can be 
inverted, and what to do when they cannot. In agricultural data, collinearity 
is common: rainfall and humidity are correlated, soil pH and organic carbon 
are correlated. These correlations can make the matrix 
$\mathbf{X}^\top \mathbf{X}$ singular, breaking the standard solution.

This page explains why this happens and what to do about it.

> **Key idea:** Not every matrix has an inverse. When it doesn't, we use the 
> pseudo-inverse or regularisation.

---

## 2. The Identity Matrix

The **identity matrix** $\mathbf{I}$ is a square matrix with 1s on the 
diagonal and 0s elsewhere.

$$
\mathbf{I}_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

**Property:** $\mathbf{A} \mathbf{I} = \mathbf{I} \mathbf{A} = \mathbf{A}$ 
for any compatible $\mathbf{A}$. It is the matrix equivalent of the number 1. 

**ML connection:** 
1. Regularisation adds a small multiple of $\mathbf{I}$ 
to a matrix to make it invertible and stabilise training. 
2. Ridge regression solves $(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$.
3. It core role in ML is mainly mathemtical stabiliser and reference point. It appears whenever models need to preserve information, define an inverse, or prevent numerical instability.
> For agrifood data, this matters because models often combine noisy, correlated, and seasonally dependent variables. The identity matrix helps keep those models numerically stable and prevents them from making extreme, unreliable predictions.
---
## 3. The Matrix Inverse

For a square matrix $\mathbf{A}$, its **inverse** $\mathbf{A}^{-1}$ 
satisfies:

$$
\mathbf{A} \mathbf{A}^{-1} = \mathbf{A}^{-1} \mathbf{A} = \mathbf{I}
$$

**Scalar analogy:** $4 \times \frac{1}{4} = 1$.

### 3.1 Worked Example

For a 2 × 2 matrix:

$$
\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
\quad\Longrightarrow\quad
\mathbf{A}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

**Example:**

$$
\mathbf{A} = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}
\quad\Longrightarrow\quad
\mathbf{A}^{-1} = \frac{1}{4(6) - 7(2)} \begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix}
= \frac{1}{10} \begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix}
$$

### 3.2 When Does an Inverse Exist?

- **Non-square matrices** never have an inverse.
- **Singular matrices** (square but with linearly dependent columns) have 
  no inverse.

A matrix is singular if its columns are collinear — that is, one column is a 
linear combination of others.

### 3.3 Collinearity in Agricultural Data

Collinearity is common in observational agricultural data:

- Rainfall and humidity are correlated.
- Soil pH and organic carbon are correlated.
- Temperature and growing degree days are correlated.

If two columns of $\mathbf{X}$ contain nearly the same information, 
$\mathbf{X}^\top \mathbf{X}$ becomes nearly singular. Linear regression 
becomes unstable — small changes in the data produce large changes in the 
coefficients.

**Solutions:**
- Drop one of the collinear features.
- Use Ridge regression (adds $\lambda \mathbf{I}$).
- Use the pseudo-inverse.

---

## 4. The Pseudo-Inverse

When a matrix has no inverse (non-square or singular), we use the 
**Moore-Penrose pseudo-inverse** $\mathbf{A}^{+}$.

**Key property:** $\mathbf{A} \mathbf{A}^{+} \mathbf{A} = \mathbf{A}$.

For a tall matrix $\mathbf{X}$ ($n > p$):

$$
\mathbf{X}^{+} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top
$$

**ML connection:** The least-squares solution to linear regression is:
$$
 \hat{\boldsymbol{\theta}} = \mathbf{X}^{+} \mathbf{y} = 
(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$
> This is what `np.linalg.lstsq` and `sklearn.LinearRegression` compute internally.
(More on this later)

---

## 5. Systems of Linear Equations

Many ML problems reduce to solving:

$$
\mathbf{A} \mathbf{x} = \mathbf{b}
$$

### 5.1 Agri-Food Example: Calibrating a Sensor

On three plots, we apply different amounts of nitrogen $x$ (kg/ha) and 
observe yield $y$ (t/ha). Suppose the relationship is $y = a + b x$.

We have three observations:

$$
a + 60b = 3.2,\quad
a + 90b = 4.1,\quad
a + 120b = 4.9
$$

In matrix form:

$$
\underbrace{\begin{bmatrix} 1 & 60 \\ 1 & 90 \\ 1 & 120 \end{bmatrix}}_{\mathbf{A}}
\underbrace{\begin{bmatrix} a \\ b \end{bmatrix}}_{\mathbf{x}}
=
\underbrace{\begin{bmatrix} 3.2 \\ 4.1 \\ 4.9 \end{bmatrix}}_{\mathbf{b}}
$$

The system is **overdetermined**: 3 equations, 2 unknowns. There is no exact 
solution. The least-squares solution is $\mathbf{x} = \mathbf{A}^{+} \mathbf{b}$.

### 5.2 Solving Systems in Practice

- **Square, invertible:** $\mathbf{x} = \mathbf{A}^{-1} \mathbf{b}$. Rarely 
  done this way — numerically unstable.
- **Numerical solutions:** Gaussian elimination, LU, QR, SVD. Python NumPy functions: 
  `np.linalg.solve` solves a system of linear equations, and `np.linalg.lstsq` solves linear-squares problem.

---

## 6. Python Snippet: Inverses and Least Squares

```python
import numpy as np

# 1. Identity matrix
I = np.eye(3)
print("Identity:\n", I)

# 2. Matrix inverse
A = np.array([[4, 7], [2, 6]])
A_inv = np.linalg.inv(A)
print("A inverse:\n", A_inv)
print("A @ A_inv:\n", A @ A_inv)  # should be identity

# 3. Pseudo-inverse
X = np.array([
    [1, 60],
    [1, 90],
    [1, 120]
])
X_pinv = np.linalg.pinv(X)
print("Pseudo-inverse shape:", X_pinv.shape)  # (2, 3)

# 4. Least-squares solution
y = np.array([3.2, 4.1, 4.9])
theta = X_pinv @ y
print("Fitted theta (a, b):", theta)

# 5. Using lstsq directly
theta2, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print("lstsq theta:", theta2)
```
## 7. Key Ideas on Special Matrices
1. The identity matrix $\mathbf{I}$ is the matrix equivalent of 1.
2. Not every matrix has an inverse. Non-square and singular matrices do not.
3. Collinearity (correlated features) makes $\mathbf{X}^\top\mathbf{X}$ singular — common in agricultural data.
4. The pseudo-inverse solves least-squares problems when the inverse does not exist.
5. Ridge regression adds $\lambda\mathbf{I}$ to stabilise the solution.
