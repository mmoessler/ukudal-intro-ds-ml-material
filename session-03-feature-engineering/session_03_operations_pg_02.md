
# Session 02 — Page 02

## Matrix Operations


**Learning outcomes:**
- Add and subtract matrices of matching dimensions
- Multiply matrices by scalars
- Perform matrix transposition
- Apply the shape rule for matrix multiplication
- Express linear regression in matrix form
- Check matrix shapes in Python before performing operations

---

## 1. Why This Matters

Matrix operations are the mechanics of machine learning. When a model makes 
predictions, it multiplies a data matrix by a weight vector. When it trains, 
it uses transpose and multiplication to compute the solution. When a neural 
network processes an image, it applies dozens of matrix multiplications in 
sequence.

Understanding shapes — and the rules for when operations are valid — is what 
separates working code from cryptic error messages.

> **Key idea:** Always check shapes before performing matrix operations. 
> Shape errors are the most common source of bugs in ML code.

---

## 2. Matrix Addition and Subtraction

Two matrices can be added or subtracted **only if their dimensions match**. 
Then, corresponding entries are added or subtracted.

### 2.1 Worked Example

$$
\begin{bmatrix} 6.8 & 2.3 \\ 6.2 & 1.8 \end{bmatrix}
+
\begin{bmatrix} 0.2 & 0.4 \\ 0.1 & 0.3 \end{bmatrix}
=
\begin{bmatrix} 7.0 & 2.7 \\ 6.3 & 2.1 \end{bmatrix}
$$

### 2.2 Agri-Food Example: Sensor Drift Correction

A sensor array records soil moisture at multiple locations over multiple days. 
The matrix $\mathbf{X}$ holds the raw readings. The matrix $\mathbf{D}$ 
holds the known drift offsets for each sensor on each day. The corrected 
readings are $\mathbf{X} - \mathbf{D}$.

If $\mathbf{X}$ and $\mathbf{D}$ have different shapes, Python raises a 
`ValueError`.

---

## 3. Scalar Multiplication and Division

A **scalar** is a single number. Multiplying a matrix by a scalar multiplies 
every entry.

### 3.1 Worked Example

$$
2 \times \begin{bmatrix} 6.8 & 2.3 \\ 6.2 & 1.8 \end{bmatrix}
=
\begin{bmatrix} 13.6 & 4.6 \\ 12.4 & 3.6 \end{bmatrix}
$$

$$
\begin{bmatrix} 6.8 & 2.3 \\ 6.2 & 1.8 \end{bmatrix} / 2
=
\begin{bmatrix} 3.4 & 1.15 \\ 3.1 & 0.9 \end{bmatrix}
$$

### 3.2 Agri-Food Examples

- Convert yields from t/ha to kg/ha: multiply $\mathbf{y}$ by 1000.
- Convert temperatures from °C to °F: $\mathbf{X}_F = 1.8 \mathbf{X}_C + 32$.
- Normalise rainfall to mm/day: divide by the number of days.

> **ML connection:** Feature scaling (standardisation, min-max normalisation) 
> is scalar multiplication applied column by column.

---

## 4. Matrix Transpose

The **transpose** of a matrix $\mathbf{A}$, written $\mathbf{A}^\top$, 
swaps rows and columns.

### 4.1 Worked Example

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
\quad\Longrightarrow\quad
\mathbf{A}^\top = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

### 4.2 Agri-Food Interpretation

- $\mathbf{X} \in \mathbb{R}^{n \times p}$: $n$ fields, $p$ features.
- $\mathbf{X}^\top \in \mathbb{R}^{p \times n}$: $p$ features, $n$ fields.

### 4.3 Where You Will See It

- Computing covariance matrices: $\mathbf{X}^\top \mathbf{X}$
- The normal equations for linear regression: 
  $\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$
- Dot product of two vectors: $\mathbf{x}^\top \mathbf{y}$

---

## 5. Matrix Multiplication

### 5.1 The Shape Rule

To multiply $\mathbf{A} \mathbf{B}$:
- The number of **columns of $\mathbf{A}$** must equal the number of 
  **rows of $\mathbf{B}$**.
- The result has shape (rows of $\mathbf{A}$) × (columns of $\mathbf{B}$).

$$
\underbrace{(m \times n)}_{\mathbf{A}}\ 
\underbrace{(n \times p)}_{\mathbf{B}}
= \underbrace{(m \times p)}_{\mathbf{A}\mathbf{B}}
$$

The **inner dimensions** must match; the **outer dimensions** give the result.

### 5.2 Worked Example

Compute $\mathbf{A}\mathbf{B}$ where:

$$
\mathbf{A} = \begin{bmatrix} 1 & 4 \\ 4 & 1 \\ 2 & 3 \\ 3 & 0 \end{bmatrix}
\quad
\mathbf{B} = \begin{bmatrix} 3 & 5 & 1 \\ 2 & 6 & 0 \end{bmatrix}
$$

Shape check: $(4 \times 2)(2 \times 3) \to (4 \times 3)$.

To compute entry $(i, j)$ of the result, take the dot product of row 
$i$ of $\mathbf{A}$ with column $j$ of $\mathbf{B}$.

Row 1 of A: $[1, 4]$. Column 1 of B: $[3, 2]$. Dot product: $1(3) + 4(2) = 11$.

Row 1 of A with column 2: $[1, 4] \cdot [5, 6] = 1(5) + 4(6) = 29$.

Row 1 of A with column 3: $[1, 4] \cdot [1, 0] = 1(1) + 4(0) = 1$.

Repeating for all rows:

$$
\mathbf{A}\mathbf{B} = \begin{bmatrix}
11 & 29 & 1 \\
14 & 26 & 4 \\
12 & 28 & 2 \\
9 & 15 & 3
\end{bmatrix}
$$

### 5.3 Matrix-Vector Multiplication

A special case where the second operand is a single-column matrix.

$$
\underbrace{(2 \times 4)}_{\mathbf{X}}\ 
\underbrace{(4 \times 1)}_{\mathbf{w}}
= \underbrace{(2 \times 1)}_{\mathbf{X}\mathbf{w}}
$$

**Worked example:**

$$
\begin{bmatrix} 3 & 5 & 1 & 4 \\ 2 & 6 & 0 & 7 \end{bmatrix}
\begin{bmatrix} 1 \\ 4 \\ 2 \\ 3 \end{bmatrix}
=
\begin{bmatrix} 3(1)+5(4)+1(2)+4(3) \\ 2(1)+6(4)+0(2)+7(3) \end{bmatrix}
=
\begin{bmatrix} 37 \\ 47 \end{bmatrix}
$$



### 5.4 Non-Commutativity

Matrix multiplication is **not commutative**: $\mathbf{A}\mathbf{B} \ne \mathbf{B}\mathbf{A}$.

In fact, one product may be defined while the other is not.

$$
\mathbf{A} = \begin{bmatrix} 1 & 4 \\ 4 & 1 \\ 2 & 3 \\ 3 & 0 \end{bmatrix}
\quad
\mathbf{B} = \begin{bmatrix} 3 & 5 & 1 \\ 2 & 6 & 0 \end{bmatrix}
$$

- $\mathbf{A}\mathbf{B}$: $(4 \times 2)(2 \times 3) \to (4 \times 3)$. **Defined.**
- $\mathbf{B}\mathbf{A}$: $(2 \times 3)(4 \times 2)$. **Undefined** (3 ≠ 4).

> **Agri-food takeaway:** When composing transformations — first normalise, 
> then project — order matters. Always check shapes.

---

## 6. Python Snippet: Matrix Operations

```python
import numpy as np

# Create matrices
A = np.array([
    [1, 4],
    [4, 1],
    [2, 3],
    [3, 0]
])

B = np.array([
    [3, 5, 1],
    [2, 6, 0]
])

# Shape check
print("A shape:", A.shape)  # (4, 2)
print("B shape:", B.shape)  # (2, 3)

# Matrix multiplication
C = A @ B  # or np.dot(A, B)
print("C shape:", C.shape)  # (4, 3)
print("C:\n", C)

# Transpose
print("A.T shape:", A.T.shape)  # (2, 4)

# Matrix-vector multiplication
w = np.array([1, 2])
result = A @ w
print("A @ w:", result)  # shape (4,)

# Scalar multiplication
print("2 * A:\n", 2 * A)

# Addition (requires matching shapes)
D = np.ones((4, 2))
print("A + D:\n", A + D)

# This will raise an error
try:
    bad = A + B  # shapes (4,2) and (2,3) don't match
except ValueError as e:
    print("Error:", e)
```
## 7. Key Ideas on Matrix Operations
1. Matrix addition requires matching dimensions; entries are added element-wise.
2. Scalar multiplication multiplies every entry — this is feature scaling.
3. Transpose swaps rows and columns — it appears in covariance matrices and the normal equations.
4. Matrix multiplication requires the inner dimensions to match: $(m×n)(n×p)=(m×p)(m×n)(n×p)=(m×p)$.
5. Matrix multiplication is not commutative — order matters.