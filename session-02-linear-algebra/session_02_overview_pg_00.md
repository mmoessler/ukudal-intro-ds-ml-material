# Session 02 — Linear Algebra for Machine Learning

## Overview

Every machine learning (ML) model — from simple linear regression to deep neural 
networks — is built on linear algebra. When a model predicts maize yield, it 
computes a dot product. When it trains, it minimises a loss over a matrix of 
data. When it reduces a 20-band satellite image to 3 informative dimensions, 
it finds eigenvectors of a covariance matrix.

This session builds the mathematical vocabulary you need to read, interpret, 
and debug ML notation. You do not need to derive anything 
by hand. You just need to understand shapes, operations, and how they 
connect to your research.

---

## Learning Outcomes

By the end of this session, you will be able to:

- **Explain** what a matrix and a vector are in the context of agricultural data
- **Identify** the dimensions of a dataset and interpret rows as samples and columns as features
- **Perform** basic matrix operations in Python using NumPy
- **Interpret** the shape rules for matrix multiplication
- **Describe** the role of eigenvalues in PCA and dimensionality reduction
- **Recognise** when a matrix is singular and why this matters for regression

---


## Key Vocabulary Preview

| Term | Short meaning |
|------|---------------|
| Matrix | 2D array of numbers; rows are samples, columns are features |
| Vector | A matrix with one column (or one row) |
| Dimension | Shape of a matrix: (rows × columns) |
| Transpose | Swapping rows and columns |
| Dot product | Sum of element-wise products of two vectors |
| Eigenvalue | Scalar λ such that Av = λv |
| Eigenvector | Vector v such that Av = λv |
| Norm | A measure of vector length |
| Identity matrix | Square matrix with 1s on diagonal, 0s elsewhere |
| Inverse | Matrix A⁻¹ such that AA⁻¹ = I |
| Pseudo-inverse | Generalised inverse for non-square or singular matrices |
| Singular matrix | Matrix with no inverse (collinear columns) |
| Collinearity | When two features contain nearly the same information |

