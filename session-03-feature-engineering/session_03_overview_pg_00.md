# Session 03 — Data Preparation & Feature Engineering

## Overview
Up to this point, we have explored the theoretical foundations of machine learning and the linear algebra that powers it. In this session, we transition from theory to practice.

Agricultural data is inherently messy. IoT sensors fail, farmers forget to log irrigation events, and extreme weather creates statistical outliers. If you feed raw data directly into a model, it will learn from the noise and fail in the real world. 

In this session, you will learn how to clean a dataset and, more importantly, how to use your agronomic expertise to engineer new, powerful features that make machine learning models work.

## Learning Outcomes
By the end of this session, you will be able to:
- **Diagnose** missing values and justify when to use median imputation.
- **Detect and cap** extreme outliers using the Interquartile Range (IQR) method.
- **Execute** a rigorous 4-step framework for feature engineering.
- **Engineer** interaction features (Temperature-Humidity Index), ratio features (N:P Ratio), and domain indices (Growing Degree Days).
- **Encode** categorical variables correctly, avoiding the Dummy Variable Trap.
- **Identify** and drop collinear features to prevent model overfitting.



## Key Vocabulary Preview
| Term | Short meaning |
|------|---------------|
| Imputation | Filling in missing data using statistical estimates (e.g., median). |
| IQR | Interquartile Range; a method for detecting and capping outliers. |
| Multicollinearity | When two or more features are highly correlated with each other. |
| One-Hot Encoding | Converting categorical labels into binary (0/1) columns. |
| Dummy Variable Trap | A mathematical error caused by keeping all one-hot encoded columns. |