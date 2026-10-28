# Session 03 — Page 04: Application Task


## Task Overview
Using the `sf24data.csv` dataset, you are to act as both an agronomist and a data scientist. You will clean the dataset, perform EDA, and engineer the features we discussed in the office hour session.

## Deliverables
Submit a Jupyter Notebook (or Google Colab link) with the following sections clearly marked:

1. **Data Cleaning:**
   - Check for missing values. Impute missing `Rainfall (mm)` with the median.
   - Use the IQR method to cap outliers in the `Rainfall (mm)` column. Print the upper bound.

2. **Feature Engineering:**
   - Create `THI` using the formula: `(1.8 * Temperature) + 32 - (0.55 - 0.0055 * Humidity) * (1.8 * Temperature - 26)`.
   - Create `N_P_Ratio`.
   - Create `Total_NPK`.
   - Create `GDD` (assume a base temperature of 10°C).

3. **Categorical Encoding:**
   - Use `pd.get_dummies()` to one-hot encode `Soil Type`. (Use `drop_first=True`).
   - Use a mapping dictionary to ordinal encode `Growth Stage` (1, 2, 3).

4. **Collinearity Check:**
   - Calculate the correlation matrix for all numerical features.
   - Identify any features with a correlation \(|r| > 0.85\). Drop them and print the final shape of your dataset.


## References & Further Reading


## 1. Core Textbooks (Data Preparation & Feature Engineering)



*   **Kuhn, M., & Johnson, K. (2019).** *Feature Engineering and Selection: A Practical Approach for Predictive Models*. Chapman & Hall/CRC.
    *   **Why read it:** The gold standard for data preprocessing and feature engineering. While the code examples are in R, the conceptual explanations of data splitting, missing value handling, and feature selection are invaluable.
*   **Zheng, A., & Casari, A. (2018).** *Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists*. O'Reilly Media.
    *   **Why read it:** Excellent, highly readable introduction to how to transform raw data (text, categorical, numerical) into model-ready features. Chapter 2 on numerical data is particularly relevant to our THI and Ratio examples.
*   **Duboue, P. (2020).** *The Art of Feature Engineering: Essentials for Machine Learning*. Cambridge University Press.
    *   **Why read it:** A highly practical guide with Python code examples. It covers the exact types of transformations we discussed (interaction features, aggregations, domain indices).

### Agriculture-Specific Context
*   **Uyar, H., Karvelas, I., Rizou, S., & Fountas, S. (2024).** Data value creation in agriculture: A review. *Computers and Electronics in Agriculture*, 227, 109602.
    *   **Why read it:** This paper provides a comprehensive overview of the data-to-decision gap in agriculture. It explains *why* data preparation is a critical bottleneck in smart farming and the adoption of AI.

---

## 2. Seminal Research Papers (The "Why" and "How")

These papers provide the theoretical and empirical backing for the techniques you learned in the Jupyter Notebook.

### On Data Cleaning (Missing Values & Outliers)
*   **Little, R. J. A., & Rubin, D. B. (2019).** *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
    *   **Why read it:** The foundational text on missing data mechanisms (MCAR, MAR, MNAR). It explains mathematically why the Mean is a poor choice for skewed agricultural data and why Maximum Likelihood or Multiple Imputation (or Median for simple cases) is better.
*   **Aggarwal, C. C. (2017).** *Outlier Analysis* (2nd ed.). Springer.
    *   **Why read it:** The definitive text on outlier detection. Chapter 2 explains the Interquartile Range (IQR) and Z-score methods mathematically.

### On Feature Engineering (Automated & Manual)
*   **Kanter, J. M., & Veeramachaneni, K. (2015).** Deep Feature Synthesis: Towards Automating Data Science Endeavors. *2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA)*.
    *   **Why read it:** This is the foundational paper that introduced "Deep Feature Synthesis." It explores how we can teach machines to automatically generate interaction and aggregation features (like our N:P Ratio). It is the basis for the `featuretools` Python library.

---

## 3. Online Courses & Video Resources (For Visual Learners)

*   **StatQuest with Josh Starmer (YouTube)**
    *   [Missing Data Imputation](https://www.youtube.com/watch?v=WkTs2jD6CMM): A fantastic, visual explanation of why simple mean imputation is dangerous and how algorithms like K-Nearest Neighbors (KNN) imputation work.
    *   [Outliers and the IQR Method](https://www.youtube.com/watch?v=1e0mJ6jT9c0): A step-by-step numerical walkthrough of the exact IQR method we used in our Python code.
*   **Kaggle Learn: Feature Engineering**
    *   **Link:** [https://www.kaggle.com/learn/feature-engineering](https://www.kaggle.com/learn/feature-engineering)
    *   **Why watch it:** A free, hands-on micro-course. It covers mutual information, creating features, and feature selection techniques.
*   **Coursera (Google Cloud): Feature Engineering**
    *   **Link:** [https://www.coursera.org/learn/feature-engineering](https://www.coursera.org/learn/feature-engineering)
    *   **Why watch it:** An excellent overview of how feature engineering is done in the real world, covering everything from thresholding and binning to handling categorical variables.

---

## 4. Python Documentation & Tools (For the Jupyter Notebook)

*   **Pandas Documentation: Missing Data**
    *   **Link:** [https://pandas.pydata.org/docs/user_guide/missing_data.html](https://pandas.pydata.org/docs/user_guide/missing_data.html)
    *   **Why use it:** The official guide on how to use `df.isna()`, `df.fillna()`, and `df.dropna()` effectively.
*   **Scikit-Learn: Preprocessing Data**
    *   **Link:** [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)
    *   **Why use it:** The official guide for `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, and `OrdinalEncoder`. We will use these extensively in Session 4 when building our ML pipelines.
*   **Featuretools (Automated Feature Engineering)**
    *   **Link:** [https://www.featuretools.com/](https://www.featuretools.com/)
    *   **Why use it:** If you have complex, relational agricultural data (e.g., farm records linked to sensor records), `featuretools` can automatically generate hundreds of features for you, based on the methodology of the Kanter & Veeramachaneni paper above.

