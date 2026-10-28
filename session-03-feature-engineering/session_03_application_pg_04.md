# Session 03 — Page 04: Application Task


## Task Overview
Using the `sf24data.csv` dataset, you are to act as both an agronomist and a data scientist. You will clean the dataset, perform EDA, and engineer the features we discussed in the office hour session.

## Deliverables
Implement in a Jupyter Notebook (or Google Colab) with the following sections:

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
   - Identify any features with a correlation $|r| > 0.85$. Drop them and print the final shape of your dataset.


## References & Further Reading


### Core Textbooks (Data Preparation & Feature Engineering)


*   **Kuhn, M., & Johnson, K. (2019).** *Feature Engineering and Selection: A Practical Approach for Predictive Models*. Chapman & Hall/CRC.
   
*   **Zheng, A., & Casari, A. (2018).** *Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists*. O'Reilly Media.
    
*   **Duboue, P. (2020).** *The Art of Feature Engineering: Essentials for Machine Learning*. Cambridge University Press.
   

### Agriculture-Specific Context
*   **Uyar, H., Karvelas, I., Rizou, S., & Fountas, S. (2024).** Data value creation in agriculture: A review. *Computers and Electronics in Agriculture*, 227, 109602.
    

### On Data Cleaning (Missing Values & Outliers)
*   **Little, R. J. A., & Rubin, D. B. (2019).** *Statistical Analysis with Missing Data* (3rd ed.). Wiley.

*   **Aggarwal, C. C. (2017).** *Outlier Analysis* (2nd ed.). Springer.
  
*   **Kaggle Learn: Feature Engineering** [https://www.kaggle.com/learn/feature-engineering](https://www.kaggle.com/learn/feature-engineering)