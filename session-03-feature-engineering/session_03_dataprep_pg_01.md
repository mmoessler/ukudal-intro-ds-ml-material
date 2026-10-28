# Session 03 — Page 01: Data Preparation

**Learning outcomes:**  
- Apply median imputation to agricultural sensor data.
- Use the IQR method to cap extreme weather outliers.
- Use EDA to find biological interactions.

## 1. Handling Missing Values (Imputation)
In agricultural datasets, missing values are incredibly common. IoT sensors lose battery, animals chew through cables, or farmers simply forget to log manual records.

If we have missing values in a continuous column like `Rainfall (mm)`, we have two main choices:
- **The Mean:** This is heavily skewed by extreme weather. If rainfall over four days is `[0, 0, 0, 100]`, the mean is `25`. But mathematically, 25mm of rain never actually happened.
- **The Median:** The median of `[0, 0, 0, 100]` is `0`. It represents the "middle" reality of the field. 

**Real-World Scenario:** Imagine you are analyzing soil moisture data across 50 farms. A rainstorm knocks out three sensors. If you use the mean, you artificially inflate the soil moisture for those days. Using the median ensures your imputed values reflect the actual dry conditions of the majority of the farm.

**Best Practice:** Always use the median for skewed agricultural data like rainfall or soil nutrients.

```python
# Median Imputation Python code
df['rainfall'] = df['rainfall'].fillna(df['rainfall'].median())
```

## 2. Handling Outliers (Winsorization)

Extreme weather events (like a 200mm storm) are real, but they can break the scale of a machine learning model. We do not delete them—we cap them (Winsorization).

We use the Interquartile Range (IQR):
1. Find the 25th percentile (Q1) and 75th percentile (Q3).
2. Calculate IQR = Q3 - Q1.
3. Calculate the Upper Bound = Q3 + (1.5 × IQR).
4. Any value above the Upper Bound is replaced by the Upper Bound itself.

**Real-World Scenario:** You are predicting crop yield. A rogue logger error records a rainfall of 5,000mm. If uncapped, the model will warp its entire understanding of rainfall to account for this one anomaly. Capping preserves the fact that a heavy rain event happened, without letting it break the model.

## 3. Exploratory Data Analysis (EDA)

Before modeling, you must visualize the data. EDA is how you generate hypotheses for Feature Engineering.
- **Correlation Matrices:** If two features have a correlation ∣r∣>0.85∣r∣>0.85, you have multicollinearity.

- **Scatter Plots:** Look at Temperature vs Yield. Is the relationship a straight line? If not, you need a non-linear model or engineered features.

- **The Agronomic Insight:** Look at the interactions. Temperature alone might have a weak correlation with crop yield. Humidity alone might be weak. But together? High heat + High humidity = Deadly fungal diseases. This is the cue to engineer a new feature.