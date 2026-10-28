# Session 03 — Page 02: The 4-Step Framework


**Learning outcomes:** 

- Explain why feature engineering requires domain knowledge.
- Apply a structured process to create new features from raw agrifood data.
- Distinguish between useful, redundant, and potentially misleading features.
- Validate engineered features using an appropriate machine-learning workflow.

## 1. What is Feature Engineering?
Feature engineering is the process of using **domain knowledge** to create new variables (features) from raw data to improve the performance of machine learning models. 

**The Core Philosophy:** "Machines find the math. You find the meaning." Models cannot invent biological interactions on their own. They just see columns of numbers. You are the bridge that translates agronomic theory into mathematical features.

## 2. The 4-Step Process
Do not guess. Follow this structured process:

### Step 1: Domain Framing (The "Why")
Start with an agronomic or food-system question:

> Which biological, physical, or management process might connect these variables to the outcome?

Ask yourself:

- What does each raw variable represent?
- Could two or more variables have a joint effect?
- Is the proposed feature meaningful in the real world?
- Would the feature be available at prediction time

*Example:* A raw temperature of 35°C is dangerous to maize **only if** humidity is high. High humidity prevents the plant from cooling itself via transpiration.

### Step 2: Transformation (The "How")
Convert the domain idea into a mathematical operation.

Common transformations include:

| Transformation | Formula example | Agrifood interpretation |
|---|---|---|
| Ratio | `N / P` | Nutrient balance |
| Sum | `N + P + K` | Total macronutrient availability |
| Interaction | `temperature * humidity` | Joint heat and humidity effect |
| Difference | `soil_moisture - rainfall` | Relative moisture condition |
| Rate of change | `(NDVI_today - NDVI_yesterday) / days` | Crop growth or decline |
| Capping | Replace extreme values with percentile limits | Reduce influence of outliers |

**Important:** Only create a feature if its calculation has a clear interpretation. Avoid combining variables simply because it is mathematically possible.


Ask: What is the mathematical formula?

*Example:* We combine Temperature (T) and Humidity (H) into the Temperature-Humidity Index (THI). This converts raw sensor readings into a standardized, interpretable index.

### Step 3: Selection (The "Filter")
Ask: Did this create redundancy (multicollinearity)?

A new feature is useful only if it adds information that the existing features do not already provide.

Check for:

- **High correlation** with existing numerical features.
- **Near-duplicate information**, such as `Total_NPK` and `N + P + K`.
- **Target leakage**, where a feature uses information that would not be available at prediction time.
- **Domain relevance**, especially when correlated variables represent different agronomic processes.

For example, `Total_NPK` may be correlated with `N`, `P`, and `K`. That does not automatically mean it should be removed. It may still capture an overall fertility effect that individual nutrients do not.

A practical rule is:

> Identify highly correlated features, inspect them, and decide which one to retain using domain knowledge and validation performance—not correlation alone.




*Example:* Now that we have THI, do we still need raw Temperature? If the correlation is > 0.85, we drop raw Temperature to prevent the model from getting confused.

### Step 4: Validation (The "Test")

Feature engineering is a hypothesis. Validation tests whether the hypothesis improves the model.

A leakage-safe workflow is:

```text
Split data into training and test sets
→ Explore and engineer features using the training set
→ Fit preprocessing on the training set
→ Apply the same transformations to the test set
→ Compare model performance with and without the new feature
```

Use the metric appropriate to your task:

| Task type | Suitable metrics |
|---|---|
| Classification, such as crop recommendation | Accuracy, precision, recall, F1-score, confusion matrix |
| Regression, such as yield prediction | RMSE, MAE, R² |
| Imbalanced classification | F1-score, precision, recall, ROC-AUC |


**Example:**  
Train two versions of a crop-classification model:

- **Baseline model:** original features only.
- **Enhanced model:** original features plus `temp_humidity_interaction`.

If the enhanced model improves cross-validated F1-score without overfitting, the feature may be useful. If performance does not improve, revise or remove the feature.

Ask: Did it actually work?

*Example:* Train a Random Forest model with and without the new feature. 
- Did the Root Mean Squared Error (RMSE) drop? 
- Did Accuracy improve on the test set? 

## 3. Common mistakes to avoid

- **Creating features without a domain rationale.** A mathematically valid feature may still be meaningless.
- **Using future information.** Do not use post-harvest yield, final pest counts, or other unavailable measurements to predict an earlier outcome.
- **Preprocessing before splitting.** Imputation, outlier capping, scaling, and correlation-based removal should be fitted on training data only.
- **Assuming correlation means causation.** A strong relationship between rainfall and yield does not prove rainfall causes the observed difference.
- **Keeping every engineered feature.** Too many features can increase overfitting and reduce interpretability.
- **Ignoring class imbalance.** In crop or pest classification, accuracy alone can be misleading.

## 4. Quick checklist

Before adding a feature, ask:

1. What agrifood process does it represent?
2. Which raw variables does it use?
3. Is the formula meaningful and correctly calculated?
4. Is it available at prediction time?
5. Does it add information beyond existing features?
6. Does it improve validation performance?
7. Can its effect be explained to an agronomist or extension officer?

## Key message

Feature engineering is not about creating as many columns as possible. It is about creating **meaningful, valid, and testable representations** of agrifood systems.

> A good feature is biologically plausible, computationally correct, available at prediction time, and demonstrably useful in validation.