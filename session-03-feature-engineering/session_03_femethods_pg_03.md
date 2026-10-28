
# Session 03 — Page 03: Feature Engineering Methods


## Learning outcomes

By the end of this session, you should be able to:

- Explain why feature engineering is needed in agrifood ML.
- Create interaction, ratio, aggregation, and domain-index features.
- Encode nominal and ordinal categorical variables appropriately.
- Recognise multicollinearity and apply correlation/VIF checks carefully.
- Avoid data leakage when engineering features.

## Feature engineering in context

Feature engineering converts raw agrifood data—weather, soil, remote sensing, management, and pest observations—into variables that better represent the biological or management process we want to model.

A good feature should:

- Have a clear agrifood interpretation.
- Be available at prediction time.
- Improve model performance on validation data.
- Not merely duplicate information already captured by other features.

**Important:** Always perform EDA before and during feature engineering. EDA helps identify skewed variables, missing values, outliers, correlations, seasonal patterns, and plausible feature relationships.

## 1. Interaction features

An interaction feature captures a **joint effect**: the effect of one variable depends on the value of another.

**General form:**

$$x_{i\text{,interaction}} = x_1 \times x_2$$

### Example: Temperature–Humidity Index (THI)

THI combines air temperature and relative humidity into a single heat-stress indicator, especially useful for livestock.

$$THI = (1.8T + 32) - (0.55 - 0.0055RH)(1.8T - 26.8)$$

Where:

- $T$ = dry-bulb air temperature in °C
- $RH$ = relative humidity in %

**Agrifood rationale:** High temperature alone does not fully describe heat stress. At high humidity, evaporative cooling is reduced, so animals and crops may experience greater stress. THI is therefore more informative than temperature or humidity separately.

**Practical note:** THI thresholds depend on the livestock species, breed, production system, and local adaptation. Treat fixed thresholds such as “THI > 79 means severe stress” as context-specific guidance, not universal rules.

### Other useful interactions

| Features | Interaction | Possible interpretation |
|---|---|---|
| Rainfall and soil type | `rainfall × soil_type` | Rainfall effectiveness may differ between sandy and clay soils |
| Temperature and growth stage | `temperature × growth_stage` | Heat stress may be more damaging during flowering |
| Fertiliser and soil organic carbon | `fertiliser × soil_organic_carbon` | Fertiliser response may depend on existing soil fertility |
| Pesticide use and pest count | `pesticide × pest_count` | Treatment effect may depend on infestation pressure |

**Caution:** Do not create every possible pairwise interaction. Use domain knowledge and validation performance to select meaningful interactions.

## 2. Ratio features

A ratio feature divides one variable by another to express a **per-unit** or **relative** measure.

\[
x_{\text{ratio}} = \frac{x_1}{x_2}
\]

### Example: Nutrient balance

A simple nitrogen-to-phosphorus ratio is:

\[
\text{N:P ratio} = \frac{N}{P}
\]

**Agrifood rationale:** Absolute nutrient amounts can be misleading. A field may have adequate nitrogen but still show poor nutrient balance if phosphorus is limiting. A ratio expresses the balance between nutrients.

**Important correction:** Avoid adding an arbitrary constant such as `P + 1` unless the units require it or you are explicitly preventing division by zero. If \(P\) can be zero, use a documented safeguard, for example:

```python
N_to_P = N / P.clip(lower=epsilon)
```

where `epsilon` is a small, justified positive value.

### Other useful ratio features

| Ratio | Formula | Meaning |
|---|---|---|
| Yield per hectare | `yield_kg / area_ha` | Productivity independent of field size |
| Fertiliser intensity | `fertiliser_kg / area_ha` | Nutrient input per unit land |
| Water-use efficiency | `yield_kg / water_m3` | Output per unit of water |
| Pest density | `pest_count / plants_sampled` | Infestation level per plant |
| Feed conversion ratio | `feed_kg / weight_gain_kg` | Livestock feed efficiency |

**Precautions:**

- Check for zero or near-zero denominators.
- Keep units consistent before dividing.
- Inspect extreme ratios; cap or transform them only with justification.
- Do not use future information—for example, do not calculate yield per hectare using final harvest data when predicting yield before harvest.

## 3. Aggregation features

Aggregation features summarise several related variables into one meaningful metric.

### Example: Total nutrient availability

\[
\text{Total NPK} = N + P + K
\]

**Agrifood rationale:** A total nutrient score can provide a compact summary of overall nutrient availability.

**Caution:** Only add variables directly when they share compatible units and agronomic meaning. N, P, and K are distinct nutrients with different roles and are often measured in different units or forms. A simple sum may hide important nutrient imbalances.

A more informative approach may be to retain:

- `N`, `P`, and `K` separately
- `N:P`, `N:K`, and `P:K` ratios
- A domain-validated composite fertility index, if available

### Other aggregation examples

| Aggregation | Formula | Use |
|---|---|---|
| Cumulative rainfall | `sum(daily_rainfall)` | Total water received over a growth period |
| Cumulative GDD | `sum(daily_GDD)` | Accumulated heat for crop development |
| Mean NDVI | `mean(NDVI over period)` | Average canopy vigour |
| Maximum pest count | `max(pest_count)` | Peak infestation pressure |

**Important:** Aggregations over time must respect the prediction date. For example, when predicting yield mid-season, use only weather and remote-sensing data available up to that date.

## 4. Domain indices

Domain indices use established scientific relationships to summarise complex agrifood processes.

### Example: Growing Degree Days (GDD)

Daily GDD is commonly calculated as:

\[
GDD_{\text{daily}} =
\max\left(
\frac{T_{\max} + T_{\min}}{2} - T_{\text{base}},
0
\right)
\]

Accumulated GDD is:

\[
GDD_{\text{cumulative}} =
\sum_{d=1}^{n} GDD_{\text{daily},d}
\]

Where:

- \(T_{\max}\) = daily maximum temperature
- \(T_{\min}\) = daily minimum temperature
- \(T_{\text{base}}\) = crop- or pest-specific base temperature
- Negative daily values are set to zero

**Agrifood rationale:** Crops and pests develop in response to accumulated heat, not simply calendar days. Cumulative GDD is widely used to estimate phenological stages, harvest timing, and pest development.

**Important:** \(T_{\text{base}}\) is crop-specific. Some crop models also apply an upper temperature threshold, beyond which additional heat does not increase development.

### Other domain indices

| Index | Inputs | Agrifood use |
|---|---|---|
| NDVI | Red and near-infrared reflectance | Vegetation vigour and canopy cover |
| EVI | Red, near-infrared, and blue reflectance | Vegetation vigour with atmospheric and canopy corrections |
| SAVI | Red and near-infrared reflectance | Vegetation in sparse-canopy or bare-soil conditions |
| THI | Temperature and humidity | Livestock heat stress |
| Cumulative GDD | Daily temperature | Crop and pest development |

## 5. Categorical encoding

Machine-learning models require numerical inputs, but the encoding method must match the type of category.

### Nominal encoding: one-hot encoding

Use one-hot encoding for categories with **no natural order**.

Example: `Soil Type`

| Soil Type | Soil_Sandy | Soil_Loamy | Soil_Clay |
|---|---:|---:|---:|
| Sandy | 1 | 0 | 0 |
| Loamy | 0 | 1 | 0 |
| Clay | 0 | 0 | 1 |

**When to drop one category:** Dropping one level with `drop_first=True` is useful for **linear models with an intercept**, because it avoids perfect multicollinearity among the dummy variables.

However, do not automatically drop a category for every model:

- Tree-based models, such as random forests and gradient boosting, usually do not require `drop_first=True`.
- Regularised linear models can sometimes handle full one-hot encoding, depending on the penalty and implementation.
- Dropping a category changes the interpretation of coefficients: the omitted category becomes the reference group.

### Ordinal encoding: ordered categories

Use ordinal encoding when categories have a meaningful order.

Example: `Growth Stage`

```text
Seedling = 1
Vegetative = 2
Flowering = 3
```

This preserves the agronomic progression from early to later development.

**Caution:** Ordinal encoding implies equal spacing between categories unless you provide additional information. If the agronomic difference between stages is not uniform, consider one-hot encoding instead, especially for linear models.

### High-cardinality categories

For variables with many categories—such as farm ID, district, or crop variety—consider:

- Target encoding, fitted only within training folds to avoid leakage.
- Frequency encoding.
- Grouping rare categories into an “Other” category.
- Hierarchical encoding, for example province → district → farm.

## 6. Multicollinearity

Multicollinearity occurs when predictor variables contain overlapping information. It can make coefficient estimates unstable in linear models and complicate interpretation.

### Examples in agrifood data

- NDVI, EVI, and SAVI may be strongly correlated.
- Rainfall, humidity, and soil moisture may move together.
- Cumulative GDD and days after planting may be highly related.
- Total NPK may correlate with its component nutrient variables.

### Correlation screening

Calculate pairwise correlations:

\[
r_{xy} =
\frac{\operatorname{cov}(x,y)}
{\sigma_x\sigma_y}
\]

A common screening rule is:

- \(|r| > 0.85\): investigate redundancy.
- \(|r| > 0.95\): strong redundancy; consider removing or combining one feature.

**Important:** Correlation only detects **linear, pairwise** relationships. Two features can have low correlation but still be redundant through a nonlinear relationship.

### Variance Inflation Factor

VIF measures how much the variance of a regression coefficient is inflated by collinearity with other predictors.

\[
VIF_j =
\frac{1}{1 - R_j^2}
\]

where \(R_j^2\) comes from regressing feature \(j\) on all other features.

Common interpretation:

| VIF | Interpretation |
|---:|---|
| 1 | No collinearity with other predictors |
| 1–5 | Usually acceptable |
| 5–10 | Moderate to high collinearity; investigate |
| > 10 | Serious collinearity; consider removal, combination, or regularisation |

**Important:** VIF thresholds are guidelines, not absolute rules. In agrifood data, correlated environmental variables may both be biologically meaningful. Consider:

- Removing one feature only if it adds little predictive value.
- Combining related variables into a domain index.
- Using PCA or another dimensionality-reduction method.
- Using regularisation, such as ridge regression, when prediction—not coefficient interpretation—is the main goal.

## 7. Avoiding data leakage

Feature engineering can accidentally leak information from the validation or test set into training.

Follow these rules:

- Split data into training and validation/test sets **before** fitting transformers that learn from data.
- Fit scalers, imputers, encoders, target encoders, and outlier thresholds on the **training set only**.
- Apply the fitted transformation to validation and test data.
- For time-series agrifood data, ensure that features at prediction time use only past or current information.
- Do not use the target variable to create features unless using a leakage-safe, cross-validated target-encoding procedure.

## 8. Practical workflow

```text
Define prediction target and prediction time
→ Clean data
→ Perform EDA
→ Create candidate features
→ Check units, missing values, and impossible values
→ Split data before fitting transformers
→ Fit preprocessing on training data only
→ Train baseline model
→ Evaluate on validation data
→ Add or remove features iteratively
→ Check feature importance and prediction errors
→ Confirm that improvements hold on unseen data
```

## Key takeaway

Feature engineering is not about adding as many variables as possible. It is about representing agrifood knowledge—such as heat stress, nutrient balance, crop development, and water-use efficiency—in a form the model can use reliably and without leakage.
```

## Main improvements made

- Corrected and clarified the **THI formula**, including the use of dry-bulb temperature and relative humidity.
- Corrected **GDD**: it is normally calculated from daily maximum and minimum temperature, then accumulated over time; your original formula described only one day’s contribution.
- Flagged the issue with `N / (P + 1)`: adding 1 is arbitrary unless justified as a division-by-zero safeguard.
- Added caution that summing N, P, and K may not be agronomically meaningful because they are distinct nutrients.
- Made the **dummy-variable trap** explanation more nuanced: dropping a category is model-dependent, not universally required.
- Replaced the rigid “drop if \(|r| > 0.85\)” rule with a more defensible screening-and-investigation approach.
- Added VIF interpretation, leakage prevention, high-cardinality encoding, and an iterative feature-engineering workflow.
- Strengthened the agrifood relevance with examples involving yield, irrigation, pests, livestock, soil, and remote sensing. [cpc.ncep.noaa](https://www.cpc.ncep.noaa.gov/products/tools/cfsr/GDD.html)