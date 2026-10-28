# Session 03 — Page 02: The 4-Step Framework


**Learning outcomes:**  
- Execute a rigorous process to create and validate new features.

## 1. What is Feature Engineering?
Feature engineering is the process of using **domain knowledge** to create new variables (features) from raw data to improve the performance of machine learning models. 

**The Core Philosophy:** "Machines find the math. You find the meaning." Models cannot invent biological interactions on their own. They just see columns of numbers. You are the bridge that translates agronomic theory into mathematical features.

## 2. The 4-Step Process
Do not guess. Follow this structured process:

### Step 1: Domain Framing (The "Why")
Ask: What biological or agronomic theory explains this? 

*Example:* A raw temperature of 35°C is dangerous to maize **only if** humidity is high. High humidity prevents the plant from cooling itself via transpiration.

### Step 2: Transformation (The "How")
Ask: What is the mathematical formula?

*Example:* We combine Temperature (T) and Humidity (H) into the Temperature-Humidity Index (THI). This converts raw sensor readings into a standardized, interpretable index.

### Step 3: Selection (The "Filter")
Ask: Did this create redundancy (multicollinearity)?

*Example:* Now that we have THI, do we still need raw Temperature? If the correlation is > 0.85, we drop raw Temperature to prevent the model from getting confused.

### Step 4: Validation (The "Test")
Ask: Did it actually work?

*Example:* Train a Random Forest model with and without the new feature. 
- Did the Root Mean Squared Error (RMSE) drop? 
- Did Accuracy improve on the test set? 

Feature engineering is a hypothesis. Validation is the experiment. If the model doesn't improve, the feature wasn't useful. This is why iteration is important in ML pipeline.