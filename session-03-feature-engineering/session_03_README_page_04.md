## About Dataset
### Overview:
The Smart Farming 2024 (SF24) dataset is a comprehensive collection of agricultural data gathered from various farms, designed to enhance crop health monitoring and environmental stress assessment. This dataset is particularly valuable for researchers and developers focused on predictive modeling, machine learning, and data-driven agriculture solutions.

---

### Features:
The dataset contains 2310 records with 23 features, each representing critical environmental and soil parameters, along with specific crop-related metrics. These features include:

1. N (ppm): Nitrogen content in the soil.
2. P (ppm): Phosphorus content in the soil.
3. K (ppm): Potassium content in the soil.
4. Temperature (°C): Ambient temperature recorded at the farm.
5. Humidity (%): Relative humidity at the location.
6. pH: Soil pH level, indicating acidity or alkalinity.
7. Rainfall (mm): Amount of rainfall received.
8. Label: Type of crop associated with the recorded conditions (e.g., rice). _22 distinct crops present in the dataset. Each crop has exactly 100 samples, making it a perfectly balanced dataset for classification tasks._
9. Soil Moisture (%): Percentage of water content in the soil.
10. Soil Type: Type of soil (1 = Sandy, 2 = Loamy, 3 = Clay).
11. Sunlight Exposure (hrs/day): Average daily sunlight exposure.
12. Wind Speed (km/h): Wind speed at the farm.
13. CO2 Concentration (ppm): Carbon dioxide concentration in the air.
14. Organic Matter (%): Percentage of organic material in the soil.
15. Irrigation Frequency (times/week): Frequency of irrigation.
16. Crop Density (plants/m²): Number of plants per square meter.
17. Pest Pressure (index): Level of pest infestation.
18. Fertilizer Usage (kg/ha): Amount of fertilizer applied per hectare.
19. Growth Stage: Current growth stage of the crop (1 = Seedling, 2 = Vegetative, 3 = Flowering).
20. Urban Area Proximity (km): Distance to the nearest urban area.
21. Water Source Type: Type of water source used for irrigation (1 = River, 2 = Groundwater, 3 = Recycled).
22. Frost Risk (index): Risk of frost occurrence.
23. Water Usage Efficiency (L/kg): Efficiency of water usage per kilogram of crop yield.

---

### Derived Features:
In addition to the original attributes, the following 5 features can be derived _(in fact, it is derived in other variant of the dataset)_ to provide deeper insights into the interactions between environmental conditions and crop performance:

1. Temperature-Humidity Index (THI): A composite measure that evaluates potential crop stress due to specific heat and moisture conditions. 
2. Nutrient Balance Ratio (NBR): Reflects the balance between nitrogen, phosphorus, and potassium in the soil.
3. Water Availability Index (WAI): Combines soil moisture and rainfall to assess overall water availability for crops.
4. Photosynthesis Potential (PP): Estimates the potential for photosynthesis based on sunlight exposure, CO2 concentration, and temperature.
5. Soil Fertility Index (SFI): A composite metric that assesses soil fertility based on organic matter content and NPK levels.

---

### Application:
This dataset is suitable for a variety of applications in smart farming, including:

- Crop prediction and classification
- Environmental stress analysis
- Precision agriculture
- Resource management optimization

---

### Usage:
Researchers can use this dataset to develop models that predict crop health and yield, optimize resource usage, and assess the impact of environmental stressors on agricultural productivity. The inclusion of derived features like THI, NBR, WAI, PP, and SFI enhances the dataset's utility for developing advanced machine learning models tailored to the agricultural sector.


