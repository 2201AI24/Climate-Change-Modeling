# 🌍 Climate Change Modeling

## 📌 Project Title

**Climate Change Modeling**

## 📘 Project Overview

This project analyzes historical climate indicators — **CO₂ concentration**, **global temperature**, and **sea level** — then builds interpretable predictive models that translate atmospheric CO₂ into expected temperature anomalies and associated sea level rise. The approach combines exploratory data analysis, feature engineering (including time-lag and cumulative features), polynomial regression models, and time-series-aware validation.

**Domain:** Data Science
**Difficulty:** Advanced
**Tools:** Jupyter Notebook, VS Code, Python (pandas, scikit-learn, NumPy, Matplotlib/Seaborn, XGBoost)

---

## 🗂 Dataset(s) — Description & Source

This project consolidates three authoritative climate datasets:

1. **CO₂ Concentration (Mauna Loa Observatory)**

   * Monthly CO₂ (ppm) since 1958. Includes seasonally adjusted values.

2. **Global Temperature**

   * Global land and land+ocean temperature records (includes uncertainties), with coverage from historical records (1750 onward for long-term datasets used).

3. **Global Sea Level**

   * Annual/global sea level change (mm) relative to 1993–2008 mean (records from ~1880 onward).

> Final merged dataset (used for modeling): **58 yearly observations** (aggregated to yearly averages, 1958–2015) with columns:
> `year`, `SeaLevel_mm`, `CO2_ppm`, `LandTemp_C`, `LandOceanTemp_C`.

---

## 🔎 Problem Statement

* What atmospheric and environmental factors most influence CO₂ and temperature anomalies?
* How does accumulated warming (cumulative temperature anomaly) affect sea level?
* Can we build an interpretable model pipeline to estimate **sea level rise** for a given CO₂ concentration?

The project answers these by building a two-step predictive pipeline:
**CO₂ → Temperature anomaly → Cumulative temperature → Sea level**

---

## 🧹 Data Preparation & Preprocessing

* **Date handling**: converted dates to `datetime`, extracted `year` and `month`.
* **Aggregation**: converted monthly/irregular data to **yearly averages** to align temporal scale across datasets.
* **Merging**: joined CO₂, temperature, and sea level on `year`.
* **Renaming**: standardized column names for clarity (`CO2_ppm`, `SeaLevel_mm`, `LandTemp_C`, `LandOceanTemp_C`).
* **Missing & duplicates**: dataset contains **no missing values** and **no duplicates** after preprocessing.
* **Final shape**: `(58, 5)` (year + 4 climate features).

---

## 📊 Exploratory Data Analysis (EDA) — Key Findings

* **Sea level** rose steadily (approx. **−70 mm → +50 mm** from early baseline to ~2015).
* **CO₂ concentration** increased from ~**315 ppm → >400 ppm** (clear exponential-like rise).
* **Land temperature** increased from ~**8.5°C → 9.8°C**.
* **Land+Ocean temperature** increased from ~~15.2°C → 16.0°C~~.
* **Overall**: CO₂, temperature, and sea level show strong, positive, and consistent trends indicating interconnection between greenhouse gases, warming, and sea-level rise.
  
<img width="2820" height="1599" alt="image" src="https://github.com/user-attachments/assets/7acf81cb-9bb0-40bc-a1ed-ec9a6b445896" />

---

## 🔗 Correlation Analysis (summary)

A high-level correlation matrix (selected values observed):

```
                 SeaLevel_mm   CO2_ppm   LandTemp_C  LandOceanTemp_C
SeaLevel_mm         1.0000     0.9863     0.8853         0.9003
CO2_ppm             0.9863     1.0000     0.9028         0.9171
LandTemp_C          0.8853     0.9028     1.0000         0.9794
LandOceanTemp_C     0.9003     0.9171     0.9794         1.0000
```

**Interpretation:** CO₂ concentrations are *very strongly* correlated with both temperatures and sea level — supporting a physically consistent chain: **higher CO₂ → higher temperature → higher sea level**.

<img width="1613" height="1474" alt="image" src="https://github.com/user-attachments/assets/5477f216-86e4-4112-998b-d050535ccffe" />

---

## 🛠 Feature Engineering

* **Temp_lag5**: temperature 5 years previous (to capture delayed effects).
* **Temp_cumsum**: cumulative temperature anomaly (long-term accumulated warming).
* Created polynomial features (degree 2) where non-linearity was expected.

Finding: **Cumulative temperature (`Temp_cumsum`) is more predictive of sea-level rise** than short-term lagged temperature.

---

## 🧮 Models & Methodology

### Models Tested

* Linear Regression
* Polynomial Regression (Quadratic / degree 2) — **primary model** for non-linear relationships
* Ridge, Lasso, ElasticNet (regularized linear models)
* XGBoost Regressor
* Time-series-aware validation (TimeSeriesSplit, 5 splits) for model robustness

### Modeling Strategy (two-step pipeline)

1. **Model A (CO₂ → Temperature anomaly)**

   * Polynomial regression (degree 2) fitted to predict temperature anomaly from `CO2_ppm`.
   * Rationale: relationship is non-linear and accelerates at higher CO₂.

2. **Update cumulative temperature**

   * Add predicted temperature anomaly to the historical cumulative anomaly to simulate accumulated warming over time.

3. **Model B (Temp_cumsum → SeaLevel_mm)**

   * Quadratic regression mapping cumulative temperature anomaly to sea level (mm).

This chained approach allows translating a CO₂ input into a plausible sea-level projection.

---

## 📈 Model Evaluation (summary)

* **Quadratic / Polynomial Regression (Model A & B)**: achieved **high R² and Adjusted R²** (models explain a large portion of variability — specific notebook cells contain exact numeric values and residual diagnostics).
* **Regularized linear models (ElasticNet)**: example performance reported with **R² ≈ 0.87** and **Adjusted R² ≈ 0.85** for certain tasks — indicates strong predictive power with regularization.
* **Time-series CV** (5 splits) was used to prevent leakage and to obtain realistic generalization performance.

> See the notebook for detailed MAE, MSE, R², adjusted R² values and residual plots.
<img width="1213" height="863" alt="image" src="https://github.com/user-attachments/assets/54df62c6-b21e-42d9-863d-a02ac7c9ecc0" />
<img width="1104" height="742" alt="image" src="https://github.com/user-attachments/assets/c5c64ac3-27af-4e32-8a3c-0629ccdddd13" />
<img width="1300" height="1037" alt="image" src="https://github.com/user-attachments/assets/a4f161ac-ee72-4097-b7ba-9b6799285c67" />
<img width="1363" height="1063" alt="image" src="https://github.com/user-attachments/assets/dd13a6d5-41a8-4c20-befb-0ef69efcefe1" />
<img width="1388" height="987" alt="image" src="https://github.com/user-attachments/assets/92946979-1b94-4883-9431-2076e3b53dc2" />

---

## 🔁 Predictive Function — Example

A convenience function implemented in the notebook: `predict_sea_level_from_co2(co2_ppm, modelA, polyA, modelB, polyB, df)`
**Workflow (pseudo-code):**

```python
# Step 1: Predict temperature anomaly from CO2
temp_pred = modelA.predict(polyA.transform([[co2_ppm]]))

# Step 2: Update cumulative temperature anomaly
updated_cumsum = df['Temp_cumsum'].iloc[-1] + temp_pred

# Step 3: Predict sea level from updated cumulative temperature
sea_level_pred = modelB.predict(polyB.transform([[updated_cumsum]]))

return {
    'co2_ppm': co2_ppm,
    'pred_temp_anomaly': float(temp_pred),
    'updated_temp_cumsum': float(updated_cumsum),
    'pred_sea_level_mm': float(sea_level_pred)
}

# Example:
result = predict_sea_level_from_co2(400, modelA, polyA, modelB, polyB, df)
print(result)
```

This function encapsulates the CO₂ → Temperature → Cumulative Temp → Sea Level pipeline and returns numerical estimates.

---

## 📂 Files (suggested repository structure)

```
Climate-Change-Modeling/
├─ data/
│  ├─ archive.csv.csv
│  ├─ GlobalLandTemperaturesByCity.csv
│  ├─ TemperaturesByCountry.csv
│  ├─ TemperaturesByMajorCity.csv
|  ├─ GlobalTemperatures.csv
│  └─ Global_sea_level_rise.csv
├─ notebooks/
│  └─ Climate_Change_Modeling.ipynb     # primary notebook with EDA, modeling & plots    
├─ requirements.txt
└─ README.md
```

---

## ⚙️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Climate-Change-Modeling.git
cd Climate-Change-Modeling
```

2. Create environment & install dependencies:

```bash
python -m venv venv
source venv/bin/activate        # on Linux / macOS
# venv\Scripts\activate         # on Windows
pip install -r requirements.txt
```

3. Start Jupyter and open the main notebook:

```bash
jupyter notebook notebooks/Climate_Change_Modeling.ipynb
```
---

## 🔍 Key Insights & Conclusions

* There is a **very strong empirical linkage** between CO₂ concentration, global temperature, and sea level rise (correlations often > 0.9).
* **Non-linear (quadratic) models** capture accelerating trends better than strictly linear fits.
* **Cumulative warming** (long-term accumulation of temperature anomalies) explains sea level rise more reliably than short-term fluctuations.
* The two-step predictive approach provides a transparent, interpretable chain from CO₂ to sea-level projections — valuable for scenario analysis.

---

## 🔮 Future Work & Extensions

* Add **LSTM / advanced time-series models** for long-range forecasting.
* Use **uncertainty quantification** (prediction intervals; Bayesian approaches) for climate projections.
* Expand dataset range (update beyond 2015) and integrate additional forcings (e.g., solar, aerosol indices).
* Build an interactive **Streamlit/Flask** dashboard for scenario simulation and visualization.
* Apply **SHAP** or other interpretability tools for deeper local explanations.

---

## 👨‍💻 Author

**M.Umesh Chandra**
📧 *[[metlaumeshchaandra2005.email@example.com](mailto:metlaumeshchandra2005.email@example.com)]*

---
