# Wind-Turbine-Analysis
🧪 Task 1 — Exploratory Data Analysis

Time-series trend plots for all four SCADA variables

Power curve visualization (Wind Speed vs Active Power)

Detection of missing values and abnormal readings (negative power, extreme wind speed, rolling z-score outliers)

🤖 Task 2 — Time-Series Forecasting (Supervised Learning)

Conversion of SCADA data into a lag-based windowed time-series format

Multi-output RandomForestRegressor to predict all variables at once:

LV ActivePower

Wind Speed

Theoretical Power Curve

Wind Direction

Train/test split using the latest 7 days

Evaluation using:

MAE

RMSE

MAPE

Predicted vs Actual visualization for each variable

The pipeline is optimized for speed using reduced lags and a multi-output model.

⚠️ Task 3 — Unsupervised Anomaly Detection

Underperformance detection using residual analysis:

Anomaly = Theoretical Power − Actual Power is significantly positive

Rolling mean/standard deviation

Residual z-score

Flags timestamps with abnormal turbine underperformance

🧠 Task 4 — AI Turbine Performance Score Generator

An AI-based performance scoring module that:

Computes Actual vs Theoretical power ratio

Converts it to a 0–100 performance score

Categorizes turbine status as:

Good (≥ 85)

Moderate (60–84)

Poor (< 60)

Generates automatic maintenance suggestions based on the score
