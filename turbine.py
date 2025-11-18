# turbine_fixed_multioutput.py
"""
Wind Turbine SCADA analysis pipeline - robust parsing and end-to-end with
faster multi-output RandomForest training (single call for all targets).

Save as turbine_fixed_multioutput.py and run:
    python turbine_fixed_multioutput.py

Adjust CSV_PATH at top if your file name differs.
"""
import os
import re
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CSV_PATH = r"C:\Users\dhara\OneDrive\Documents\ML Project\novintix\T1.csv"  
DATECOL_CANONICAL = "date"
RANDOM_STATE = 42
N_LAGS = 12                 
TEST_SIZE_DAYS = 7           
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def normalize_colname(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace('°', 'deg')
    s = re.sub(r'[\(\)\-/,]', ' ', s)    
    s = re.sub(r'[_\s]+', ' ', s)        
    return s

def map_columns(df: pd.DataFrame):
    print("Raw columns found in CSV:")
    for c in df.columns:
        print(" -", repr(c))

    norm_to_orig = {normalize_colname(c): c for c in df.columns}

    expected = {
        'date': ['date time', 'date/time', 'datetime', 'timestamp', 'time', 'date'],
        'active_power': ['lv activepower kw', 'lv activepower (kw)', 'active power', 'lv active power (kw)', 'active_power', 'power_kw', 'active power (kw)'],
        'wind_speed': ['wind speed m s', 'wind speed (m/s)', 'wind_speed', 'windspeed', 'wind speed'],
        'theoretical_power': ['theoretical power curve kwh', 'theoretical power curve (kwh)', 'theoretical power', 'theoretical_power', 'theoretical_power_curve', 'theoretical power (kwh)'],
        'wind_direction': ['wind direction deg', 'wind direction (°)', 'wind_direction', 'wind direction', 'wind dir']
    }

    rename_map = {}
    for canonical, cand_list in expected.items():
        found = None
        for cand in cand_list:
            if cand in norm_to_orig:
                found = norm_to_orig[cand]
                break
        if not found:
            for norm, orig in norm_to_orig.items():
                for cand in cand_list:
                    if cand.replace(' ', '') in norm.replace(' ', ''):
                        found = orig
                        break
                if found:
                    break
        if found:
            rename_map[found] = canonical

    print("\nColumn mapping to canonical names (applied):")
    for orig, can in rename_map.items():
        print(f" - {orig}  -->  {can}")

    df = df.rename(columns=rename_map)
    return df
def parse_datetime_column(df: pd.DataFrame):
    if DATECOL_CANONICAL in df.columns:
        raw_col = DATECOL_CANONICAL
    else:
        possible = [c for c in df.columns if 'date' in normalize_colname(c) or 'time' in normalize_colname(c)]
        raw_col = possible[0] if possible else None

    if raw_col is None:
        raise ValueError("No date/time column found in CSV. Please check headers.")

    s = df[raw_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    parsed = pd.to_datetime(s, format="%d %m %Y %H:%M", errors="coerce")  
    mask_na = parsed.isna()
    if mask_na.any():
        parsed_other = pd.to_datetime(s[mask_na], dayfirst=True, infer_datetime_format=True, errors="coerce")
        parsed.loc[mask_na] = parsed_other
    mask_na = parsed.isna()
    if mask_na.any():
        parsed_inferred = pd.to_datetime(s[mask_na], infer_datetime_format=True, errors="coerce")
        parsed.loc[mask_na] = parsed_inferred

    if parsed.isna().any():
        print("Warning: some Date/Time rows could not be parsed. Examples:")
        print(s[parsed.isna()].head(10).to_list())

    df = df.copy()
    df[DATECOL_CANONICAL] = parsed
    df = df.dropna(subset=[DATECOL_CANONICAL]).set_index(DATECOL_CANONICAL).sort_index()
    return df

def plot_timeseries(df, title_prefix=''):
    desired = ['active_power', 'wind_speed', 'theoretical_power', 'wind_direction']
    present = [c for c in desired if c in df.columns]
    missing = [c for c in desired if c not in df.columns]
    if missing:
        print("Warning: missing expected columns (won't be plotted):", missing)

    n = len(present)
    if n == 0:
        print("No expected numeric columns found to plot.")
        return
    plt.figure(figsize=(14, 3 * n))
    for i, c in enumerate(present, 1):
        plt.subplot(n, 1, i)
        plt.plot(df.index, df[c], marker='.', linestyle='-', markersize=2)
        plt.ylabel(c)
        plt.grid(True)
        if i == 1:
            plt.title(title_prefix + 'Time-series plots')
    plt.tight_layout()
    plt.show()

def scatter_power_curve(df, sample=10000):
    if 'active_power' not in df.columns or 'wind_speed' not in df.columns:
        print("Can't draw scatter: active_power or wind_speed missing.")
        return
    s = df[['active_power', 'wind_speed']].dropna()
    if len(s) > sample:
        s = s.sample(sample, random_state=RANDOM_STATE)
    plt.figure(figsize=(8,6))
    plt.scatter(s['wind_speed'], s['active_power'], alpha=0.3, s=6)
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('LV ActivePower (kW)')
    plt.title('Power curve: Wind Speed vs Active Power')
    plt.grid(True)
    plt.show()

def detect_missing_and_abnormal(df):
    print("Missing values per column:\n", df.isna().sum())
    abnormalities = {}
    if 'active_power' in df.columns:
        abnormalities['negative_power'] = (df['active_power'] < 0).sum()
    if 'wind_speed' in df.columns:
        abnormalities['extreme_wind_speed'] = (df['wind_speed'] > 80).sum()
    z_flags = pd.DataFrame(index=df.index)
    for c in ['active_power','wind_speed','theoretical_power','wind_direction']:
        if c in df.columns:
            rolling_mean = df[c].rolling(window=48, min_periods=1, center=True).mean()
            rolling_std = df[c].rolling(window=48, min_periods=1, center=True).std().replace(0, np.nan)
            z = (df[c] - rolling_mean) / rolling_std
            z_flags[c + '_z_outlier'] = z.abs() > 4
    print("Abnormal counts (examples):", abnormalities)
    if not z_flags.empty:
        print("Rolling-z outlier counts:\n", z_flags.sum())
    return abnormalities, z_flags

def create_lag_features(df, n_lags=N_LAGS, cols=None):
    if cols is None:
        cols = [c for c in ['active_power','wind_speed','theoretical_power','wind_direction'] if c in df.columns]
    X = pd.DataFrame(index=df.index)
    for c in cols:
        for lag in range(1, n_lags+1):
            X[f'{c}_lag_{lag}'] = df[c].shift(lag)
    y = df[[c for c in cols]].copy()
    return X, y

def train_test_split_time(X, y, test_size_days=TEST_SIZE_DAYS):
    if X.empty:
        raise ValueError("Feature matrix X is empty (likely not enough data after lags).")
    last_date = X.index.max()
    test_start = last_date - pd.Timedelta(days=test_size_days)
    train_mask = X.index < test_start
    test_mask = X.index >= test_start
    X_train = X.loc[train_mask].dropna()
    X_test = X.loc[test_mask].dropna()
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]
    return X_train, X_test, y_train, y_test

def evaluate_preds(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE%': mape}

def plot_pred_vs_actual(idx, actual, pred, title=None, max_points=500):
    if len(idx) == 0:
        return
    n = min(len(idx), max_points)
    plt.figure(figsize=(12,4))
    plt.plot(idx[:n], actual[:n], label='actual')
    plt.plot(idx[:n], pred[:n], label='predicted', alpha=0.8)
    plt.legend()
    plt.title(title or 'Predicted vs Actual')
    plt.grid(True)
    plt.show()

def detect_underperformance(df, z_thresh=3.0):
    if 'theoretical_power' not in df.columns or 'active_power' not in df.columns:
        print("Underperformance detection requires both theoretical_power and active_power columns.")
        return pd.DataFrame(index=df.index)
    df2 = df.copy()
    df2['residual'] = df2['theoretical_power'] - df2['active_power']
    mu = df2['residual'].rolling(window=48, min_periods=1, center=True).mean()
    sigma = df2['residual'].rolling(window=48, min_periods=1, center=True).std().replace(0, np.nan)
    df2['res_z'] = (df2['residual'] - mu) / sigma
    df2['underperf_anomaly'] = df2['res_z'] > z_thresh
    print('Underperformance anomalies detected:', int(df2['underperf_anomaly'].sum()))
    return df2[['residual','res_z','underperf_anomaly']]

def performance_score_and_advice(df):
    if 'theoretical_power' not in df.columns or 'active_power' not in df.columns:
        print("Performance scoring requires theoretical_power and active_power.")
        return pd.DataFrame(index=df.index)
    df2 = df.copy()
    df2['performance_ratio'] = np.where(df2['theoretical_power'] > 0,
                                       df2['active_power'] / df2['theoretical_power'],
                                       np.nan)
    df2['performance_ratio'] = df2['performance_ratio'].clip(lower=0)
    df2['performance_score'] = (df2['performance_ratio'] * 100).clip(0,100)

    def category(score):
        if pd.isna(score):
            return 'Unknown'
        if score >= 85:
            return 'Good'
        if score >= 60:
            return 'Moderate'
        return 'Poor'

    def suggestion(cat):
        return {
            'Good': 'Turbine performing well. Continue scheduled maintenance.',
            'Moderate': 'Investigate minor performance losses: check blade, yaw alignment, or curtailment logs.',
            'Poor': 'Significant underperformance: schedule immediate inspection — check pitch, gearbox, generator, SCADA sensors.'
        }.get(cat, 'Insufficient data')

    df2['performance_category'] = df2['performance_score'].apply(category)
    df2['performance_advice'] = df2['performance_category'].apply(suggestion)
    return df2[['performance_ratio','performance_score','performance_category','performance_advice']]

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Place dataset there or update CSV_PATH.")
    df_raw = pd.read_csv(CSV_PATH)

    df_mapped = map_columns(df_raw)

    df_time = parse_datetime_column(df_mapped)
    print("Loaded dataset shape after datetime parsing:", df_time.shape)

    plot_timeseries(df_time, title_prefix='Task 1 - EDA: ')
    scatter_power_curve(df_time)
    abnormalities, z_flags = detect_missing_and_abnormal(df_time)

    X, y = create_lag_features(df_time, n_lags=N_LAGS)
    X = X.dropna()
    y = y.loc[X.index]
    if X.empty:
        print("Not enough data after creating lag features. Try reducing N_LAGS or providing more data.")
        return

    X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_size_days=TEST_SIZE_DAYS)
    print('Train size:', X_train.shape, 'Test size:', X_test.shape)

    RF_PARAMS = {
        'n_estimators': 50,     
        'max_depth': 12,        
        'max_samples': 0.7,     
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
    }

    try:
        print("\nTraining a single multi-output RandomForest for all targets with params:", RF_PARAMS)
        t0 = time.time()
        rf_multi = RandomForestRegressor(**RF_PARAMS)
        rf_multi.fit(X_train, y_train)
        t1 = time.time()
        print(f"Multi-output RandomForest trained in {(t1-t0):.1f} seconds")
    except TypeError as te:
        print("Warning: sklearn RandomForestRegressor rejected params (dropping max_samples). Error:", te)
        RF_PARAMS.pop('max_samples', None)
        print("Retrying with params:", RF_PARAMS)
        t0 = time.time()
        rf_multi = RandomForestRegressor(**RF_PARAMS)
        rf_multi.fit(X_train, y_train)
        t1 = time.time()
        print(f"Multi-output RandomForest trained in {(t1-t0):.1f} seconds (without max_samples)")

    preds_multi = rf_multi.predict(X_test)  
    preds_df = pd.DataFrame(preds_multi, index=X_test.index, columns=y_test.columns)

    models = {'multi_rf': rf_multi}
    results = {}
    for col in y_test.columns:
        y_true = y_test[col].loc[X_test.index].values
        y_pred = preds_df[col].values
        scores = evaluate_preds(y_true, y_pred)
        results[col] = {'preds': y_pred, 'scores': scores, 'index': X_test.index}
        print(f"Evaluation for {col}: {scores}")
        plot_pred_vs_actual(X_test.index, y_test[col].values, preds_df[col].values, title=f'{col} predicted vs actual')

    underperf_df = detect_underperformance(df_time, z_thresh=3.0)
    if not underperf_df.empty:
        top_anom = underperf_df[underperf_df['underperf_anomaly']].sort_values('res_z', ascending=False).head(10)
        print("\nTop underperformance anomalies (sample):")
        print(top_anom)

    perf_df = performance_score_and_advice(df_time)
    if not perf_df.empty:
        print("\nPerformance category counts:\n", perf_df['performance_category'].value_counts(dropna=False))

    try:
        X_train.to_csv(os.path.join(OUT_DIR, 'X_train.csv'))
        X_test.to_csv(os.path.join(OUT_DIR, 'X_test.csv'))
        y_test.to_csv(os.path.join(OUT_DIR, 'y_test.csv'))
        if not perf_df.empty:
            perf_df.to_csv(os.path.join(OUT_DIR, 'performance_scores.csv'))
        if not underperf_df.empty:
            underperf_df.to_csv(os.path.join(OUT_DIR, 'underperformance_flags.csv'))
    except Exception as e:
        print("Warning: could not save outputs:", e)

if __name__ == "__main__":
    main()
