# Importing relevant modules
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CONFIG

FILE = r"C:\Users\nikol\Desktop\cs2_pipeline\dummy_linear_ridge_input.csv"

TRAIN_END = "2025-03-01"
VAL_END   = "2025-06-01"
TEST_END  = "2025-09-01"

RUN_RF = True
HORIZON_DAYS = 1

# HELPER FUNCTIONS

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def report(name, split, y_true, y_pred):
    return {
        "model": name,
        "split": split,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "n": len(y_true),
    }

# LOAD DATA

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["median_price"] = pd.to_numeric(df["median_price"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df.dropna(subset=["hash_name", "date", "median_price", "volume"])

# Safety filters
df = df[df["median_price"] > 0]
df = df[df["volume"] >= 0]

df = df.sort_values(["hash_name", "date"]).copy()

# TARGET CONSTRUCTION: EXACT 1 CALENDAR DAY AHEAD
# Daily log return:
# logret_t = log(price_t) - log(price_t-1)

df["logret"] = (
    df.groupby("hash_name")["median_price"]
      .transform(lambda s: np.log(s).diff())
)

# Next observation within each skin
df["future_date"] = (
    df.groupby("hash_name")["date"]
      .shift(-HORIZON_DAYS)
)

df["price_h"] = (
    df.groupby("hash_name")["median_price"]
      .shift(-HORIZON_DAYS)
)

# Calendar gap between current row and future target row
df["target_gap_days"] = (df["future_date"] - df["date"]).dt.days

print("\n============================================================")
print("TARGET GAP BEFORE EXACT-HORIZON FILTER")
print("============================================================")
print(df["target_gap_days"].value_counts(dropna=False).sort_index().head(20))

# Keep only rows where the next observation is exactly +1 calendar day
df = df[df["target_gap_days"] == HORIZON_DAYS].copy()

# Target:
# target_logret_h = log(price_t+1) - log(price_t)
df["target_logret_h"] = np.log(df["price_h"]) - np.log(df["median_price"])

# FEATURE ENGINEERING: PAST-ONLY FEATURES

df["logret_lag1"] = df.groupby("hash_name")["logret"].shift(1)
df["logret_lag2"] = df.groupby("hash_name")["logret"].shift(2)

df["vol_lag1"] = df.groupby("hash_name")["volume"].shift(1)
df["vol_lag2"] = df.groupby("hash_name")["volume"].shift(2)

df["logret_ma7"] = (
    df.groupby("hash_name")["logret"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

df["vol_ma7"] = (
    df.groupby("hash_name")["volume"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

needed = [
    "price_h",
    "future_date",
    "target_logret_h",
    "target_gap_days",
    "logret_lag1",
    "logret_lag2",
    "vol_lag1",
    "vol_lag2",
    "logret_ma7",
    "vol_ma7"
]

df = df.dropna(subset=needed).copy()

# SANITY CHECKS

print("\n============================================================")
print("SANITY CHECKS")
print("============================================================")

print("Exact horizon days:", HORIZON_DAYS)
print("Rows after exact-horizon filter + feature engineering:", len(df))
print("Number of items:", df["hash_name"].nunique())

print("\nTarget gap counts after filtering:")
print(df["target_gap_days"].value_counts(dropna=False).sort_index())

print("\nMissing values:")
print(df[needed].isna().sum())

print("\nTarget distribution:")
print(df["target_logret_h"].describe())

sample_skin = df["hash_name"].iloc[0]

debug_cols = [
    "hash_name",
    "date",
    "future_date",
    "target_gap_days",
    "median_price",
    "price_h",
    "logret",
    "logret_lag1",
    "logret_lag2",
    "logret_ma7",
    "target_logret_h"
]

print("\nAlignment check for one skin:")
print(df[df["hash_name"] == sample_skin][debug_cols].head(12))

# TIME SPLIT

train_end = pd.to_datetime(TRAIN_END)
val_end = pd.to_datetime(VAL_END)
test_end = pd.to_datetime(TEST_END)

train = df[df["date"] <= train_end].copy()
val   = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
test  = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()

print("\n============================================================")
print("DATA SPLIT")
print("============================================================")
print("Rows used:", len(df))
print("Items:", df["hash_name"].nunique())
print("Train/Val/Test:", len(train), len(val), len(test))

print("\nDate ranges:")
print("Train:", train["date"].min(), "->", train["date"].max())
print("Val:  ", val["date"].min(), "->", val["date"].max())
print("Test: ", test["date"].min(), "->", test["date"].max())

# PREPARE MATRICES

num_cols = [
    "logret_lag1",
    "logret_lag2",
    "vol_lag1",
    "vol_lag2",
    "logret_ma7",
    "vol_ma7"
]

cat_cols = [
    c for c in ["weapon", "wear", "rarity", "is_stattrak"]
    if c in df.columns
]

X_train = train[num_cols + cat_cols]
y_train = train["target_logret_h"]

X_val = val[num_cols + cat_cols]
y_val = val["target_logret_h"]

X_test = test[num_cols + cat_cols]
y_test = test["target_logret_h"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

results = []

# BASELINES
# Baseline 1: predict zero return
results.append(report("baseline_zero_return", "val", y_val, np.zeros(len(val))))
results.append(report("baseline_zero_return", "test", y_test, np.zeros(len(test))))

# Baseline 2: predict future return as yesterday's return
results.append(report("baseline_lag1_return", "val", y_val, val["logret_lag1"]))
results.append(report("baseline_lag1_return", "test", y_test, test["logret_lag1"]))

# Baseline 3: predict future return as previous 7-day average return
results.append(report("baseline_ma7_return", "val", y_val, val["logret_ma7"]))
results.append(report("baseline_ma7_return", "test", y_test, test["logret_ma7"]))

# RIDGE REGRESSION

ridge = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0))
])

ridge.fit(X_train, y_train)

results.append(report("ridge", "val", y_val, ridge.predict(X_val)))
results.append(report("ridge", "test", y_test, ridge.predict(X_test)))

# RANDOM FOREST

if RUN_RF:
    rf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=10
        ))
    ])

    rf.fit(X_train, y_train)

    results.append(report("random_forest", "val", y_val, rf.predict(X_val)))
    results.append(report("random_forest", "test", y_test, rf.predict(X_test)))

# PRINT AND SAVE RESULTS

results_df = pd.DataFrame(results)

results_df["MAE"] = results_df["MAE"].round(6)
results_df["RMSE"] = results_df["RMSE"].round(6)
results_df["R2"] = results_df["R2"].round(6)

print("\n============================================================")
print("RESULTS: EXACT ONE-DAY LOG RETURN FORECASTING")
print("============================================================")
print(results_df)

results_df.to_csv("DUMMY_RESULTS.csv", index=False)

print("\nSaved results to: DUMMY_RESULTS.csv")