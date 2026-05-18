# Importing relevant modules
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

# Filepath
FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready_clean_final.csv"

# Train, validation, and test end-dates
TRAIN_END = "2025-03-01"
VAL_END   = "2025-06-01"
TEST_END  = "2025-09-01"

RUN_RF = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FILE)

# Ensuring features are the correct datatypes
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["median_price"] = pd.to_numeric(df["median_price"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

# Remove invalid essential rows
df = df.dropna(subset=["hash_name", "date", "median_price", "volume"])

# Safety filters
df = df[df["median_price"] > 0]
df = df[df["volume"] >= 0]

# Sort per skin and date
df = df.sort_values(["hash_name", "date"]).copy()


# ============================================================
# TARGET CONSTRUCTION
# ============================================================

# Daily log return:
# logret_t = log(price_t) - log(price_t-1)
df["logret"] = (
    df.groupby("hash_name")["median_price"]
      .transform(lambda s: np.log(s).diff())
)

# Next-day price:
# price_next = price at t+1 within the same hash_name
df["price_next"] = (
    df.groupby("hash_name")["median_price"]
      .shift(-1)
)

# Target:
# target_logret_t1 = log(price_t+1) - log(price_t)
df["target_logret_t1"] = np.log(df["price_next"]) - np.log(df["median_price"])


# ============================================================
# FEATURE ENGINEERING: PAST-ONLY FEATURES
# ============================================================

# Lagged returns
df["logret_lag1"] = df.groupby("hash_name")["logret"].shift(1)
df["logret_lag2"] = df.groupby("hash_name")["logret"].shift(2)

# Lagged volume
df["vol_lag1"] = df.groupby("hash_name")["volume"].shift(1)
df["vol_lag2"] = df.groupby("hash_name")["volume"].shift(2)

# Rolling mean features
# IMPORTANT:
# shift(1) makes sure the rolling window only uses past information.
# transform(...) makes sure rolling is calculated separately within each hash_name.
df["logret_ma7"] = (
    df.groupby("hash_name")["logret"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

df["vol_ma7"] = (
    df.groupby("hash_name")["volume"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

# Drop rows where target or features are missing
df = df.dropna(subset=[
    "price_next",
    "target_logret_t1",
    "logret_lag1",
    "logret_lag2",
    "vol_lag1",
    "vol_lag2",
    "logret_ma7",
    "vol_ma7"
]).copy()


# ============================================================
# SANITY CHECKS
# ============================================================

print("\n============================================================")
print("SANITY CHECKS")
print("============================================================")

print("Rows after feature engineering:", len(df))
print("Number of items:", df["hash_name"].nunique())

print("\nMissing values:")
print(df[[
    "price_next",
    "target_logret_t1",
    "logret_lag1",
    "logret_lag2",
    "vol_lag1",
    "vol_lag2",
    "logret_ma7",
    "vol_ma7"
]].isna().sum())

print("\nTarget distribution:")
print(df["target_logret_t1"].describe())

print("\nTarget mean:")
print(df["target_logret_t1"].mean())

print("\nTarget standard deviation:")
print(df["target_logret_t1"].std())

# Alignment check for one item
sample_skin = df["hash_name"].iloc[0]

debug_cols = [
    "hash_name",
    "date",
    "median_price",
    "price_next",
    "logret",
    "logret_lag1",
    "logret_lag2",
    "logret_ma7",
    "target_logret_t1"
]

print("\nAlignment check for one skin:")
print(df[df["hash_name"] == sample_skin][debug_cols].head(12))


# ============================================================
# TIME SPLIT
# ============================================================

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


# ============================================================
# PREPARE MATRICES
# ============================================================

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
y_train = train["target_logret_t1"]

X_val = val[num_cols + cat_cols]
y_val = val["target_logret_t1"]

X_test = test[num_cols + cat_cols]
y_test = test["target_logret_t1"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

results = []


# ============================================================
# BASELINES
# ============================================================

# Baseline 1:
# Predict zero return for all rows
val_base_zero = np.zeros(len(val))
test_base_zero = np.zeros(len(test))

results.append(report("baseline_zero_return", "val", y_val, val_base_zero))
results.append(report("baseline_zero_return", "test", y_test, test_base_zero))

# Baseline 2:
# Predict tomorrow's return as yesterday's return
results.append(report("baseline_lag1_return", "val", y_val, val["logret_lag1"]))
results.append(report("baseline_lag1_return", "test", y_test, test["logret_lag1"]))

# Baseline 3:
# Predict tomorrow's return as the 7-day past average return
results.append(report("baseline_ma7_return", "val", y_val, val["logret_ma7"]))
results.append(report("baseline_ma7_return", "test", y_test, test["logret_ma7"]))


# ============================================================
# RIDGE REGRESSION
# ============================================================

ridge = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0))
])

ridge.fit(X_train, y_train)

val_pred_ridge = ridge.predict(X_val)
test_pred_ridge = ridge.predict(X_test)

results.append(report("ridge", "val", y_val, val_pred_ridge))
results.append(report("ridge", "test", y_test, test_pred_ridge))


# ============================================================
# RANDOM FOREST
# ============================================================

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

    val_pred_rf = rf.predict(X_val)
    test_pred_rf = rf.predict(X_test)

    results.append(report("random_forest", "val", y_val, val_pred_rf))
    results.append(report("random_forest", "test", y_test, test_pred_rf))


# ============================================================
# PRINT AND SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df["MAE"] = results_df["MAE"].round(6)
results_df["RMSE"] = results_df["RMSE"].round(6)
results_df["R2"] = results_df["R2"].round(6)

print("\n============================================================")
print("RESULTS: ONE-DAY LOG RETURN FORECASTING")
print("============================================================")
print(results_df)

# Save results as CSV
results_df.to_csv("one_day_logret_results.csv", index=False)

print("\nSaved results to: one_day_logret_results.csv")