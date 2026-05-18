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

FILE = r"C:\Users\nikol\Desktop\cs2_pipeline\data_exploration\model_ready_clean_final.csv"

TRAIN_END = "2025-03-01"
VAL_END   = "2025-06-01"
TEST_END  = "2025-09-01"

RUN_RF = True
HORIZON_DAYS = 1


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

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["median_price"] = pd.to_numeric(df["median_price"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df.dropna(subset=["hash_name", "date", "median_price", "volume"])

# Safety filters
df = df[df["median_price"] > 0]
df = df[df["volume"] >= 0]

df = df.sort_values(["hash_name", "date"]).copy()


# ============================================================
# TARGET CONSTRUCTION: EXACT 1-DAY RAW PRICE
# ============================================================

# Future date and future price within each skin
df["future_date"] = (
    df.groupby("hash_name")["date"]
      .shift(-HORIZON_DAYS)
)

df["target_price_h"] = (
    df.groupby("hash_name")["median_price"]
      .shift(-HORIZON_DAYS)
)

# Calendar gap between current row and future target row
df["target_gap_days"] = (df["future_date"] - df["date"]).dt.days

print("\n============================================================")
print("TARGET GAP BEFORE EXACT-HORIZON FILTER")
print("============================================================")
print(df["target_gap_days"].value_counts(dropna=False).sort_index().head(20))

# Keep only exact one-calendar-day targets
df = df[df["target_gap_days"] == HORIZON_DAYS].copy()


# ============================================================
# FEATURE ENGINEERING: RAW PRICE FEATURES + PAST FEATURES
# ============================================================

# Current price is allowed, because it is known at prediction time
df["price_t"] = df["median_price"]

# Price lags
df["price_lag1"] = df.groupby("hash_name")["median_price"].shift(1)
df["price_lag2"] = df.groupby("hash_name")["median_price"].shift(2)

# Price rolling mean, using only past values
df["price_ma7"] = (
    df.groupby("hash_name")["median_price"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

# Daily log return, for extra past movement features
df["logret"] = (
    df.groupby("hash_name")["median_price"]
      .transform(lambda s: np.log(s).diff())
)

df["logret_lag1"] = df.groupby("hash_name")["logret"].shift(1)
df["logret_lag2"] = df.groupby("hash_name")["logret"].shift(2)

df["logret_ma7"] = (
    df.groupby("hash_name")["logret"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

# Volume features
df["vol_lag1"] = df.groupby("hash_name")["volume"].shift(1)
df["vol_lag2"] = df.groupby("hash_name")["volume"].shift(2)

df["vol_ma7"] = (
    df.groupby("hash_name")["volume"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)


# ============================================================
# DROP MISSING VALUES
# ============================================================

num_cols = [
    "price_t",
    "price_lag1",
    "price_lag2",
    "price_ma7",
    "logret_lag1",
    "logret_lag2",
    "logret_ma7",
    "vol_lag1",
    "vol_lag2",
    "vol_ma7",
]

cat_cols = [
    c for c in ["weapon", "wear", "rarity", "is_stattrak"]
    if c in df.columns
]

needed = [
    "future_date",
    "target_price_h",
    "target_gap_days",
] + num_cols

df = df.dropna(subset=needed).copy()


# ============================================================
# SANITY CHECKS
# ============================================================

print("\n============================================================")
print("SANITY CHECKS")
print("============================================================")

print("Experiment: exact one-day raw price prediction")
print("Rows after filtering + feature engineering:", len(df))
print("Number of items:", df["hash_name"].nunique())

print("\nTarget gap counts after filtering:")
print(df["target_gap_days"].value_counts(dropna=False).sort_index())

print("\nTarget price distribution:")
print(df["target_price_h"].describe())

sample_skin = df["hash_name"].iloc[0]

debug_cols = [
    "hash_name",
    "date",
    "future_date",
    "target_gap_days",
    "median_price",
    "target_price_h",
    "price_t",
    "price_lag1",
    "price_ma7",
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

X_train = train[num_cols + cat_cols]
y_train = train["target_price_h"]

X_val = val[num_cols + cat_cols]
y_val = val["target_price_h"]

X_test = test[num_cols + cat_cols]
y_test = test["target_price_h"]

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
# Persistence baseline: predict tomorrow's price as today's price
results.append(report("baseline_current_price", "val", y_val, val["price_t"]))
results.append(report("baseline_current_price", "test", y_test, test["price_t"]))

# Baseline 2:
# Predict tomorrow's price as yesterday's price
results.append(report("baseline_price_lag1", "val", y_val, val["price_lag1"]))
results.append(report("baseline_price_lag1", "test", y_test, test["price_lag1"]))

# Baseline 3:
# Predict tomorrow's price as previous 7-day average price
results.append(report("baseline_price_ma7", "val", y_val, val["price_ma7"]))
results.append(report("baseline_price_ma7", "test", y_test, test["price_ma7"]))


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

results.append(report("ridge_raw_price", "val", y_val, val_pred_ridge))
results.append(report("ridge_raw_price", "test", y_test, test_pred_ridge))


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

    results.append(report("random_forest_raw_price", "val", y_val, val_pred_rf))
    results.append(report("random_forest_raw_price", "test", y_test, test_pred_rf))


# ============================================================
# PRINT AND SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df["MAE"] = results_df["MAE"].round(6)
results_df["RMSE"] = results_df["RMSE"].round(6)
results_df["R2"] = results_df["R2"].round(6)

print("\n============================================================")
print("RESULTS: EXACT ONE-DAY RAW PRICE PREDICTION")
print("============================================================")
print(results_df)

results_df.to_csv("SLET.CSV", index=False)

print("\nSaved results to: SLET.csv")