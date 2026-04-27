import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------
# Config
# -----------------------
FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready.csv"

# Time split (edit if you want)
TRAIN_END = "2025-03-01"
VAL_END   = "2025-06-01"
TEST_END  = "2025-09-01"  # should match your dataset end

OUT_METRICS = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\ml_metrics_step1.csv"

# -----------------------
# Load + basic cleanup
# -----------------------
df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["median_price"] = pd.to_numeric(df["median_price"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df.dropna(subset=["hash_name", "date", "median_price", "volume"])
df = df.sort_values(["hash_name", "date"])

# -----------------------
# Target: next-day price per item
# -----------------------
df["target_price_t1"] = df.groupby("hash_name")["median_price"].shift(-1)

# Baseline prediction: "tomorrow = today"
df["baseline_pred"] = df["median_price"]

# Drop rows where target is missing (last day of each item)
df = df.dropna(subset=["target_price_t1"])

# -----------------------
# Feature engineering (simple + robust)
# -----------------------
# lag features (per item)
df["price_lag1"] = df.groupby("hash_name")["median_price"].shift(1)
df["price_lag2"] = df.groupby("hash_name")["median_price"].shift(2)

df["vol_lag1"] = df.groupby("hash_name")["volume"].shift(1)
df["vol_lag2"] = df.groupby("hash_name")["volume"].shift(2)

# rolling means (use past values only -> shift(1))
df["price_ma7"] = df.groupby("hash_name")["median_price"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
df["vol_ma7"]   = df.groupby("hash_name")["volume"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)

# Remove rows where lags/rolling are missing
df = df.dropna(subset=["price_lag1", "price_lag2", "vol_lag1", "vol_lag2", "price_ma7", "vol_ma7"])

# -----------------------
# Time-based split
# -----------------------
train_end = pd.to_datetime(TRAIN_END)
val_end = pd.to_datetime(VAL_END)
test_end = pd.to_datetime(TEST_END)

train = df[df["date"] <= train_end].copy()
val   = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
test  = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()

print("Rows:", len(df), "Items:", df["hash_name"].nunique())
print("Train rows:", len(train), "Val rows:", len(val), "Test rows:", len(test))

# -----------------------
# Evaluate baseline
# -----------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def eval_block(name, block, y_col="target_price_t1", pred_col="baseline_pred"):
    y_true = block[y_col].values
    y_pred = block[pred_col].values
    return {
        "model": name,
        "split": block.name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "n": len(block),
    }

train.name = "train"
val.name = "val"
test.name = "test"

baseline_metrics = [
    eval_block("baseline_persistence", train),
    eval_block("baseline_persistence", val),
    eval_block("baseline_persistence", test),
]

# -----------------------
# Ridge Regression pipeline
# -----------------------
feature_cols_numeric = ["price_lag1", "price_lag2", "vol_lag1", "vol_lag2", "price_ma7", "vol_ma7"]
feature_cols_cat = []

# only add if they exist in your file
for c in ["weapon", "wear", "rarity", "is_stattrak"]:
    if c in df.columns:
        feature_cols_cat.append(c)

X_train = train[feature_cols_numeric + feature_cols_cat]
y_train = train["target_price_t1"]

X_val = val[feature_cols_numeric + feature_cols_cat]
y_val = val["target_price_t1"]

X_test = test[feature_cols_numeric + feature_cols_cat]
y_test = test["target_price_t1"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ("num", "passthrough", feature_cols_numeric),
    ],
    remainder="drop",
)

model = Ridge(alpha=1.0)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

pipe.fit(X_train, y_train)

# Predictions
val_pred = pipe.predict(X_val)
test_pred = pipe.predict(X_test)

ridge_metrics = [
    {
        "model": "ridge",
        "split": "val",
        "MAE": mean_absolute_error(y_val, val_pred),
        "RMSE": rmse(y_val, val_pred),
        "n": len(val),
    },
    {
        "model": "ridge",
        "split": "test",
        "MAE": mean_absolute_error(y_test, test_pred),
        "RMSE": rmse(y_test, test_pred),
        "n": len(test),
    },
]

# -----------------------
# Save metrics
# -----------------------
metrics = pd.DataFrame(baseline_metrics + ridge_metrics)
print("\n=== Metrics ===")
print(metrics)

metrics.to_csv(OUT_METRICS, index=False)
print("\nWrote:", OUT_METRICS)