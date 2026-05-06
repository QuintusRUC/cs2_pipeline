import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------
# CONFIG
# -----------------------
FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready_clean_final.csv"

TRAIN_END = "2025-03-01"
VAL_END   = "2025-06-01"
TEST_END  = "2025-09-01"

RUN_RF = True
HORIZON_DAYS = 7  # 7-day ahead target

# -----------------------
# Helpers
# -----------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def report(name, split, y_true, y_pred):
    return {
        "model": name,
        "split": split,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "n": len(y_true),
    }

# -----------------------
# Load + cleanup
# -----------------------
df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["median_price"] = pd.to_numeric(df["median_price"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df.dropna(subset=["hash_name", "date", "median_price", "volume"])
df = df.sort_values(["hash_name", "date"])

# -----------------------
# Target: HORIZON-day log return
# -----------------------
df["price_h"] = df.groupby("hash_name")["median_price"].shift(-HORIZON_DAYS)
df = df.dropna(subset=["price_h"])
df["target_logret_h"] = np.log(df["price_h"]) - np.log(df["median_price"])

# Baseline prediction: 0 return
df["baseline_pred"] = 0.0

# -----------------------
# Create daily log returns series for features
# -----------------------
df["logret"] = df.groupby("hash_name")["median_price"].apply(lambda s: np.log(s).diff()).reset_index(level=0, drop=True)

# -----------------------
# Feature engineering (past only)
# -----------------------
df["logret_lag1"] = df.groupby("hash_name")["logret"].shift(1)
df["logret_lag2"] = df.groupby("hash_name")["logret"].shift(2)

df["logret_sum7"] = df.groupby("hash_name")["logret"].shift(1).rolling(7).sum().reset_index(level=0, drop=True)
df["logret_std7"] = df.groupby("hash_name")["logret"].shift(1).rolling(7).std().reset_index(level=0, drop=True)

df["vol_lag1"] = df.groupby("hash_name")["volume"].shift(1)
df["vol_lag2"] = df.groupby("hash_name")["volume"].shift(2)

df["vol_ma7"] = df.groupby("hash_name")["volume"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
df["vol_std7"] = df.groupby("hash_name")["volume"].shift(1).rolling(7).std().reset_index(level=0, drop=True)
df["vol_z7"] = (df["vol_lag1"] - df["vol_ma7"]) / (df["vol_std7"] + 1e-9)

needed = [
    "target_logret_h",
    "logret_lag1","logret_lag2",
    "logret_sum7","logret_std7",
    "vol_lag1","vol_lag2",
    "vol_ma7","vol_std7","vol_z7"
]
df = df.dropna(subset=needed)

# -----------------------
# Time split
# -----------------------
train_end = pd.to_datetime(TRAIN_END)
val_end = pd.to_datetime(VAL_END)
test_end = pd.to_datetime(TEST_END)

train = df[df["date"] <= train_end].copy()
val   = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
test  = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()

print("Horizon (days):", HORIZON_DAYS)
print("Rows used:", len(df), "| Items:", df["hash_name"].nunique())
print("Train/Val/Test:", len(train), len(val), len(test))

# -----------------------
# Prepare matrices
# -----------------------
num_cols = [
    "logret_lag1","logret_lag2",
    "logret_sum7","logret_std7",
    "vol_lag1","vol_lag2",
    "vol_ma7","vol_std7","vol_z7"
]
cat_cols = [c for c in ["weapon","wear","rarity","is_stattrak"] if c in df.columns]

X_train = train[num_cols + cat_cols]
y_train = train["target_logret_h"]

X_val = val[num_cols + cat_cols]
y_val = val["target_logret_h"]

X_test = test[num_cols + cat_cols]
y_test = test["target_logret_h"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

results = []

# -----------------------
# Baseline
# -----------------------
results.append(report("baseline_zero_return", "val", y_val, np.zeros(len(val))))
results.append(report("baseline_zero_return", "test", y_test, np.zeros(len(test))))

# -----------------------
# Ridge
# -----------------------
ridge = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Ridge(alpha=10.0))
])
ridge.fit(X_train, y_train)

results.append(report("ridge", "val", y_val, ridge.predict(X_val)))
results.append(report("ridge", "test", y_test, ridge.predict(X_test)))

# -----------------------
# Random Forest
# -----------------------
if RUN_RF:
    rf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=10
        ))
    ])
    rf.fit(X_train, y_train)

    results.append(report("random_forest", "val", y_val, rf.predict(X_val)))
    results.append(report("random_forest", "test", y_test, rf.predict(X_test)))

# -----------------------
# Print results
# -----------------------
results_df = pd.DataFrame(results)
print("\n=== RESULTS (7-day log returns) ===")
print(results_df)