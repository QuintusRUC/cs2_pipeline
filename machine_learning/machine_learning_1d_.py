# Importing relevant modules
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Filepath
FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready_clean_final.csv"

# Train, validation, and test end-dates
TRAIN_END = "2025-03-01"
VAL_END   = "2025-06-01"
TEST_END  = "2025-09-01"

RUN_RF = True  

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

# Loading the cleaned and final dataset
df = pd.read_csv(FILE)

# Ensuring features are the correct datatypes
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["median_price"] = pd.to_numeric(df["median_price"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df.dropna(subset=["hash_name", "date", "median_price", "volume"])
# Sort per skin and date
df = df.sort_values(["hash_name", "date"])


# Next-day price (price_next) and log return (target_logret_t1)

df["price_next"] = df.groupby("hash_name")["median_price"].shift(-1)
df = df.dropna(subset=["price_next"])
df["target_logret_t1"] = np.log(df["price_next"]) - np.log(df["median_price"])

# Create daily log return series for lags
df["logret"] = df.groupby("hash_name")["median_price"].apply(lambda s: np.log(s).diff()).reset_index(level=0, drop=True)


# Feature engineering (past only)

df["logret_lag1"] = df.groupby("hash_name")["logret"].shift(1)
df["logret_lag2"] = df.groupby("hash_name")["logret"].shift(2)

df["vol_lag1"] = df.groupby("hash_name")["volume"].shift(1)
df["vol_lag2"] = df.groupby("hash_name")["volume"].shift(2)

df["logret_ma7"] = df.groupby("hash_name")["logret"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
df["vol_ma7"] = df.groupby("hash_name")["volume"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)

df = df.dropna(subset=["logret_lag1","logret_lag2","vol_lag1","vol_lag2","logret_ma7","vol_ma7"])


# Time split

train_end = pd.to_datetime(TRAIN_END)
val_end = pd.to_datetime(VAL_END)
test_end = pd.to_datetime(TEST_END)

train = df[df["date"] <= train_end].copy()
val   = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
test  = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()

print("Rows used:", len(df), "| Items:", df["hash_name"].nunique())
print("Train/Val/Test:", len(train), len(val), len(test))


# Prepare matrices

num_cols = ["logret_lag1","logret_lag2","vol_lag1","vol_lag2","logret_ma7","vol_ma7"]
cat_cols = [c for c in ["weapon","wear","rarity","is_stattrak"] if c in df.columns]

X_train = train[num_cols + cat_cols]
y_train = train["target_logret_t1"]

X_val = val[num_cols + cat_cols]
y_val = val["target_logret_t1"]

X_test = test[num_cols + cat_cols]
y_test = test["target_logret_t1"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

results = []


# Baseline: 0 return

val_base = np.zeros(len(val))
test_base = np.zeros(len(test))
results.append(report("baseline_zero_return", "val", y_val, val_base))
results.append(report("baseline_zero_return", "test", y_test, test_base))


# Ridge

ridge = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0))
])
ridge.fit(X_train, y_train)

val_pred = ridge.predict(X_val)
test_pred = ridge.predict(X_test)

results.append(report("ridge", "val", y_val, val_pred))
results.append(report("ridge", "test", y_test, test_pred))


# Random Forest (optional)

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


# Print results

results_df = pd.DataFrame(results)
print("\nRESULTS (log returns)")
print(results_df)   