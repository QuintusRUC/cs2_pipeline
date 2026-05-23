# Importing relevant modules
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# CONFIG

FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready_clean_final.csv"

TRAIN_END = "2025-03-01"
VAL_END   = "2025-06-01"
TEST_END  = "2025-09-01"

RUN_RF = True
HORIZON_DAYS = 7


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

df = df[df["median_price"] > 0]
df = df[df["volume"] >= 0]

df = df.sort_values(["hash_name", "date"]).copy()


# TARGET CONSTRUCTION: EXACT 7 CALENDAR DAYS AHEAD

df["future_date"] = (
    df.groupby("hash_name")["date"]
      .shift(-HORIZON_DAYS)
)

df["price_h"] = (
    df.groupby("hash_name")["median_price"]
      .shift(-HORIZON_DAYS)
)

df["target_gap_days"] = (df["future_date"] - df["date"]).dt.days

print("\n============================================================")
print("TARGET GAP BEFORE EXACT-HORIZON FILTER")
print("============================================================")
print(df["target_gap_days"].value_counts(dropna=False).sort_index().head(20))

df = df[df["target_gap_days"] == HORIZON_DAYS].copy()

df["target_logret_h"] = np.log(df["price_h"]) - np.log(df["median_price"])


# FEATURE ENGINEERING: ORIGINAL PAST-ONLY FEATURES

df["logret"] = (
    df.groupby("hash_name")["median_price"]
      .transform(lambda s: np.log(s).diff())
)

df["logret_lag1"] = df.groupby("hash_name")["logret"].shift(1)
df["logret_lag2"] = df.groupby("hash_name")["logret"].shift(2)

df["logret_sum7"] = (
    df.groupby("hash_name")["logret"]
      .transform(lambda s: s.shift(1).rolling(7).sum())
)

df["logret_std7"] = (
    df.groupby("hash_name")["logret"]
      .transform(lambda s: s.shift(1).rolling(7).std())
)

df["vol_lag1"] = df.groupby("hash_name")["volume"].shift(1)
df["vol_lag2"] = df.groupby("hash_name")["volume"].shift(2)

df["vol_ma7"] = (
    df.groupby("hash_name")["volume"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

df["vol_std7"] = (
    df.groupby("hash_name")["volume"]
      .transform(lambda s: s.shift(1).rolling(7).std())
)

df["vol_z7"] = (
    (df["vol_lag1"] - df["vol_ma7"]) / (df["vol_std7"] + 1e-9)
)


# FEATURE ENGINEERING: PRICE DEVIATION MA7

df["price_log"] = np.log(df["median_price"])

df["price_log_lag1"] = (
    df.groupby("hash_name")["price_log"]
      .shift(1)
)

df["price_log_ma7"] = (
    df.groupby("hash_name")["price_log"]
      .transform(lambda s: s.shift(1).rolling(7).mean())
)

df["price_deviation_ma7"] = (
    df["price_log_lag1"] - df["price_log_ma7"]
)


# DROP MISSING VALUES

num_cols = [
    "logret_lag1",
    "logret_lag2",
    "logret_sum7",
    "logret_std7",
    "vol_lag1",
    "vol_lag2",
    "vol_ma7",
    "vol_std7",
    "vol_z7",
    "price_deviation_ma7",
]

cat_cols = [
    c for c in ["weapon", "wear", "rarity", "is_stattrak"]
    if c in df.columns
]

needed = [
    "price_h",
    "future_date",
    "target_logret_h",
    "target_gap_days",
] + num_cols

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
    "target_logret_h",
    "logret",
    "logret_lag1",
    "logret_lag2",
    "logret_sum7",
    "logret_std7",
    "price_log_lag1",
    "price_log_ma7",
    "price_deviation_ma7",
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


# TARGET DISTRIBUTION PER SPLIT

target_summary = pd.DataFrame({
    "train": train["target_logret_h"].describe(),
    "val": val["target_logret_h"].describe(),
    "test": test["target_logret_h"].describe(),
})

print("\n============================================================")
print("TARGET DISTRIBUTION PER SPLIT")
print("============================================================")
print(target_summary)


# CORRELATION MATRIX: SEVEN-DAY MODEL WITH PRICE_DEVIATION_MA7

corr_cols = [
    "target_logret_h",
    "logret_lag1",
    "logret_lag2",
    "logret_sum7",
    "logret_std7",
    "vol_lag1",
    "vol_lag2",
    "vol_ma7",
    "vol_std7",
    "vol_z7",
    "price_deviation_ma7",
]

corr_data = train[corr_cols].copy()
corr_matrix = corr_data.corr(numeric_only=True)

print("\n============================================================")
print("CORRELATION MATRIX: SEVEN-DAY MODEL WITH PRICE_DEVIATION_MA7")
print("============================================================")
print(corr_matrix)

plt.figure(figsize=(13, 9))
plt.imshow(corr_matrix, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(
    ticks=np.arange(len(corr_cols)),
    labels=corr_cols,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(corr_cols)),
    labels=corr_cols
)

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        plt.text(
            j,
            i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

plt.title("Correlation matrix - seven-day model with price_deviation_ma7")
plt.tight_layout()
plt.savefig("correlation_matrix_seven_day_price_deviation.png", dpi=300)
plt.close()

print("\nSaved correlation matrix to:")
print("correlation_matrix_seven_day_price_deviation.png")


# CORRELATION WITH TARGET

target_corr = (
    corr_matrix["target_logret_h"]
    .drop("target_logret_h")
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

print("\n============================================================")
print("CORRELATION WITH TARGET: TRAIN SET")
print("============================================================")
print(target_corr)


# PREPARE MATRICES

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

train_base = np.zeros(len(train))
val_base = np.zeros(len(val))
test_base = np.zeros(len(test))

results.append(report("baseline_zero_return", "train", y_train, train_base))
results.append(report("baseline_zero_return", "val", y_val, val_base))
results.append(report("baseline_zero_return", "test", y_test, test_base))

results.append(report("baseline_lag1_return", "train", y_train, train["logret_lag1"]))
results.append(report("baseline_lag1_return", "val", y_val, val["logret_lag1"]))
results.append(report("baseline_lag1_return", "test", y_test, test["logret_lag1"]))

results.append(report("baseline_sum7_return", "train", y_train, train["logret_sum7"]))
results.append(report("baseline_sum7_return", "val", y_val, val["logret_sum7"]))
results.append(report("baseline_sum7_return", "test", y_test, test["logret_sum7"]))


# RIDGE REGRESSION

ridge = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Ridge(alpha=10.0))
])

ridge.fit(X_train, y_train)

train_pred_ridge = ridge.predict(X_train)
val_pred_ridge = ridge.predict(X_val)
test_pred_ridge = ridge.predict(X_test)

results.append(report("ridge_price_deviation", "train", y_train, train_pred_ridge))
results.append(report("ridge_price_deviation", "val", y_val, val_pred_ridge))
results.append(report("ridge_price_deviation", "test", y_test, test_pred_ridge))


# RANDOM FOREST

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

    train_pred_rf = rf.predict(X_train)
    val_pred_rf = rf.predict(X_val)
    test_pred_rf = rf.predict(X_test)

    results.append(report("random_forest_price_deviation", "train", y_train, train_pred_rf))
    results.append(report("random_forest_price_deviation", "val", y_val, val_pred_rf))
    results.append(report("random_forest_price_deviation", "test", y_test, test_pred_rf))


# PRINT AND SAVE RESULTS

results_df = pd.DataFrame(results)

results_df["MAE"] = results_df["MAE"].round(6)
results_df["RMSE"] = results_df["RMSE"].round(6)
results_df["R2"] = results_df["R2"].round(6)

print("\n============================================================")
print("RESULTS: EXACT SEVEN-DAY WITH PRICE_DEVIATION_MA7")
print("============================================================")
print(results_df)

results_df.to_csv("seven_day_price_deviation_results.csv", index=False)

print("\n============================================================")
print("FILES SAVED")
print("============================================================")
print("seven_day_price_deviation_results.csv")
print("correlation_matrix_seven_day_price_deviation.png")