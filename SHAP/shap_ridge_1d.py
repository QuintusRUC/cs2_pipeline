# Importing relevant modules
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CONFIG

FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready_clean_final.csv"

TRAIN_END = "2025-03-01"
VAL_END   = "2025-06-01"
TEST_END  = "2025-09-01"

HORIZON_DAYS = 1
SHAP_SAMPLE_SIZE = 3000
RANDOM_STATE = 42

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

# TARGET CONSTRUCTION: EXACT 1 CALENDAR DAY AHEAD

df["logret"] = (
    df.groupby("hash_name")["median_price"]
      .transform(lambda s: np.log(s).diff())
)

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
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

# RIDGE REGRESSION

ridge = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0))
])

ridge.fit(X_train, y_train)

val_pred = ridge.predict(X_val)
test_pred = ridge.predict(X_test)

results = []

results.append(report("ridge", "val", y_val, val_pred))
results.append(report("ridge", "test", y_test, test_pred))

results_df = pd.DataFrame(results)

results_df["MAE"] = results_df["MAE"].round(6)
results_df["RMSE"] = results_df["RMSE"].round(6)
results_df["R2"] = results_df["R2"].round(6)

print("\n============================================================")
print("RIDGE PERFORMANCE")
print("============================================================")
print(results_df)

results_df.to_csv("shap_ridge_1d_model_results.csv", index=False)

# SHAP DATA PREPARATION

X_train_transformed = ridge.named_steps["preprocess"].transform(X_train)
X_test_transformed = ridge.named_steps["preprocess"].transform(X_test)

feature_names = ridge.named_steps["preprocess"].get_feature_names_out()

X_train_transformed = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

if len(X_test_transformed) > SHAP_SAMPLE_SIZE:
    X_test_sample = X_test_transformed.sample(
        n=SHAP_SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
else:
    X_test_sample = X_test_transformed.copy()

# SHAP EXPLANATION

ridge_model = ridge.named_steps["model"]

explainer = shap.LinearExplainer(
    ridge_model,
    X_train_transformed
)

shap_values = explainer(X_test_sample)

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\n============================================================")
print("SHAP FEATURE IMPORTANCE")
print("============================================================")
print(shap_importance.head(30))

shap_importance.to_csv("shap_ridge_1d_feature_importance.csv", index=False)

# RIDGE COEFFICIENTS

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": ridge_model.coef_
}).sort_values("coefficient", ascending=False)

coef_df.to_csv("ridge_1d_coefficients.csv", index=False)

# SHAP BAR PLOT

top_n = 20
top_features = shap_importance.head(top_n).sort_values("mean_abs_shap")

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features["mean_abs_shap"])
plt.xlabel("Mean absolute SHAP value")
plt.ylabel("Feature")
plt.title("Top SHAP features - Ridge one-day model")
plt.tight_layout()
plt.savefig("shap_ridge_1d_top20_bar.png", dpi=300)
plt.close()

# SHAP SUMMARY PLOT

plt.figure()

shap.summary_plot(
    shap_values.values,
    X_test_sample,
    feature_names=feature_names,
    show=False,
    max_display=20
)

plt.tight_layout()
plt.savefig("shap_ridge_1d_summary.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n============================================================")
print("FILES SAVED")
print("============================================================")
print("shap_ridge_1d_model_results.csv")
print("shap_ridge_1d_feature_importance.csv")
print("ridge_1d_coefficients.csv")
print("shap_ridge_1d_top20_bar.png")
print("shap_ridge_1d_summary.png")