# ============================================================
# DAY 1 EXPERIMENT WITH SANITY CHECKS
# ============================================================

# Importing relevant modules
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone


# ============================================================
# SETTINGS
# ============================================================

FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready_clean_final.csv"

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
        "n": len(y_true),
    }


def describe_y(name, y):
    print(
        f"{name}: "
        f"mean={y.mean():.10f}, "
        f"std={y.std():.10f}, "
        f"min={y.min():.10f}, "
        f"max={y.max():.10f}"
    )


def describe_pred(name, pred):
    print(
        f"{name}: "
        f"mean={np.mean(pred):.10f}, "
        f"std={np.std(pred):.10f}, "
        f"min={np.min(pred):.10f}, "
        f"max={np.max(pred):.10f}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FILE)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["median_price"] = pd.to_numeric(df["median_price"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df.dropna(subset=["hash_name", "date", "median_price", "volume"])

df = df.sort_values(["hash_name", "date"]).reset_index(drop=True)


# ============================================================
# CREATE TARGET: NEXT-DAY LOG RETURN
# ============================================================

df["price_next"] = df.groupby("hash_name")["median_price"].shift(-1)

df = df.dropna(subset=["price_next"]).copy()

df["target_logret_t1"] = np.log(df["price_next"]) - np.log(df["median_price"])


# ============================================================
# CREATE PAST-ONLY FEATURES
# ============================================================

df["logret"] = df.groupby("hash_name")["median_price"].transform(
    lambda s: np.log(s).diff()
)

df["logret_lag1"] = df.groupby("hash_name")["logret"].shift(1)
df["logret_lag2"] = df.groupby("hash_name")["logret"].shift(2)

df["vol_lag1"] = df.groupby("hash_name")["volume"].shift(1)
df["vol_lag2"] = df.groupby("hash_name")["volume"].shift(2)

# Important:
# The rolling mean must be calculated inside each hash_name group.
# Otherwise it can leak across different skins.
df["logret_ma7"] = df.groupby("hash_name")["logret"].transform(
    lambda s: s.shift(1).rolling(7).mean()
)

df["vol_ma7"] = df.groupby("hash_name")["volume"].transform(
    lambda s: s.shift(1).rolling(7).mean()
)

required_feature_cols = [
    "logret_lag1",
    "logret_lag2",
    "vol_lag1",
    "vol_lag2",
    "logret_ma7",
    "vol_ma7",
]

df = df.dropna(subset=required_feature_cols).copy()


# ============================================================
# TIME SPLIT
# ============================================================

train_end = pd.to_datetime(TRAIN_END)
val_end = pd.to_datetime(VAL_END)
test_end = pd.to_datetime(TEST_END)

train = df[df["date"] <= train_end].copy()
val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
test = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

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
    "vol_ma7",
]

cat_cols = [
    c for c in ["weapon", "wear", "rarity", "is_stattrak"]
    if c in df.columns
]

feature_cols = num_cols + cat_cols

X_train = train[feature_cols].copy()
y_train = train["target_logret_t1"].copy()

X_val = val[feature_cols].copy()
y_val = val["target_logret_t1"].copy()

X_test = test[feature_cols].copy()
y_test = test["target_logret_t1"].copy()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)


# ============================================================
# MODELS
# ============================================================

results = []

# Baseline: predict 0 return
val_base = np.zeros(len(y_val))
test_base = np.zeros(len(y_test))

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


# Random Forest
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
# PRINT RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("\nRaw results:")
print(results_df)

print("\nResults without scientific notation:")
pretty_results = results_df.copy()
pretty_results["MAE"] = pretty_results["MAE"].map(lambda x: f"{x:.10f}")
pretty_results["RMSE"] = pretty_results["RMSE"].map(lambda x: f"{x:.10f}")
print(pretty_results)


# ============================================================
# SANITY CHECKS
# ============================================================

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Target/future columns must not be in X
# ------------------------------------------------------------

print("\n1) Checking that target/future columns are not in X")

for forbidden_col in ["target_logret_t1", "price_next", "logret", "median_price"]:
    assert forbidden_col not in X_train.columns, f"ERROR: {forbidden_col} is in X_train!"
    assert forbidden_col not in X_val.columns, f"ERROR: {forbidden_col} is in X_val!"
    assert forbidden_col not in X_test.columns, f"ERROR: {forbidden_col} is in X_test!"

print("OK: target/future/raw price columns are not in X.")


# ------------------------------------------------------------
# 2. Time split checks
# ------------------------------------------------------------

print("\n2) Checking time split")

assert train["date"].max() <= train_end, "ERROR: train contains dates after TRAIN_END"
assert val["date"].min() > train_end, "ERROR: val contains train-period dates"
assert val["date"].max() <= val_end, "ERROR: val contains dates after VAL_END"
assert test["date"].min() > val_end, "ERROR: test contains val-period dates"
assert test["date"].max() <= test_end, "ERROR: test contains dates after TEST_END"

print("OK: time split looks correct.")


# ------------------------------------------------------------
# 3. Train/val/test row overlap
# ------------------------------------------------------------

print("\n3) Checking overlap between train/val/test rows")

train_keys = set(zip(train["hash_name"], train["date"]))
val_keys = set(zip(val["hash_name"], val["date"]))
test_keys = set(zip(test["hash_name"], test["date"]))

train_val_overlap = len(train_keys & val_keys)
train_test_overlap = len(train_keys & test_keys)
val_test_overlap = len(val_keys & test_keys)

print("Train/Val overlap: ", train_val_overlap)
print("Train/Test overlap:", train_test_overlap)
print("Val/Test overlap:  ", val_test_overlap)

assert train_val_overlap == 0, "ERROR: train and val overlap!"
assert train_test_overlap == 0, "ERROR: train and test overlap!"
assert val_test_overlap == 0, "ERROR: val and test overlap!"

print("OK: no row overlap.")


# ------------------------------------------------------------
# 4. Target distribution
# ------------------------------------------------------------

print("\n4) Target distribution")

describe_y("y_train", y_train)
describe_y("y_val", y_val)
describe_y("y_test", y_test)


# ------------------------------------------------------------
# 5. Prediction distribution
# ------------------------------------------------------------

print("\n5) Ridge prediction distribution")

train_pred = ridge.predict(X_train)

describe_pred("ridge train pred", train_pred)
describe_pred("ridge val pred", val_pred)
describe_pred("ridge test pred", test_pred)


# ------------------------------------------------------------
# 6. Ridge train/val/test metrics
# ------------------------------------------------------------

print("\n6) Ridge train/val/test metrics")

print(f"Ridge TRAIN MAE:  {mean_absolute_error(y_train, train_pred):.10f}")
print(f"Ridge VAL MAE:    {mean_absolute_error(y_val, val_pred):.10f}")
print(f"Ridge TEST MAE:   {mean_absolute_error(y_test, test_pred):.10f}")

print(f"Ridge TRAIN RMSE: {rmse(y_train, train_pred):.10f}")
print(f"Ridge VAL RMSE:   {rmse(y_val, val_pred):.10f}")
print(f"Ridge TEST RMSE:  {rmse(y_test, test_pred):.10f}")

print(f"Ridge TRAIN R2:   {r2_score(y_train, train_pred):.4f}")
print(f"Ridge VAL R2:     {r2_score(y_val, val_pred):.4f}")
print(f"Ridge TEST R2:    {r2_score(y_test, test_pred):.4f}")


# ------------------------------------------------------------
# 7. Correlation check for numeric features
# ------------------------------------------------------------

print("\n7) Correlation between numeric features and target")

corrs = X_train[num_cols].corrwith(y_train).abs().sort_values(ascending=False)
print(corrs)

very_high_corr = corrs[corrs > 0.95]

if len(very_high_corr) > 0:
    print("\nWARNING: Some numeric features have very high correlation with target:")
    print(very_high_corr)
else:
    print("OK: no numeric feature has suspiciously high correlation > 0.95.")


# ------------------------------------------------------------
# 8. Shuffle target test
# ------------------------------------------------------------

print("\n8) Shuffle target sanity check")

np.random.seed(42)
y_train_shuffled = np.random.permutation(y_train)

ridge_shuffled = clone(ridge)
ridge_shuffled.fit(X_train, y_train_shuffled)

val_pred_shuffled = ridge_shuffled.predict(X_val)
test_pred_shuffled = ridge_shuffled.predict(X_test)

print(f"Original Ridge VAL MAE:     {mean_absolute_error(y_val, val_pred):.10f}")
print(f"Shuffled target VAL MAE:    {mean_absolute_error(y_val, val_pred_shuffled):.10f}")

print(f"Original Ridge TEST MAE:    {mean_absolute_error(y_test, test_pred):.10f}")
print(f"Shuffled target TEST MAE:   {mean_absolute_error(y_test, test_pred_shuffled):.10f}")

print("Expected: shuffled target MAE should be much worse, ideally close to baseline.")


# ------------------------------------------------------------
# 9. Feature shuffle test
# ------------------------------------------------------------

print("\n9) Feature shuffle sanity check")

X_val_shuffled = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
y_val_reset = y_val.reset_index(drop=True)

val_pred_feature_shuffled = ridge.predict(X_val_shuffled)

print(f"Original Ridge VAL MAE:      {mean_absolute_error(y_val, val_pred):.10f}")
print(f"Shuffled features VAL MAE:   {mean_absolute_error(y_val_reset, val_pred_feature_shuffled):.10f}")

print("Expected: shuffled features MAE should be worse than original Ridge.")


# ------------------------------------------------------------
# 10. First 20 predictions
# ------------------------------------------------------------

print("\n10) First 20 test predictions")

prediction_check = pd.DataFrame({
    "hash_name": test["hash_name"].values,
    "date": test["date"].values,
    "y_true": y_test.values,
    "y_pred": test_pred,
    "abs_error": np.abs(y_test.values - test_pred)
})

print(prediction_check.head(20).to_string(index=False))


# ------------------------------------------------------------
# 11. Baseline improvement factor
# ------------------------------------------------------------

print("\n11) Baseline improvement factor")

baseline_val_mae = mean_absolute_error(y_val, val_base)
baseline_test_mae = mean_absolute_error(y_test, test_base)

ridge_val_mae = mean_absolute_error(y_val, val_pred)
ridge_test_mae = mean_absolute_error(y_test, test_pred)

print(f"VAL baseline MAE / Ridge MAE:  {baseline_val_mae / ridge_val_mae:.2f}x")
print(f"TEST baseline MAE / Ridge MAE: {baseline_test_mae / ridge_test_mae:.2f}x")


# ------------------------------------------------------------
# 12. Data leakage warning summary
# ------------------------------------------------------------

print("\n12) Quick interpretation guide")

print("""
Good signs:
- Target/future columns are not in X.
- Train/val/test overlap is 0.
- Shuffled target MAE is much worse than original Ridge.
- Shuffled features MAE is worse than original Ridge.
- No feature has correlation close to 1.0 with the target.

Bad signs:
- Shuffled target MAE is almost the same as original Ridge.
- A feature has correlation 0.95+ with the target.
- Train MAE is extremely low compared to val/test MAE.
- Train/val/test overlap is not 0.
""")

print("\nSanity checks finished.")