import pandas as pd

# ----------------------------
# INPUT / OUTPUT
# ----------------------------
INFILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready.csv"
OUTFILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready_clean_final.csv"

# ----------------------------
# Helpers: Unknown table by wear + TOTAL row
# ----------------------------
def unknown_by_wear_with_total(df, rarity_col):
    tmp = df[rarity_col].fillna("Unknown")

    out = (
        df.assign(_rar=tmp)
          .groupby("wear")
          .agg(
              rows=("_rar", "size"),
              unknown_rows=("_rar", lambda s: (s == "Unknown").sum())
          )
          .reset_index()
    )
    out["unknown_pct"] = (out["unknown_rows"] / out["rows"] * 100).round(2)

    # TOTAL row
    total_rows = int(out["rows"].sum())
    total_unknown = int(out["unknown_rows"].sum())
    total_pct = round((total_unknown / total_rows * 100), 2) if total_rows > 0 else 0.0

    total_row = pd.DataFrame([{
        "wear": "Total",
        "rows": total_rows,
        "unknown_rows": total_unknown,
        "unknown_pct": total_pct
    }])

    # Sort by unknown_pct desc but keep Total at bottom
    out = out.sort_values("unknown_pct", ascending=False)
    out = pd.concat([out, total_row], ignore_index=True)
    return out

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(INFILE)

required_cols = ["weapon", "skin_name", "wear", "rarity", "hash_name"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in input: {missing}")

print("Loaded rows:", len(df), "| unique hash_name:", df["hash_name"].nunique())

# ----------------------------
# BEFORE: Unknown distribution table
# ----------------------------
print("\n=== UNKNOWN BY WEAR (BEFORE) ===")
before_table = unknown_by_wear_with_total(df, "rarity")
print(before_table.to_string(index=False))

unknown_rows_before = (df["rarity"].fillna("Unknown") == "Unknown").sum()
print("\nUnknown rarity rows (before):", unknown_rows_before)

# ----------------------------
# 1) Impute rarity by base skin (weapon, skin_name)
# ----------------------------
df["rarity_fix"] = df["rarity"].replace("Unknown", pd.NA)

def fill_mode(series):
    non_missing = series.dropna()
    if len(non_missing) == 0:
        return series  # leave missing for now
    return series.fillna(non_missing.mode().iloc[0])

df["rarity_imputed"] = df.groupby(["weapon", "skin_name"])["rarity_fix"].transform(fill_mode)
df["rarity_imputed"] = df["rarity_imputed"].fillna("Unknown")

# ----------------------------
# AFTER IMPUTATION: Unknown distribution table
# ----------------------------
print("\n=== UNKNOWN BY WEAR (AFTER IMPUTATION) ===")
after_imp_table = unknown_by_wear_with_total(df, "rarity_imputed")
print(after_imp_table.to_string(index=False))

unknown_rows_after_imp = (df["rarity_imputed"] == "Unknown").sum()
print("\nUnknown rarity rows (after imputation):", unknown_rows_after_imp)

# ----------------------------
# 2) Find base skins still Unknown AFTER imputation
# ----------------------------
unknown_bases = df.loc[df["rarity_imputed"] == "Unknown", ["weapon", "skin_name"]].drop_duplicates()

print("\n=== BASE SKINS STILL UNKNOWN (AFTER IMPUTATION) ===")
print("Count:", len(unknown_bases))
if len(unknown_bases) > 0:
    print(unknown_bases.to_string(index=False))

# ----------------------------
# 3) Drop all rows belonging to those base skins
# ----------------------------
if len(unknown_bases) > 0:
    df2 = df.merge(unknown_bases.assign(_drop=1), on=["weapon", "skin_name"], how="left")
    df2 = df2[df2["_drop"].isna()].drop(columns=["_drop"])
else:
    df2 = df.copy()

# Replace rarity with cleaned rarity
df2["rarity"] = df2["rarity_imputed"]

# Cleanup helper cols
df2 = df2.drop(columns=[c for c in ["rarity_fix", "rarity_imputed"] if c in df2.columns])

# ----------------------------
# FINAL CHECK: Unknown must be 0
# ----------------------------
unknown_final = (df2["rarity"].fillna("Unknown") == "Unknown").sum()
print("\n=== FINAL CHECK ===")
print("Rows before:", len(df), "after:", len(df2))
print("Unique hash_name before:", df["hash_name"].nunique(), "after:", df2["hash_name"].nunique())
print("Unknown rarity rows final:", unknown_final)

if unknown_final != 0:
    raise ValueError("Still have Unknown rarity rows after cleaning. Check logic or input data.")

# ----------------------------
# Write ONE final output CSV
# ----------------------------
df2.to_csv(OUTFILE, index=False)
print("\nWrote cleaned file:", OUTFILE)