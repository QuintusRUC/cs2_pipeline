import pandas as pd

FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_preparation\ds_ready_master.csv"
OUT  = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\model_ready.csv"

df = pd.read_csv(FILE)

# 1) Drop leakage + constant feature
drop_cols = ["sell_listings", "sell_price_text", "sell_price_eur_snapshot", "currency"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 2) Parse / basic validity
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

df["median_price"] = pd.to_numeric(df["median_price"], errors="coerce")
df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

df = df[df["median_price"].notna() & (df["median_price"] > 0)]
df = df[df["volume"].notna() & (df["volume"] >= 0)]

# 3) Fix rarity missing per item (static feature)
def fill_group_mode(series):
    non_missing = series.dropna()
    if len(non_missing) == 0:
        return series.fillna("Unknown")
    mode_vals = non_missing.mode()
    if len(mode_vals) == 0:
        return series.fillna("Unknown")
    return series.fillna(mode_vals.iloc[0])

print(f"Missing rarity before: {df['rarity'].isna().mean()*100:.2f}%")
df["rarity"] = df.groupby("hash_name")["rarity"].transform(fill_group_mode)
print(f"Missing rarity after: {df['rarity'].isna().mean()*100:.2f}%")
print(df["rarity"].value_counts().head(10))

# 4) Sort + deduplicate
df = df.sort_values(["hash_name", "date"])
df = df.drop_duplicates(["hash_name", "date"], keep="last")

# 5) Save
df.to_csv(OUT, index=False)
print("Wrote:", OUT)
print("Rows:", len(df), "Items:", df["hash_name"].nunique())