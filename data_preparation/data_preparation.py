import pandas as pd 
import numpy as np
import re 
#File path
file = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_collection\master_daily_2023-09-01_to_2025-09-01.csv"
#Read the file through pandas and set it as a variable
df = pd.read_csv(file)

#Checking the first five rows, columns and their types
print(df.head())
print(df.columns.tolist())
print(df.dtypes)

#Convert the date from object to datetime
df["date"] = pd.to_datetime(df["date"], errors="coerce")
#Convert sell_listings to integer from float
df["sell_listings"] = pd.to_numeric(df.get("sell_listings"), errors="coerce")

#Check the updated columns
print(df.columns.tolist())
print(df.dtypes)

#Convert sell_price_text into floats
def parse_price_text_eur(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("€", "").replace("--", "00")
    s = re.sub(r"[^0-9,\.]", "", s)
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

df["sell_price_eur_snapshot"] = df["sell_price_text"].apply(parse_price_text_eur)

#Sanity check
print("\nDate range:", df["date"].min(), "->", df["date"].max())
print("Unique hash_name:", df["hash_name"].nunique())
print("Duplicates (hash_name,date):", df.duplicated(["hash_name", "date"]).sum())

print("\nCurrency distribution:")
print(df["currency"].value_counts(dropna=False))

# Drop duplicates and sort
df = df.sort_values(["hash_name", "date"])
df = df.drop_duplicates(["hash_name", "date"], keep="last")

# Save DS-ready file
#out_file = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_preparation\ds_ready_master.csv"
#df.to_csv(out_file, index=False)
#print("\nWrote:", out_file)

