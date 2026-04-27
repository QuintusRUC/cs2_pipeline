import pandas as pd
import matplotlib.pyplot as plt

FILE = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_preparation\ds_ready_master.csv"
df = pd.read_csv(FILE)

# Continuous features
continuous = ["median_price", "volume", "sell_listings", "sell_price_eur_snapshot"]

cont = []
n = len(df)

for col in continuous:
    s = pd.to_numeric(df[col], errors="coerce")
    s_clean = s.dropna()
    card = s_clean.nunique()

    cont.append({
        "Feature": col,
        "Count": s.notna().sum(),
        "% Miss.": round(s.isna().mean() * 100, 2),
        "Card.": card,
        "Min.": s.min(),
        "1st Qrt.": s.quantile(0.25),
        "Mean": s.mean(),
        "Median": s.median(),
        "3rd Qrt.": s.quantile(0.75),
        "Max.": s.max(),
        "Std. Dev.": s.std(),
    })

    # Show plot inside program
    plt.figure(figsize=(8, 5))

    if card >= 10:
        plt.hist(s_clean, bins=20, edgecolor="black")
        plt.title(f"Histogram of {col} (Cardinality = {card})")
        plt.xlabel(col)
        plt.ylabel("Frequency")
    else:
        s_clean.value_counts().sort_index().plot(kind="bar", edgecolor="black")
        plt.title(f"Bar Plot of {col} (Cardinality = {card})")
        plt.xlabel(col)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

out_file_cont = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\dq_continuous.csv"
dq_cont = pd.DataFrame(cont)
dq_cont.to_csv(out_file_cont, index=False)
print("Wrote dq_continuous.csv")


# Categorical features
categorical = ["weapon", "wear", "rarity", "is_stattrak", "currency"]

cat = []

for col in categorical:
    s = df[col].astype("string")
    miss = (s.isna() | (s.str.strip() == "")).sum()
    s_clean = s[~s.isna() & (s.str.strip() != "")]
    vc = s_clean.value_counts()
    card = s_clean.nunique()

    mode = vc.index[0] if len(vc) else ""
    mode_freq = int(vc.iloc[0]) if len(vc) else 0
    mode_pct = round(mode_freq / n * 100, 2) if n else 0

    mode2 = vc.index[1] if len(vc) > 1 else ""
    mode2_freq = int(vc.iloc[1]) if len(vc) > 1 else 0
    mode2_pct = round(mode2_freq / n * 100, 2) if n else 0

    cat.append({
        "Feature": col,
        "Count": n - miss,
        "% Miss.": round(miss / n * 100, 2),
        "Card.": card,
        "Mode": str(mode),
        "Mode Freq.": mode_freq,
        "Mode %": mode_pct,
        "2nd Mode": str(mode2),
        "2nd Mode Freq.": mode2_freq,
        "2nd Mode %": mode2_pct,
    })

    # Always show bar plot for categorical features
    plt.figure(figsize=(8, 5))
    vc.plot(kind="bar", edgecolor="black")
    plt.title(f"Bar Plot of {col} (Cardinality = {card})")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

out_file_cat = r"C:\Users\nikol\OneDrive\Skrivebord\cs2\cs2_pipeline\data_exploration\dq_categorical.csv"
dq_cat = pd.DataFrame(cat)
dq_cat.to_csv(out_file_cat, index=False)
print("Wrote dq_categorical.csv")


