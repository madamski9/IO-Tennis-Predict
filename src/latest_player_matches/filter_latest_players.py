import pandas as pd

df = pd.read_csv("data/processed/atp_players_features_latest.csv")

df["Elo"] = pd.to_numeric(df["Elo"], errors='coerce')

df_sorted = df.sort_values(by="Elo", ascending=False)
df_unique = df_sorted.drop_duplicates(subset=["Player"], keep="first")

print(f"Liczba rekordów przed usuwaniem duplikatów: {len(df)}")
print(f"Liczba rekordów po usuwaniu duplikatów: {len(df_unique)}")

df_unique.to_csv("data/processed/atp_players_features_latest.csv", index=False)
