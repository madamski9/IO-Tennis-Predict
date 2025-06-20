import pandas as pd

df = pd.read_csv("data/processed/atp_players_features_latest.csv")

# Konwersja Elo na float (na pewno)
df["Elo"] = pd.to_numeric(df["Elo"], errors='coerce')

# Sortowanie malejąco po Elo
df_sorted = df.sort_values(by="Elo", ascending=False)

# Usuwanie duplikatów - zostawiamy gracza z najwyższym Elo
df_unique = df_sorted.drop_duplicates(subset=["Player"], keep="first")

print(f"Liczba rekordów przed usuwaniem duplikatów: {len(df)}")
print(f"Liczba rekordów po usuwaniu duplikatów: {len(df_unique)}")

df_unique.to_csv("data/processed/atp_players_features_latest.csv", index=False)
