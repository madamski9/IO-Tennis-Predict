import pandas as pd

df = pd.read_csv('data/raw/atp_matches_till_2022.csv')

# Konwersja liczbowej daty YYYYMMDD na datetime
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')

# Filtrujemy od roku 2000-01-01
df_filtered = df[df['tourney_date'] >= pd.Timestamp('2000-01-01')]

# Zapisz z powrotem
df_filtered.to_csv('data/raw/atp_matches_till_2022.csv', index=False)
