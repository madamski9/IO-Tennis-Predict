import pandas as pd
from calculate_surface_elo import calculate_surface_elo

df = pd.read_csv("data/processed/atp_tennis_processed.csv")
df = calculate_surface_elo(df)
df.to_csv("data/processed/atp_tennis_processed_test.csv", index=False)
