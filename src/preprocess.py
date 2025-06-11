import pandas as pd
from calculate_elo import calculate_elo
from calculate_surface_elo import calculate_surface_elo, build_surface_elo_history
from plot_surface_elo import plot_surface_elos
from feature_engineering import add_h2h_and_surface_winrate
from custom_features import add_custom_features 

df = pd.read_csv("data/raw/atp_tennis.csv")

df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
df["pts_diff"] = df["Pts_1"] - df["Pts_2"]
df["is_player1_winner"] = (df["Winner"] == df["Player_1"]).astype(int)

df, elo_df = calculate_elo(df)
df = calculate_surface_elo(df)
build_surface_elo_history(df)
plot_surface_elos()
df = add_h2h_and_surface_winrate(df)
df = add_custom_features(df)

df.to_csv("data/processed/atp_tennis_processed.csv", index=False)
elo_df.to_csv("data/processed/elo_history.csv", index=False)

