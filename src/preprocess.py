import pandas as pd
from calculateElo import calculate_elo

df = pd.read_csv("data/raw/atp_tennis.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df[df["Date"] >= "2015-01-01"]

df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
df["pts_diff"] = df["Pts_1"] - df["Pts_2"]
df["is_player1_winner"] = (df["Winner"] == df["Player_1"]).astype(int)

df["Surface"] = df["Surface"].fillna("Unknown")
df = pd.get_dummies(df, columns=["Surface"], drop_first=True)

df, elo_df = calculate_elo(df)

df.to_csv("data/processed/atp_tennis_2015.csv", index=False)
elo_df.to_csv("data/processed/elo_history.csv", index=False)

