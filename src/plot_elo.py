import pandas as pd
import matplotlib.pyplot as plt

elo_df = pd.read_csv("data/processed/elo_history.csv")
elo_df["Date"] = pd.to_datetime(elo_df["Date"])

final_elos = elo_df.groupby("Player")["Elo"].last().sort_values(ascending=False)
top5_players = final_elos.head(5).index

plt.figure(figsize=(14, 8))

for player in elo_df["Player"].unique():
    player_df = elo_df[elo_df["Player"] == player]
    
    if player in top5_players:
        plt.plot(player_df["Date"], player_df["Elo"], label=player, linewidth=2)
    else:
        plt.plot(player_df["Date"], player_df["Elo"], color="gray", alpha=0.3, linewidth=0.5)

plt.title("ELO rating over time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("ELO")
plt.legend(title="Top 5 Players", loc="upper left")
plt.tight_layout()

plt.savefig("images/elo_over_time.png")
