def plot_surface_elos():
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    surfaces = ["Hard", "Clay", "Grass"]
    os.makedirs("images/decision_tree/", exist_ok=True)

    for surface in surfaces:
        path = f"data/processed/elo_history_{surface.lower()}.csv"
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])

        final_elos = df.groupby("Player")["Elo"].last().sort_values(ascending=False)
        top5_players = final_elos.head(5).index

        plt.figure(figsize=(14, 8))

        for player in df["Player"].unique():
            player_df = df[df["Player"] == player]

            if player in top5_players:
                plt.plot(player_df["Date"], player_df["Elo"], label=player, linewidth=2)
            else:
                plt.plot(player_df["Date"], player_df["Elo"], color="gray", alpha=0.3, linewidth=0.5)

        plt.title(f"ELO rating over time ({surface})", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("ELO")
        plt.legend(title="Top 5 Players", loc="upper left")
        plt.tight_layout()
        plt.savefig(f"images/elo_over_time_{surface.lower()}.png")
        plt.close()