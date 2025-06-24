import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/processed/atp_tennis_processed.csv")

features = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Elo_1", "Elo_2", "is_player1_winner"]

os.makedirs("images/pairplots", exist_ok=True)
sns.pairplot(df[features], hue="is_player1_winner", diag_kind="kde", plot_kws={'alpha': 0.6, 's': 30})
plt.suptitle("Pairplot: Rank & Points/Elo vs Match Outcome", y=1.02)
plt.tight_layout()
plt.savefig("images/pairplots/rank_points_pairplot.png")
