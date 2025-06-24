import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/processed/atp_tennis_processed_test.csv")
features = ["Elo_1","Elo_2","Elo_Hard_1","Elo_Hard_2","Elo_Grass_1","Elo_Grass_2", "Elo_Clay_1", "Elo_Clay_2", "is_player1_winner"]

os.makedirs("images/pairplots", exist_ok=True)
sns.pairplot(df[features], hue="is_player1_winner", diag_kind="kde", plot_kws={'alpha': 0.6, 's': 30})
plt.suptitle("Pairplot: Elo points on specific surface vs Match Outcome", y=1.02)
plt.tight_layout()
plt.savefig("images/pairplots/elo_surface_pairplot.png")
