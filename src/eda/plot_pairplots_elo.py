import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Wczytanie danych
df = pd.read_csv("data/processed/atp_tennis_processed.csv")

# Filtrowanie tylko potrzebnych kolumn
df = df[["Elo_1","Elo_2","Elo_Hard_1","Elo_Hard_2","Elo_Grass_1","Elo_Grass_2"]]

# Opcjonalnie: wybierz jedną nawierzchnię (np. Hard)
df = df[df["surface"] == "Hard"]

# Pairplot
sns.pairplot(df, hue="is_player1_winner", diag_kind="kde", corner=True)
plt.suptitle("ELO na nawierzchni: Hard", y=1.02)
plt.tight_layout()
plt.show()
