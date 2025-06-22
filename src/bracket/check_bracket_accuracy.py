import pandas as pd

# Wczytaj pliki
real = pd.read_csv("data/predict_tourney/real_bracket.csv")
pred = pd.read_csv("data/predict_tourney/predicted_bracket.csv")

# Dopasuj kolumny i rundy
merged = pd.merge(
    pred,
    real,
    how="inner",
    on=["Round", "Player_1", "Player_2"]
)

accuracy = len(merged) / len(real)
print(f"Accuracy w każdej rundzie: {accuracy:.4f}")
