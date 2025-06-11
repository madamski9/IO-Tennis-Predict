import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# === 1. Wczytaj dane ===
df = pd.read_csv("data/processed/atp_tennis_processed.csv")
df["Date"] = pd.to_datetime(df["Date"])

# === 2. usuń puste mecze ===
df = df[
    (df["win_last_25_diff"] != 0) |
    (df["elo_grad_50_diff"] != 0) |
    (df["h2h_diff"] != 0)
]

# === 3. Dodaj surface_elo_diff ===
surface_elo_cols = {
    "Hard": "Elo_Hard_diff",
    "Clay": "Elo_Clay_diff",
    "Grass": "Elo_Grass_diff",
}

def get_surface_elo_diff(row):
    surface = row["Surface"]
    return row.get(surface_elo_cols.get(surface, ""), 0.0)

df["surface_elo_diff"] = df.apply(get_surface_elo_diff, axis=1)

# === 4. Dodaj log_odds_diff (jeśli kursy są) ===
if "Odd_1" in df.columns and "Odd_2" in df.columns:
    df["log_odds_diff"] = np.log(df["Odd_2"]) - np.log(df["Odd_1"])
    has_odds = True
else:
    has_odds = False

# === 5. Przygotuj cechy i target ===
features = [
    "Elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "pts_diff",
    "win_last_5_diff",
    "win_last_25_diff",
    "win_last_50_diff",
    "win_last_100_diff",
    "win_last_250_diff",
    "diff_N_games",
    "elo_grad_20_diff",
    "elo_grad_35_diff",
    "elo_grad_50_diff",
    "elo_grad_100_diff",
    "h2h_diff",
    "h2h_surface_diff"
]

X = df[features]
y = df["is_player1_winner"]

# === 6. Podział na zbiór treningowy, walidacyjny i testowy ===
train_df = df[df["Date"] < "2022-01-01"]
val_df = df[(df["Date"] >= "2022-01-01") & (df["Date"] < "2023-01-01")]
test_df = df[df["Date"] >= "2023-01-01"]

X_tr = train_df[features]
y_tr = train_df["is_player1_winner"]
X_val = val_df[features]
y_val = val_df["is_player1_winner"]
X_test = test_df[features]
y_test = test_df["is_player1_winner"]

# === 7. Konwersja do DMatrix ===
dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

# === 8. Parametry XGBoost ===
scale_pos_weight = y_tr.value_counts()[0] / y_tr.value_counts()[1]

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.03,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight": scale_pos_weight,
    "seed": 42
}

# === 9. Trenowanie modelu ===
evals = [(dtrain, "train"), (dval, "val")]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=30,
    verbose_eval=50
)

# === 10. Ewaluacja ===
y_prob = model.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nAccuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")

# === 11. Zapis modelu ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.joblib")
print("Model saved to models/xgb_model.joblib")

# === 12. Feature importance ===
os.makedirs("images", exist_ok=True)
xgb.plot_importance(model, importance_type="gain", max_num_features=15)
plt.tight_layout()
plt.savefig("images/xgb_feature_importance.png")
print("Feature importance plot saved.")
