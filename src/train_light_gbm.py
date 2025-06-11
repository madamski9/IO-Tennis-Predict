import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation

df = pd.read_csv("data/processed/atp_tennis_processed.csv")
df["Date"] = pd.to_datetime(df["Date"])

surface_elo_cols = {
    "Hard": "Elo_Hard_diff",
    "Clay": "Elo_Clay_diff",
    "Grass": "Elo_Grass_diff",
}

def get_surface_elo_diff(row):
    surface = row["Surface"]
    return row.get(surface_elo_cols.get(surface, ""), 0.0)

df["surface_elo_diff"] = df.apply(get_surface_elo_diff, axis=1)

features = [
    "Elo_diff", "surface_elo_diff", "rank_diff", "pts_diff",
    "win_last_5_diff", "win_last_25_diff", "win_last_50_diff", "win_last_100_diff", "win_last_250_diff",
    "diff_N_games", "elo_grad_20_diff", "elo_grad_35_diff", "elo_grad_50_diff", "elo_grad_100_diff",
    "h2h_diff", "h2h_surface_diff"
]

train_df = df[df["Date"] < "2022-01-01"]
val_df = df[(df["Date"] >= "2022-01-01") & (df["Date"] < "2023-01-01")]
test_df = df[df["Date"] >= "2023-01-01"]

X_tr = train_df[features]
y_tr = train_df["is_player1_winner"]
X_val = val_df[features]
y_val = val_df["is_player1_winner"]
X_test = test_df[features]
y_test = test_df["is_player1_winner"]

model = LGBMClassifier(
    objective='binary',
    learning_rate=0.01,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=5000,
    random_state=42
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric='binary_logloss',
    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(100)]
)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nAccuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")

lgb.plot_importance(model, max_num_features=10, importance_type='gain')
plt.tight_layout()
plt.savefig("images/lgb_feature_importance.png")
print("Feature importance plot saved.")
