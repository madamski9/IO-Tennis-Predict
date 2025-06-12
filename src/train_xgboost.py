import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import optuna

# === 1. Wczytaj dane ===
df = pd.read_csv("data/processed/atp_tennis_processed.csv")
df["Date"] = pd.to_datetime(df["Date"])

# === 2. Usuń mecze bez różnic ===
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

df["surface_elo_diff"] = df.apply(lambda row: row.get(surface_elo_cols.get(row["Surface"], ""), 0.0), axis=1)
df["elo_x_form"] = df["Elo_diff"] * df["win_last_100_diff"]
df["elo_plus_form"] = df["Elo_diff"] + df["win_last_100_diff"]
df["elo_form_ratio"] = df["Elo_diff"] / (df["win_last_100_diff"] + 1e-5)

# === 5. Feature list ===
features = [
    "Elo_diff", "surface_elo_diff", "rank_diff", "pts_diff",
    "win_last_5_diff", "win_last_25_diff", "win_last_50_diff", "win_last_100_diff", "win_last_250_diff",
    "diff_N_games", "elo_grad_20_diff", "elo_grad_35_diff", "elo_grad_50_diff", "elo_grad_100_diff",
    "h2h_diff", "h2h_surface_diff", "elo_x_form", "elo_plus_form", "elo_form_ratio"
]

X = df[features]
y = df["is_player1_winner"]

# === 6. Podział czasowy ===
train_df = df[df["Date"] < "2022-01-01"]
val_df = df[(df["Date"] >= "2022-01-01") & (df["Date"] < "2023-01-01")]
test_df = df[df["Date"] >= "2023-01-01"]

X_tr = train_df[features]
y_tr = train_df["is_player1_winner"]
X_val = val_df[features]
y_val = val_df["is_player1_winner"]
X_test = test_df[features]
y_test = test_df["is_player1_winner"]

# === 7. Optuna tuning ===
def objective(trial):
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "tree_method": "hist",  # szybsze trenowanie
        "eta": trial.suggest_float("eta", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 0.1, 10),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "scale_pos_weight": y_tr.value_counts()[0] / y_tr.value_counts()[1],
        "seed": 42,
    }

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        params=param,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    y_val_pred = model.predict(dval)
    auc = roc_auc_score(y_val, y_val_pred)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)
print("Best AUC on validation:", study.best_value)

# === 8. Finalny model z najlepszymi parametrami ===
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "scale_pos_weight": y_tr.value_counts()[0] / y_tr.value_counts()[1],
    "seed": 42
})

dtrain = xgb.DMatrix(X_tr, label=y_tr)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

final_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=[(dval, "val")],
    early_stopping_rounds=30,
    verbose_eval=50
)

# === 9. Ewaluacja ===
y_prob = final_model.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")

# === 10. Zapis modelu ===
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/xgb_model_optuna.joblib")
print("Model saved to models/xgb_model_optuna.joblib")

# === 11. Feature importance ===
os.makedirs("images/decision_tree/", exist_ok=True)
xgb.plot_importance(final_model, importance_type="gain", max_num_features=15)
plt.tight_layout()
plt.savefig("images/decision_tree/xgb_feature_importance.png")
print("Feature importance plot saved.")
