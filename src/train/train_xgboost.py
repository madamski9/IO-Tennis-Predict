import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import os
import optuna

# wczytanie danych 
df = pd.read_csv("data/processed/atp_tennis_processed_test.csv")
df["Date"] = pd.to_datetime(df["Date"])

# usuniecie meczy bez roznic
df = df[
    (df["win_last_25_diff"] != 0) |
    (df["elo_grad_50_diff"] != 0) |
    (df["h2h_diff"] != 0)
]

surface_elo_cols = {
    "Hard": "Elo_Hard_diff",
    "Clay": "Elo_Clay_diff",
    "Grass": "Elo_Grass_diff",
}
df["surface_elo_diff"] = df.apply(lambda row: row.get(surface_elo_cols.get(row["Surface"], ""), 0.0), axis=1)
df["elo_x_form"] = df["Elo_diff"] * df["win_last_100_diff"]
df["elo_plus_form"] = df["Elo_diff"] + df["win_last_100_diff"]
df["elo_form_ratio"] = df["Elo_diff"] / (df["win_last_100_diff"] + 1e-5)

# === feature list ===
features = [
    "Elo_diff", "surface_elo_diff", "rank_diff", "pts_diff",
    "win_last_5_diff", "win_last_25_diff", "win_last_50_diff", "win_last_100_diff", "win_last_250_diff",
    "diff_N_games", "elo_grad_20_diff", "elo_grad_35_diff", "elo_grad_50_diff", "elo_grad_100_diff",
    "h2h_diff", "h2h_surface_diff", "elo_x_form", "elo_plus_form", "elo_form_ratio"
]

X = df[features]
y = df["is_player1_winner"]

train_df = df[df["Date"] < "2022-01-01"]
val_df = df[(df["Date"] >= "2022-01-01") & (df["Date"] < "2023-01-01")]
test_df = df[df["Date"] >= "2023-01-01"]

X_tr = train_df[features]
y_tr = train_df["is_player1_winner"]
X_val = val_df[features]
y_val = val_df["is_player1_winner"]
X_test = test_df[features]
y_test = test_df["is_player1_winner"]

# # Naive Bayes
# nb = GaussianNB()
# nb.fit(X_tr, y_tr)
# nb_preds = nb.predict(X_test)
# nb_acc = accuracy_score(y_test, nb_preds)
# print(f"Naive Bayes accuracy: {nb_acc:.4f}")

# # kNN
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_tr, y_tr)
# knn_preds = knn.predict(X_test)
# knn_acc = accuracy_score(y_test, knn_preds)
# print(f"kNN accuracy: {knn_acc:.4f}")

# # Decision Tree
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(X_tr, y_tr)
# dt_preds = dt.predict(X_test)
# dt_acc = accuracy_score(y_test, dt_preds)
# print(f"Decision Tree accuracy: {dt_acc:.4f}")

# optuna tuning dla XGBoost 
def objective(trial):
    param = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "booster": "gbtree",
        "tree_method": "hist",
        "eta": trial.suggest_float("eta", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "lambda": trial.suggest_float("lambda", 0.1, 10),
        "alpha": trial.suggest_float("alpha", 0.0, 1.0),
        "scale_pos_weight": y_tr.value_counts()[0] / y_tr.value_counts()[1],
        "seed": 42,
        "verbosity": 0,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X_tr, y_tr):
        X_train, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_train, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

        model = xgb.train(param, dtrain, num_boost_round=1000,
                          evals=[(dval, "val")], early_stopping_rounds=30, verbose_eval=False)
        preds = model.predict(dval)
        aucs.append(roc_auc_score(y_val_fold, preds))
    return np.mean(aucs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best XGB params:", study.best_params)

# === budowa modeli ===
best_xgb_params = study.best_params
best_xgb_params.update({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "scale_pos_weight": y_tr.value_counts()[0] / y_tr.value_counts()[1],
    "seed": 42,
    "verbosity": 0,
})

xgb_model = XGBClassifier(**best_xgb_params, use_label_encoder=False)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
base_models = [xgb_model, rf_model]
meta_model = LogisticRegression(max_iter=1000)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

meta_features_train = np.zeros((X_tr.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    meta_feature = np.zeros(X_tr.shape[0])
    for train_idx, val_idx in skf.split(X_tr, y_tr):
        X_train_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_train_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        cloned_model = clone(model)
        cloned_model.fit(X_train_fold, y_train_fold)
        preds = cloned_model.predict_proba(X_val_fold)[:, 1]
        meta_feature[val_idx] = preds
    meta_features_train[:, i] = meta_feature

meta_model.fit(meta_features_train, y_tr)

# === predykcje bazowych modeli na zbiorze testowym ===
meta_features_test = np.zeros((X_test.shape[0], len(base_models)))
for i, model in enumerate(base_models):
    model.fit(X_tr, y_tr)  
    preds_test = model.predict_proba(X_test)[:, 1]
    meta_features_test[:, i] = preds_test

# === predykcje finalne meta modelu na podstawie meta cech ===
final_preds_proba = meta_model.predict_proba(meta_features_test)[:, 1]
final_preds = (final_preds_proba > 0.5).astype(int)

acc = accuracy_score(y_test, final_preds)
auc = roc_auc_score(y_test, final_preds_proba)

print(f"Stacking Test Accuracy: {acc:.4f}")
print(f"Stacking Test AUC: {auc:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(meta_model, "models/meta_model_logreg.joblib")
joblib.dump(xgb_model, "models/xgb_base_model.joblib")
joblib.dump(rf_model, "models/rf_base_model.joblib")
print("Models saved to models/")

# === feature importance dla XGBoost ===
xgb_model.fit(X_tr, y_tr)
plt.figure(figsize=(10,6))
xgb.plot_importance(xgb_model, importance_type="gain", max_num_features=15)
plt.tight_layout()
os.makedirs("images/decision_tree/", exist_ok=True)
plt.savefig("images/decision_tree/xgb_feature_importance.png")
print("Feature importance plot saved.")

# === wizualizacja jednego z drzew XGBoost ===
plt.figure(figsize=(20, 10))
xgb.plot_tree(xgb_model, num_trees=0, rankdir='LR', ax=plt.gca())
plt.tight_layout()
plt.savefig("images/decision_tree/xgb_tree_0.png")
