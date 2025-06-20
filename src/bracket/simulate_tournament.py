import pandas as pd
import numpy as np
import joblib
import re

# === Normalizacja i dopasowywanie nazw ===

def name_to_token_set(name):
    """Zamienia imię/nazwisko na zbiór słów: usuwa kropki, myślniki, zamienia na małe litery."""
    name = re.sub(r"[\.\-]", " ", name).lower()
    tokens = set(name.strip().split())
    return tokens

def find_best_player_match(name, player_list):
    """Dopasowuje nazwisko z drabinki do nazwiska z bazy danych niezależnie od kolejności słów."""
    name_tokens = name_to_token_set(name)
    for candidate in player_list:
        candidate_tokens = name_to_token_set(candidate)
        if name_tokens.issubset(candidate_tokens) or candidate_tokens.issubset(name_tokens):
            return candidate
    return None

# === Wczytaj modele ===
xgb_model = joblib.load("models/xgb_base_model.joblib")
rf_model = joblib.load("models/rf_base_model.joblib")
meta_model = joblib.load("models/meta_model_logreg.joblib")

# === Wczytaj bazę danych ===
players_data = pd.read_csv("data/processed/atp_players_features_latest.csv")

# Lista cech wymaganych przez modele
features = [
    "Elo_diff", "surface_elo_diff", "rank_diff", "pts_diff",
    "win_last_5_diff", "win_last_25_diff", "win_last_50_diff", "win_last_100_diff", "win_last_250_diff",
    "diff_N_games", "elo_grad_20_diff", "elo_grad_35_diff", "elo_grad_50_diff", "elo_grad_100_diff",
    "h2h_diff", "h2h_surface_diff", "elo_x_form", "elo_plus_form", "elo_form_ratio"
]

# === Generowanie cech pojedynku ===

def get_player_row(name):
    matched_name = find_best_player_match(name, players_data["Player"].tolist())
    if matched_name:
        return players_data[players_data["Player"] == matched_name].iloc[0]
    else:
        return None

def generate_match_features(p1_raw, p2_raw, surface="Clay"):
    f1 = get_player_row(p1_raw)
    f2 = get_player_row(p2_raw)

    if f1 is None or f2 is None:
        print(f"Brak danych dla gracza: {p1_raw if f1 is None else p2_raw}")
        return None

    f1 = f1[f1.index.difference(["Player"])]
    f2 = f2[f2.index.difference(["Player"])]
    f1 = pd.to_numeric(f1, errors="coerce")
    f2 = pd.to_numeric(f2, errors="coerce")

    diff = f1 - f2

    surface_elo = {
        "Hard": diff.get("Elo_Hard", 0),
        "Clay": diff.get("Elo_Clay", 0),
        "Grass": diff.get("Elo_Grass", 0),
    }.get(surface, 0.0)

    elo_x_form = diff.get("Elo", 0) * diff.get("win_last_100_diff", 0)
    elo_plus_form = diff.get("Elo", 0) + diff.get("win_last_100_diff", 0)
    elo_form_ratio = diff.get("Elo", 0) / (diff.get("win_last_100_diff", 1e-5) + 1e-5)

    features_dict = {
        "surface_elo_diff": surface_elo,
        "elo_x_form": elo_x_form,
        "elo_plus_form": elo_plus_form,
        "elo_form_ratio": elo_form_ratio,
    }

    for feat in features:
        if feat not in features_dict:
            val = diff.get(feat, 0)
            features_dict[feat] = val if not isinstance(val, pd.Series) else val.iloc[0]

    df = pd.DataFrame([features_dict])
    df = df.apply(pd.to_numeric, errors='coerce')
    return df[features]

# features = generate_match_features("Sinner J.", "Gasquet R.")
# print(features)
# prediction = xgb_model.predict(features)
# print("Predykcja modelu:", prediction)

# === Wczytaj drabinkę turniejową ===
bracket_df = pd.read_csv("data/brackets/draw_roland_garros_2025_64.csv")
bracket_df.columns = ["Round", "Player_1", "Player_2"]
bracket_df_original = bracket_df.copy()

bracket_df["Round"] = bracket_df["Round"].astype(str).str.strip()
bracket_df = bracket_df[bracket_df["Round"].str.isdigit()]
bracket_df["Round"] = bracket_df["Round"].astype(int)

results = []

# === Główna pętla predykcyjna przez wszystkie rundy ===
current_round_num = bracket_df["Round"].min()

while True:
    current_round = bracket_df[bracket_df["Round"] == current_round_num]
    winners = []

    if current_round.empty:
        break

    for idx, row in current_round.iterrows():
        p1 = row["Player_1"]
        p2 = row["Player_2"]

        if pd.isna(p1) or pd.isna(p2):
            print(f"Pominięto mecz z brakującym zawodnikiem: {p1} vs {p2}")
            continue

        match_features = generate_match_features(p1, p2)
        if match_features is None:
            print(f"Pominięto mecz: {p1} vs {p2} (brak cech)")
            continue

        xgb_pred = xgb_model.predict_proba(match_features)[:, 1]
        rf_pred = rf_model.predict_proba(match_features)[:, 1]

        meta_input = np.column_stack([xgb_pred, rf_pred])
        final_pred = meta_model.predict(meta_input)[0]

        winner = p1 if final_pred == 1 else p2
        results.append({
            "Round": current_round_num,
            "Player_1": p1,
            "Player_2": p2,
            "Pred_Winner": winner
        })
        winners.append(winner)

    # Twórz kolejną rundę
    if len(winners) >= 2 and len(winners) % 2 == 0:
        next_round_df = pd.DataFrame({
            "Round": [current_round_num + 1] * (len(winners) // 2),
            "Player_1": winners[::2],
            "Player_2": winners[1::2]
        })
        bracket_df = pd.concat([bracket_df, next_round_df], ignore_index=True)
        bracket_df_original = pd.concat([bracket_df_original, next_round_df[["Round", "Player_1", "Player_2"]]], ignore_index=True)
        current_round_num += 1
    else:
        print(f"Nie utworzono kolejnej rundy po rundzie {current_round_num} — liczba zwycięzców: {len(winners)}")
        break

# === Zapisz wyniki ===
results_df = pd.DataFrame(results)
results_df.to_csv("data/predict_tourney/predicted_bracket.csv", index=False)
print("Zapisano: data/predict_tourney/predicted_bracket.csv")
