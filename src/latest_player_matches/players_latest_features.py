import pandas as pd

# Wczytaj dane
matches = pd.read_csv("data/processed/atp_tennis_processed_test.csv")
matches["Date"] = pd.to_datetime(matches["Date"])

surfaces = ["Hard", "Clay", "Grass"]
players = set(matches["Player_1"]).union(set(matches["Player_2"]))

# Lista cech z ostatniego meczu, które chcesz dodać (przykładowo)
extra_features = [
    "Elo_diff", "surface_winrate", "rank_diff", "pts_diff",
    "win_last_5_diff", "win_last_25_diff", "win_last_50_diff", "win_last_100_diff", "win_last_250_diff",
    "diff_N_games", "elo_grad_20_diff", "elo_grad_35_diff", "elo_grad_50_diff", "elo_grad_100_diff",
    "h2h_diff", "h2h_surface_diff", "elo_plus_form", "elo_form_ratio", "elo_x_form"
]

player_rows = []

for player in players:
    player_data = {"Player": player}
    has_any_elo = False

    # Pobieranie Elo po nawierzchniach (tak jak masz)
    for surface in surfaces:
        surface_matches = matches[
            ((matches["Player_1"] == player) | (matches["Player_2"] == player)) &
            (matches["Surface"] == surface)
        ].sort_values("Date", ascending=False)

        elo = None
        for _, row in surface_matches.iterrows():
            if row["Player_1"] == player:
                elo = row.get(f"Elo_{surface}_1", None)
            elif row["Player_2"] == player:
                elo = row.get(f"Elo_{surface}_2", None)
            if elo not in [None, -1.0]:
                has_any_elo = True
                break

        player_data[f"Elo_{surface}"] = elo if elo not in [-1.0] else None

    # Pobieranie ostatniego meczu gracza (dowolna nawierzchnia)
    all_matches = matches[
        (matches["Player_1"] == player) | (matches["Player_2"] == player)
    ].sort_values("Date", ascending=False)

    if not all_matches.empty:
        last = all_matches.iloc[0]
        suffix = "_1" if last["Player_1"] == player else "_2"

        # Podstawowe statystyki z ostatniego meczu (Rank, Pts, Elo)
        for col in ["Rank", "Pts", "Elo"]:
            key = f"{col}{suffix}"
            value = last.get(key, None)
            if value != -1:
                player_data[col] = value
            else:
                player_data[col] = None
        
        # Dodanie dodatkowych cech z ostatniego meczu, które są w extra_features
        for feat in extra_features:
            # Sprawdź, czy jest z sufiksem _1 lub _2
            key_1 = feat + "_1"
            key_2 = feat + "_2"
            if key_1 in last:
                val = last[key_1]
            elif key_2 in last:
                val = last[key_2]
            else:
                val = last.get(feat, None)
            
            # Jeśli wartość jest -1, traktujemy jako brak danych
            if val == -1 or val == -1.0:
                val = None
            
            player_data[feat] = val
    
    if has_any_elo:
        player_rows.append(player_data)

# Tworzenie DataFrame i zapis
players_df = pd.DataFrame(player_rows)
players_df.to_csv("data/processed/atp_players_features_latest.csv", index=False)
print("Gotowe: zapisano dane graczy z ostatnimi Elo i dodatkowymi cechami.")
