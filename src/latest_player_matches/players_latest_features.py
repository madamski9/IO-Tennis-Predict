import pandas as pd

matches = pd.read_csv("data/processed/atp_tennis_processed_test.csv")
matches["Date"] = pd.to_datetime(matches["Date"])

# Zrobimy "stack" graczy i potem znajdziemy ich ostatnie mecze:

# Weź tylko potrzebne kolumny, np. Player_1, Player_2, Date, Elo_1, Elo_2, Rank_1, Rank_2 itd.

# Ale łatwiej będzie stworzyć osobną ramkę dla każdego zawodnika, z ich danymi z ostatniego meczu.

# Przygotuj listę zawodników
players = set(matches["Player_1"]).union(set(matches["Player_2"]))

player_rows = []

for player in players:
    # Znajdź wszystkie mecze z udziałem tego zawodnika
    player_matches = matches[(matches["Player_1"] == player) | (matches["Player_2"] == player)]
    
    # Posortuj po dacie malejąco (ostatni mecz na górze)
    player_matches = player_matches.sort_values("Date", ascending=False)
    
    # Weź pierwszy (ostatni) mecz
    last_match = player_matches.iloc[0]
    
    # Ustal, czy zawodnik był Player_1 czy Player_2 w tym meczu
    if last_match["Player_1"] == player:
        suffix = "_1"
    else:
        suffix = "_2"
    
    # Pobierz cechy tego zawodnika z odpowiednimi suffixami
    features = {"Player": player}
    for col in matches.columns:
        if col.endswith(suffix):
            base_col = col[:-2]  # usuń _1 lub _2
            features[base_col] = last_match[col]
    
    player_rows.append(features)

# Stwórz ramkę danych
players_df = pd.DataFrame(player_rows)

# Teraz masz unikalne rekordy z ostatnimi wartościami dla każdego zawodnika
players_df.to_csv("data/processed/atp_players_features_latest.csv", index=False)
print("Zapisano dane z ostatnich meczów zawodników")
