import pandas as pd

def compute_h2h_winrate(df):
    h2h = {}

    winrates = []
    for _, row in df.iterrows():
        p1 = row["Player_1"]
        p2 = row["Player_2"]
        key = tuple(sorted([p1, p2]))

        if key not in h2h:
            h2h[key] = []

        results = h2h[key]
        # 1 if player_1 won, 0 if lost
        winrate = sum(1 for r in results if r == p1) / len(results) if results else 0.5
        winrates.append(winrate)

        # add current result
        h2h[key].append(row["Winner"])

    df["h2h_winrate"] = winrates
    return df


def compute_form_score(df, n_matches=5):
    history = {}
    form_scores = []

    for _, row in df.iterrows():
        player = row["Player_1"]
        winner = row["Winner"]

        past_matches = history.get(player, [])
        wins = sum(1 for r in past_matches[-n_matches:] if r == "W")
        form_scores.append(wins / n_matches)

        # update history
        result = "W" if winner == player else "L"
        history.setdefault(player, []).append(result)

    df["form_score"] = form_scores
    return df


def compute_surface_winrate(df):
    winrate_history = {}
    surface_wr = []

    for _, row in df.iterrows():
        player = row["Player_1"]
        surface = row["Surface"]
        winner = row["Winner"]

        key = (player, surface)
        past = winrate_history.get(key, [])
        winrate = sum(1 for r in past if r == "W") / len(past) if past else 0.5
        surface_wr.append(winrate)

        # update
        result = "W" if winner == player else "L"
        winrate_history.setdefault(key, []).append(result)

    df["surface_winrate"] = surface_wr
    return df


def add_h2h_and_surface_winrate(df):
    df = compute_h2h_winrate(df)
    df = compute_form_score(df)
    df = compute_surface_winrate(df)
    return df
