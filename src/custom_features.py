import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def calc_win_ratio_last_n(df, player, current_date, n=50):
    past_matches = df[
        ((df['Player_1'] == player) | (df['Player_2'] == player)) & 
        (df['Date'] < current_date)
    ].sort_values('Date', ascending=False).head(n)

    if past_matches.empty:
        return 0.5

    wins = 0
    total = 0
    for _, row in past_matches.iterrows():
        if row['Player_1'] == player:
            wins += row['is_player1_winner']
        else:
            wins += 1 - row['is_player1_winner']
        total += 1
    return wins / total if total > 0 else 0.5

def calc_elo_gradient(df, player, current_date, n=50):
    past_matches = df[
        ((df['Player_1'] == player) | (df['Player_2'] == player)) &
        (df['Date'] < current_date)
    ].sort_values('Date', ascending=False).head(n)

    if len(past_matches) < 2:
        return 0.0

    elos = []
    for _, row in past_matches.iterrows():
        if row['Player_1'] == player:
            elos.append(row['Elo_1'])
        else:
            elos.append(row['Elo_2'])

    return (elos[0] - elos[-1]) / n

def calc_h2h(df, player1, player2, current_date):
    past_matches = df[
        (
            ((df['Player_1'] == player1) & (df['Player_2'] == player2)) |
            ((df['Player_1'] == player2) & (df['Player_2'] == player1))
        ) & (df['Date'] < current_date)
    ]

    if past_matches.empty:
        return 0

    p1_wins = 0
    p2_wins = 0
    for _, row in past_matches.iterrows():
        if row['Player_1'] == player1:
            p1_wins += row['is_player1_winner']
            p2_wins += 1 - row['is_player1_winner']
        else:
            p2_wins += row['is_player1_winner']
            p1_wins += 1 - row['is_player1_winner']

    return p1_wins - p2_wins

def calc_h2h_surface(df, player1, player2, current_date, surface):
    past_matches = df[
        (
            (
                ((df['Player_1'] == player1) & (df['Player_2'] == player2)) |
                ((df['Player_1'] == player2) & (df['Player_2'] == player1))
            ) & 
            (df['Surface'] == surface)
        ) & (df['Date'] < current_date)
    ]

    if past_matches.empty:
        return 0

    p1_wins = 0
    p2_wins = 0
    for _, row in past_matches.iterrows():
        if row['Player_1'] == player1:
            p1_wins += row['is_player1_winner']
            p2_wins += 1 - row['is_player1_winner']
        else:
            p2_wins += row['is_player1_winner']
            p1_wins += 1 - row['is_player1_winner']

    return p1_wins - p2_wins

def add_custom_features(df):
    df = df.sort_values("Date").reset_index(drop=True)

    win_last_5_diff = []
    win_last_25_diff = []
    win_last_50_diff = []
    win_last_100_diff = []
    win_last_250_diff = []

    elo_grad_20_diff = []
    elo_grad_35_diff = []
    elo_grad_50_diff = []
    elo_grad_100_diff = []

    h2h_diff = []
    h2h_surface_diff = []

    diff_N_games = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        p1 = row['Player_1']
        p2 = row['Player_2']
        date = row['Date']
        surface = row['Surface']

        w5_1 = calc_win_ratio_last_n(df, p1, date, n=5)
        w5_2 = calc_win_ratio_last_n(df, p2, date, n=5)
        win_last_5_diff.append(w5_1 - w5_2)

        w25_1 = calc_win_ratio_last_n(df, p1, date, n=25)
        w25_2 = calc_win_ratio_last_n(df, p2, date, n=25)
        win_last_25_diff.append(w25_1 - w25_2)

        w50_1 = calc_win_ratio_last_n(df, p1, date, n=50)
        w50_2 = calc_win_ratio_last_n(df, p2, date, n=50)
        win_last_50_diff.append(w50_1 - w50_2)

        w100_1 = calc_win_ratio_last_n(df, p1, date, n=100)
        w100_2 = calc_win_ratio_last_n(df, p2, date, n=100)
        win_last_100_diff.append(w100_1 - w100_2)

        w250_1 = calc_win_ratio_last_n(df, p1, date, n=250)
        w250_2 = calc_win_ratio_last_n(df, p2, date, n=250)
        win_last_250_diff.append(w250_1 - w250_2)

        g20_1 = calc_elo_gradient(df, p1, date, n=20)
        g20_2 = calc_elo_gradient(df, p2, date, n=20)
        elo_grad_20_diff.append(g20_1 - g20_2)

        g35_1 = calc_elo_gradient(df, p1, date, n=35)
        g35_2 = calc_elo_gradient(df, p2, date, n=35)
        elo_grad_35_diff.append(g35_1 - g35_2)

        g50_1 = calc_elo_gradient(df, p1, date, n=50)
        g50_2 = calc_elo_gradient(df, p2, date, n=50)
        elo_grad_50_diff.append(g50_1 - g50_2)

        g100_1 = calc_elo_gradient(df, p1, date, n=100)
        g100_2 = calc_elo_gradient(df, p2, date, n=100)
        elo_grad_100_diff.append(g100_1 - g100_2)

        h2h_diff.append(calc_h2h(df, p1, p2, date))
        h2h_surface_diff.append(calc_h2h_surface(df, p1, p2, date, surface))

        # Różnica liczby rozegranych meczy
        matches_p1 = len(df[((df['Player_1'] == p1) | (df['Player_2'] == p1)) & (df['Date'] < date)])
        matches_p2 = len(df[((df['Player_1'] == p2) | (df['Player_2'] == p2)) & (df['Date'] < date)])
        diff_N_games.append(matches_p1 - matches_p2)

    df["win_last_5_diff"] = win_last_5_diff
    df["win_last_25_diff"] = win_last_25_diff
    df["win_last_50_diff"] = win_last_50_diff
    df["win_last_100_diff"] = win_last_100_diff
    df["win_last_250_diff"] = win_last_250_diff

    df["elo_grad_20_diff"] = elo_grad_20_diff
    df["elo_grad_35_diff"] = elo_grad_35_diff
    df["elo_grad_50_diff"] = elo_grad_50_diff
    df["elo_grad_100_diff"] = elo_grad_100_diff

    df["h2h_diff"] = h2h_diff
    df["h2h_surface_diff"] = h2h_surface_diff

    df["diff_N_games"] = diff_N_games

    return df
