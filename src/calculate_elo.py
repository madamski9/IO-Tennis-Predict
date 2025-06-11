import collections
import pandas as pd

def calculate_elo(df, k=32, start_elo=1500):
    elo = collections.defaultdict(lambda: start_elo)
    elo_1 = []
    elo_2 = []
    elo_history = []

    df = df.sort_values("Date").reset_index(drop=True)

    for _, row in df.iterrows():
        p1, p2 = row["Player_1"], row["Player_2"]
        winner = row["Winner"]
        date = row["Date"]

        R1, R2 = elo[p1], elo[p2]

        expected_1 = 1 / (1 + 10 ** ((R2 - R1) / 400))
        S1 = 1 if winner == p1 else 0

        R1_new = R1 + k * (S1 - expected_1)
        R2_new = R2 + k * ((1 - S1) - (1 - expected_1))

        elo[p1] = R1_new
        elo[p2] = R2_new

        elo_1.append(R1)
        elo_2.append(R2)

        elo_history.append((date, p1, R1_new))
        elo_history.append((date, p2, R2_new))

    df["Elo_1"] = elo_1
    df["Elo_2"] = elo_2
    df["Elo_diff"] = df["Elo_1"] - df["Elo_2"]

    elo_df = pd.DataFrame(elo_history, columns=["Date", "Player", "Elo"])
    elo_df["Date"] = pd.to_datetime(elo_df["Date"])

    return df, elo_df
