import collections
import pandas as pd
from collections import defaultdict
import os

def build_surface_elo_history(df, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)

    surfaces = ["Hard", "Clay", "Grass"]
    
    for surface in surfaces:
        surface_elo = defaultdict(lambda: 1500)
        history = []

        surface_df = df[df[f"Surface"] == surface].copy()
        surface_df = surface_df.sort_values("Date")

        for _, row in surface_df.iterrows():
            date = pd.to_datetime(row["Date"])
            p1 = row["Player_1"]
            p2 = row["Player_2"]
            winner = row["Winner"]

            R1 = surface_elo[p1]
            R2 = surface_elo[p2]

            expected_1 = 1 / (1 + 10 ** ((R2 - R1) / 400))
            S1 = 1 if winner == p1 else 0

            R1_new = R1 + 32 * (S1 - expected_1)
            R2_new = R2 + 32 * ((1 - S1) - (1 - expected_1))

            surface_elo[p1] = R1_new
            surface_elo[p2] = R2_new

            history.append((date, p1, R1_new))
            history.append((date, p2, R2_new))

        surface_elo_df = pd.DataFrame(history, columns=["Date", "Player", "Elo"])
        surface_elo_df.to_csv(f"{output_dir}/elo_history_{surface.lower()}.csv", index=False)

def calculate_surface_elo(df, k=32, start_elo=1500):
    surface_elos = {
        "Hard": collections.defaultdict(lambda: start_elo),
        "Grass": collections.defaultdict(lambda: start_elo)
    }

    for surface in surface_elos:
        df[f"Elo_{surface}_1"] = 0.0
        df[f"Elo_{surface}_2"] = 0.0

    df = df.sort_values("Date").reset_index(drop=True)

    for i, row in df.iterrows():
        p1 = row["Player_1"]
        p2 = row["Player_2"]
        winner = row["Winner"]
        surface = row["Surface"]
        date = row["Date"]

        if surface not in surface_elos:
            continue

        elo_dict = surface_elos[surface]
        R1 = elo_dict[p1]
        R2 = elo_dict[p2]

        expected_1 = 1 / (1 + 10 ** ((R2 - R1) / 400))
        S1 = 1 if winner == p1 else 0

        R1_new = R1 + k * (S1 - expected_1)
        R2_new = R2 + k * ((1 - S1) - (1 - expected_1))

        df.at[i, f"Elo_{surface}_1"] = R1
        df.at[i, f"Elo_{surface}_2"] = R2

        elo_dict[p1] = R1_new
        elo_dict[p2] = R2_new

    for surface in surface_elos:
        df[f"Elo_{surface}_diff"] = df[f"Elo_{surface}_1"] - df[f"Elo_{surface}_2"]

    return df
