import pandas as pd

def flip_name(name):
    parts = name.split()
    if len(parts) == 2:
        # np. J. Sinner -> Sinner J.
        return f"{parts[1]} {parts[0]}"
    elif len(parts) > 2:
        # np. P. Llamas Ruiz -> Ruiz P. Llamas
        return f"{parts[-1]} {' '.join(parts[:-1])}"
    else:
        return name  # jeśli nieznany format, zostawiamy

# Wczytanie pliku
df = pd.read_csv('data/predict_tourney/real_bracket.csv')

# Zamiana nazwisk i imion w obu kolumnach
df["Player_1"] = df["Player_1"].apply(flip_name)
df["Player_2"] = df["Player_2"].apply(flip_name)

# Zapis do nowego pliku lub wypisanie na ekran
df.to_csv('data/predict_tourney/real_bracket.csv', index=False)
print("Zapisano do: draw_roland_garros_2025_128_flipped.csv")
