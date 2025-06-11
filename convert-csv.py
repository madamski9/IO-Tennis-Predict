import pandas as pd

df = pd.read_csv("atp_tennis.csv", parse_dates=["Date"])
df = df[df["Date"] >= "2015-01-01"]
df.to_csv("atp_tennis_2015.csv", index=False)

print(f"Liczba meczów od 2015 roku: {len(df)}")
