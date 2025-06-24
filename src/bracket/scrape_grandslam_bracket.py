import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_draw(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def parse_draw(soup):
    data = []

    for round_num in range(1, 8):  # runda 1–7
        round_divs = soup.select(f'div.draw-round-{round_num} div.draw-item')
        for match in round_divs:
            names = match.select("div.name > a")
            if len(names) != 2:
                continue  # pomin jeśli mecz niepelny

            p1 = names[0].text.strip()
            p2 = names[1].text.strip()

            stats = match.select("div.draw-stats")
            winner = ""
            if stats:
                players_stats = stats[0].select("div.player-info")
                for ps in players_stats:
                    if ps.find_next_sibling("div", class_="winner"):
                        name_tag = ps.select_one("div.name > a")
                        if name_tag:
                            winner = name_tag.text.strip()

            data.append({
                "Round": round_num,
                "Player_1": p1,
                "Player_2": p2,
                "Winner": winner,
            })

    return pd.DataFrame(data)

if __name__ == "__main__":
    url = "https://www.atptour.com/en/scores/archive/roland-garros/520/2025/draws"
    soup = fetch_draw(url)
    df = parse_draw(soup)
    df.to_csv("data/predict_tourney/real_bracket.csv", index=False)
    print(df.head())
