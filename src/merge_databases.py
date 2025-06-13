import pandas as pd

# Wczytanie danych
df1 = pd.read_csv('data/raw/atp_matches_till_2022.csv')
df2 = pd.read_csv('data/processed/atp_tennis_processed.csv')

# Filtracja meczów od 2000 roku w df1
df1['tourney_date'] = pd.to_datetime(df1['tourney_date'], format='%Y%m%d')
df1 = df1[df1['tourney_date'].dt.year >= 2000]

# Normalizacja nazw i dat
df1['tourney_name_norm'] = df1['tourney_name'].str.lower().str.strip()
df2['Tournament_norm'] = df2['Tournament'].str.lower().str.strip()

df1['winner_name_norm'] = df1['winner_name'].str.lower().str.split().str[-1]
df1['loser_name_norm'] = df1['loser_name'].str.lower().str.split().str[-1]

df2['Player_1_norm'] = df2['Player_1'].str.lower().str.split().str[-1]
df2['Player_2_norm'] = df2['Player_2'].str.lower().str.split().str[-1]
df2['Winner_norm'] = df2['Winner'].str.lower().str.split().str[-1]

# Zamiana daty w df1 na string w formacie YYYY-MM-DD, by móc połączyć
df1['tourney_date_str'] = df1['tourney_date'].dt.strftime('%Y-%m-%d')

# Wybieramy z df1 tylko kolumny które chcesz dołączyć + kluczowe do łączenia i filtracji
cols_to_add = [
    'tourney_name_norm', 'tourney_date_str', 'winner_name_norm', 'loser_name_norm',
    'loser_seed', 'loser_entry', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age',
    'score', 'best_of', 'round', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
    'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
    'l_SvGms', 'l_bpSaved', 'l_bpFaced'
]
df1_sub = df1[cols_to_add].copy()

# Scalanie na podstawie nazwy turnieju i daty - left join by zachować wszystkie mecze z df2
merged = pd.merge(
    df2,
    df1_sub,
    left_on=['Tournament_norm', 'Date'],
    right_on=['tourney_name_norm', 'tourney_date_str'],
    how='left',
    suffixes=('', '_df1')
)

# Filtracja dopasowania zawodników i zwycięzcy
mask = (
    (merged['winner_name_norm'] == merged['Winner_norm']) & (
        ((merged['winner_name_norm'] == merged['Player_1_norm']) & (merged['loser_name_norm'] == merged['Player_2_norm'])) |
        ((merged['winner_name_norm'] == merged['Player_2_norm']) & (merged['loser_name_norm'] == merged['Player_1_norm']))
    )
)

# Zostaw tylko wiersze, gdzie jest dopasowanie (albo można też zostawić te bez dopasowania, ale z NaN w df1)
# Tu zakładam, że chcesz zachować wszystkie z df2, więc jeśli nie dopasowane - kolumny z df1 będą NaN
# Jeśli chcesz usunąć bez dopasowania, odkomentuj poniższą linię:
# merged = merged[mask]

# Wypełnij kolumny df1 NaN wartościami pustymi lub innymi jeśli chcesz, np.:
merged.loc[~mask, cols_to_add[2:]] = None  # kolumny z df1 poza kluczowymi na None tam gdzie brak dopasowania

# Usuń kolumny pomocnicze, które nie są potrzebne w finalnym pliku
merged.drop(columns=['tourney_name_norm', 'tourney_date_str', 'winner_name_norm', 'loser_name_norm'], inplace=True)

# Usuń duplikaty na podstawie unikalnych kolumn z df2 (np. Tournament, Date, Player_1, Player_2)
merged = merged.drop_duplicates(subset=['Tournament', 'Date', 'Player_1', 'Player_2'])

# Zapis do nowego pliku CSV
merged.to_csv('data/processed/atp_combined_2000_onwards.csv', index=False)

print("Zapisano wynikową bazę z danymi z bazy2 i wybranymi kolumnami z bazy1.")
