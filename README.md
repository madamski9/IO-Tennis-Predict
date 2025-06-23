# Projekt: Predykcja wyników meczów tenisowych ATP

## Cel projektu

Celem projektu było stworzenie modelu machine learningowego, który na podstawie danych historycznych przewiduje **czy zawodnik nr 1 wygra mecz**. Predykcja ta ma charakter binarny (0 = przegrana, 1 = wygrana) i opiera się wyłącznie na cechach zawodników oraz danych meczowych, bez użycia kursów bukmacherskich.

---

## Dane i preprocessing

Dane wejściowe pochodziły ze zbioru spotkań ATP (`atp_tennis.csv`), a następnie zostały przetworzone (`atp_tennis_processed.csv`). Początkowo zawierał on podstawowe informacje o meczach i zawodnikach. Aby zwiększyć wartość predykcyjną modelu, znacząco rozszerzyłem bazę o nowe zmienne poprzez preprocessing. Szczególnie ważną zmienna okazały się punkty ELO, które stworzyłem za pomocą specjalnego równania (`https://en.wikipedia.org/wiki/Elo_rating_system`).

### Przetwarzanie i inżynieria cech:

- Usunięto mecze bez różnicy w kluczowych cechach (`win_last_25_diff`, `elo_grad_50_diff`, `h2h_diff`).
- Dodano:
  - Punkty ELO (`Elo_diff`, `surface_elo_diff`)
  - Różnice w rankingach, punktach, formie
  - Historia bezpośrednich spotkań (`h2h_diff`, `h2h_surface_diff`)
  - Pochodne ELO z formą:
    - `elo_x_form = Elo_diff * win_last_100_diff`
    - `elo_form_ratio = Elo_diff / (win_last_100_diff + ε)`

Zmienna celu (`is_player1_winner`) to etykieta binarna wskazująca zwycięstwo zawodnika nr 1.

---

## Co zostało pominięte?

W zbiorze danych dostępne są również kolumny `Odd_1` i `Odd_2`, które przedstawiają kursy bukmacherskie. Są one bardzo silnym predyktorem, ponieważ zawierają wiedzę rynku. Jednak zostały świadomie pominięte, ponieważ ich użycie uczyniłoby problem trywialnym i nie oddawałoby faktycznej skuteczności modelu opartego wyłącznie na cechach graczy.

---

## Modele ML

### Główny model:
- **XGBoost (drzewa gradientowe)**
  - **Tuning hiperparametrów za pomocą Optuna** (z `StratifiedKFold`)
    - Poprawa skuteczności o około **2% AUC**
    - Użycie `early_stopping` i `logloss` jako metryki walidacyjnej
  - Najlepsze wyniki:
    - **AUC ≈ 0.738**
    - **Accuracy ≈ 65.3%**

### Eksperymenty:
- **Ensemble stacking**: Połączenie `XGBoost` i `Random Forest` w modelu `StackingClassifier`
  - Poprawa accuracy o ~0.5 punktu procentowego

- **Sieć neuronowa (Keras)**:
  - Porównywalna skuteczność jak XGBoost
  - Wykresy pokazują, że nie wnosi dużej poprawy

- **kNN, Native Bayers, Decision Tree**:
 - kNN accuracy: 0.6080
 - Naive Bayes accuracy: 0.6482
 - Decision Tree accuracy: 0.5705

---

## Wizualizacje i interpretacja

- Wygenerowano wykresy:
  - **ROC curves** (porównanie modeli)
  - **Feature importance** (XGBoost)

Z wykresów i interpretacji cech wynika, że:
> **Najbardziej wpływowym predyktorem są punkty ELO**, a szczególnie wersje dostosowane do nawierzchni (`surface_elo_diff`) oraz forma (`win_last_100_diff`).

---

## Test na prawdziwej drabince turniejowej

Po wytrenowaniu wszystkich modeli (najlepszy osiągał accuracy ~66%) zdecydowałem się przetestować model w realistycznym scenariuszu:

1. **Pobranie drabinki z aktualnego turnieju wielkoszlemowego**:
   - Stworzony został skrypt: `src/bracket/scrape_grandslam_bracket`
   - Automatycznie pobiera drabinkę (matchupy) graczy z internetu

2. **Predykcja meczów turniejowych**:
   - Na podstawie przygotowanej bazy danych (zawierającej cechy graczy z ich ostatnich meczów) 
     skrypt: `src/latest_player_matches/players_latest_features`, csv: `data/processed/atp_players_features_latest`
   - Dla każdego gracza pobierany był jego ostatni mecz
   - Model przewidywał zwycięzców kolejnych rund turnieju

3. **Model skutecznie wytypował zwycięzcę całego turnieju!**

4. **Wizualizacje**:
   - Dla każdego meczu stworzono wykres pokazujący, które cechy miały największy wpływ na decyzję modelu
   - Stworzono również **pełny wykres drabinki z wynikami modelu**
   - Nadal występowały problemy w meczach z niewielką różnicą ELO — model jest **najpewniejszy, gdy różnica ELO jest wyraźna**

---

## Techniczne szczegóły

- Dane: `data/processed/atp_tennis_processed.csv`
- Język: Python
- Biblioteki: wszystkie zostały zapisane w: `requirements.txt`
- Modele zapisane w `models/`
- Wykresy zapisane w `images/`

---

## Podsumowanie

Stworzenie własnego systemu ELO oraz wzbogacenie zbioru danych o kilkanaście cech inżynieryjnych pozwoliło zbudować skuteczny model predykcyjny bez korzystania z zewnętrznych kursów. Pomimo eksperymentów z sieciami neuronowymi i ensemble, XGBoost nadal pozostaje najbardziej wydajnym rozwiązaniem.

Dodatkowo, zastosowanie tuningu hiperparametrów przy użyciu Optuna przyniosło zauważalną, choć umiarkowaną poprawę (~2% AUC), co świadczy o solidności modelu już w wersji bazowej.

Model osiąga solidne wyniki przy zachowaniu wysokiej interpretowalności i niezależności od wiedzy rynku.
