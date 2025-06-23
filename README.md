# TennisPredict – Predykcja wyników meczów tenisowych ATP

**TennisPredict** to projekt, którego celem jest przewidywanie wyników meczów tenisowych na podstawie danych historycznych i zaawansowanej inżynierii cech. W projekcie wykorzystano różne algorytmy uczenia maszynowego, w tym XGBoost, sieci neuronowe oraz klasyczne klasyfikatory, a także własny system rankingowy ELO.

---

## Główne funkcjonalności

- **Preprocessing i inżynieria cech**: automatyczne przetwarzanie surowych danych ATP, wyliczanie różnic rankingowych, punktowych, formy, historii bezpośrednich spotkań (H2H), gradientów ELO i wielu innych cech.
- **System ELO**: własna implementacja rankingu ELO, także osobno dla każdej nawierzchni (Hard, Clay, Grass).
- **Trening modeli ML**: porównanie skuteczności XGBoost, LightGBM, sieci neuronowej (Keras), Random Forest, stacking oraz klasycznych klasyfikatorów (kNN, Naive Bayes, Decision Tree).
- **Tuning hiperparametrów**: automatyczny tuning XGBoost z użyciem Optuna.
- **Wizualizacje**: wykresy ELO w czasie, pairploty, feature importance, wizualizacja drabinki turniejowej.
- **Symulacja turnieju**: przewidywanie przebiegu rzeczywistej drabinki turniejowej (np. Roland Garros) i wyznaczanie zwycięzcy na podstawie aktualnych cech graczy.

---

## Struktura projektu

- `src/` – kod źródłowy (preprocessing, modele, symulacje, wizualizacje)
- `data/` – dane surowe i przetworzone
- `models/` – zapisane modele ML
- `images/` – wykresy i wizualizacje
- `raport/` – raport końcowy projektu

---

## Jak uruchomić?

1. **Zainstaluj wymagane biblioteki**  
   ```
   pip install -r requirements.txt
   ```

2. **Przetwórz dane**  
   ```
   python src/preprocess.py
   ```

3. **Wytrenuj model**  
   ```
   python src/train_xgboost.py
   ```

4. **Symuluj turniej**  
   ```
   python src/bracket/simulate_tournament.py
   ```

---

## Wyniki

- Najlepszy model (XGBoost) osiąga accuracy ~65% i AUC ~0.74 na danych testowych.
- Model skutecznie przewiduje przebieg rzeczywistych turniejów, a najważniejszą cechą okazuje się ranking ELO oraz forma zawodnika.

---

## Technologie

- Python, pandas, numpy, scikit-learn, xgboost, lightgbm, keras, matplotlib, seaborn, optuna, joblib

---

## Autor

Projekt zrealizowany w ramach kursu Inteligencja Obliczeniowa, Uniwersytet Gdański.

---

**Więcej szczegółów znajdziesz w pliku [`raport/raport.md`](raport/raport.md).**