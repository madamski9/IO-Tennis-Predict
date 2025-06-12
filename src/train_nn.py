import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import os

# === 1. Wczytanie i przygotowanie danych ===
df = pd.read_csv("data/processed/atp_tennis_processed.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Dodaj kolumnę surface_elo_diff
surface_elo_cols = {
    "Hard": "Elo_Hard_diff",
    "Clay": "Elo_Clay_diff",
    "Grass": "Elo_Grass_diff",
}
df["surface_elo_diff"] = df.apply(
    lambda row: row.get(surface_elo_cols.get(row["Surface"], ""), 0.0), axis=1
)

features = [
    "Elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "pts_diff",
    "win_last_5_diff",
    "win_last_25_diff",
    "win_last_50_diff",
    "win_last_100_diff",
    "win_last_250_diff",
    "diff_N_games",
    "elo_grad_20_diff",
    "elo_grad_35_diff",
    "elo_grad_50_diff",
    "elo_grad_100_diff",
    "h2h_diff",
    "h2h_surface_diff"
]

target = "is_player1_winner"

# Podział na zbiór uczący/walidacyjny/testowy (po dacie)
train_df = df[df["Date"] < "2022-01-01"]
val_df = df[(df["Date"] >= "2022-01-01") & (df["Date"] < "2023-01-01")]
test_df = df[df["Date"] >= "2023-01-01"]

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]

# Skalowanie (ważne dla sieci neuronowych)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# === 2. Definicja modelu ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(len(features),)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 3. Trenowanie modelu ===
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# === 4. Ewaluacja ===
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nAccuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")

# Upewnij się, że katalog istnieje
os.makedirs("images/neural_network", exist_ok=True)

# === 5. Wykresy accuracy i loss ===
history = model.history
plt.figure(figsize=(12, 5))

# === Accuracy ===
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# === Loss ===
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Zapis wykresów
plt.tight_layout()
plt.savefig("images/neural_network/training_curves.png")
plt.close()
print("Wykresy zapisane w images/neural_network/training_curves.png")
