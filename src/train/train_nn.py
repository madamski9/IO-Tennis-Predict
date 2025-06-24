import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === wczytanie danych ===
df = pd.read_csv("data/processed/atp_tennis_processed.csv")
df["Date"] = pd.to_datetime(df["Date"])

# === usuwanie meczów bez sensownych roznic ===
df = df[
    (df["win_last_25_diff"] != 0) |
    (df["elo_grad_50_diff"] != 0) |
    (df["h2h_diff"] != 0)
]

# === dodanie feature: surface_elo_diff i inne interakcje ===
surface_elo_cols = {
    "Hard": "Elo_Hard_diff",
    "Clay": "Elo_Clay_diff",
    "Grass": "Elo_Grass_diff",
}
df["surface_elo_diff"] = df.apply(lambda row: row.get(surface_elo_cols.get(row["Surface"], ""), 0.0), axis=1)
df["elo_x_form"] = df["Elo_diff"] * df["win_last_100_diff"]
df["elo_plus_form"] = df["Elo_diff"] + df["win_last_100_diff"]
df["elo_form_ratio"] = df["Elo_diff"] / (df["win_last_100_diff"] + 1e-5)

# === wybor cech ===
features = [
    "Elo_diff", "surface_elo_diff", "rank_diff", "pts_diff",
    "win_last_5_diff", "win_last_25_diff", "win_last_50_diff", "win_last_100_diff", "win_last_250_diff",
    "diff_N_games", "elo_grad_20_diff", "elo_grad_35_diff", "elo_grad_50_diff", "elo_grad_100_diff",
    "h2h_diff", "h2h_surface_diff", "elo_x_form", "elo_plus_form", "elo_form_ratio"
]
target = "is_player1_winner"

train_df = df[df["Date"] < "2022-01-01"]
val_df = df[(df["Date"] >= "2022-01-01") & (df["Date"] < "2023-01-01")]
test_df = df[df["Date"] >= "2023-01-01"]

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]

# === skalowanie danych ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# === budowa modelu ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(features),)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === early stopping ===
early_stop = EarlyStopping(patience=10, restore_best_weights=True)

# === trening ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nNeural Network Test Accuracy: {acc:.4f}")
print(f"Neural Network Test AUC: {auc:.4f}")

best_acc, best_f1, best_thresh = 0, 0, 0.5
for thresh in np.arange(0.3, 0.7, 0.01):
    preds = (y_pred_prob > thresh).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh
    if f1 > best_f1:
        best_f1 = f1

print(f"Best threshold for accuracy: {best_thresh:.2f} with accuracy {best_acc:.4f}")
print(f"Best F1 score: {best_f1:.4f}")

os.makedirs("models", exist_ok=True)
model.save("models/nn_model.keras")
joblib.dump(scaler, "models/scaler_nn.joblib")
print("Neural Network model saved to models/nn_model.keras")

# === wykresy ===
os.makedirs("images/smaller_dataset/neural_network", exist_ok=True)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("images/smaller_dataset/neural_network/training_curves_full.png")
plt.close()
print("Training plots saved to images/smaller_dataset/neural_network/training_curves_full.png")
