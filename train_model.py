import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def train_anomaly_detection_model(data_path="data/synthetic_temperatures.csv", 
                                 model_path="models/temperature_anomaly_model.joblib"):
    """
    Trenowanie modelu Isolation Forest do wykrywania anomalii
    
    Parametry:
    - data_path: ścieżka do danych treningowych
    - model_path: ścieżka do zapisania modelu
    """
    # Wczytanie danych
    print(f"Wczytywanie danych z {data_path}")
    df = pd.read_csv(data_path)
    
    # Konwersja timestamp do datetime (jeśli jest string)
    if isinstance(df['timestamp'][0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ekstrakcja cech czasowych (godzina dnia, dzień tygodnia)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Wybór cech do trenowania
    features = ['cpu_temp', 'gpu_temp', 'mb_temp', 'hour', 'minute', 'day_of_week']
    X = df[features].values
    
    # Normalizacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Trenowanie modelu Isolation Forest
    print("Trenowanie modelu Isolation Forest...")
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,  # Spodziewany procent anomalii w danych
        random_state=42
    )
    model.fit(X_scaled)
    
    # Ocena modelu na danych treningowych
    df['anomaly_score'] = model.decision_function(X_scaled)
    df['predicted_anomaly'] = model.predict(X_scaled)
    
    # Konwersja oznaczeń (-1 = anomalia, 1 = normalne) do (1 = anomalia, 0 = normalne) 
    df['predicted_anomaly'] = df['predicted_anomaly'].map({1: 0, -1: 1})
    
    # Zapisanie modelu i skalera
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'features': features}, model_path)
    print(f"Model zapisany do {model_path}")
    
    # Wizualizacja wyników treningu
    plt.figure(figsize=(14, 8))
    
    # Wykres z wykrytymi anomaliami
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['cpu_temp'], label='CPU', alpha=0.7)
    plt.plot(df['timestamp'], df['gpu_temp'], label='GPU', alpha=0.7)
    plt.plot(df['timestamp'], df['mb_temp'], label='Płyta główna', alpha=0.7)
    
    # Zaznaczone rzeczywiste anomalie
    real_anomalies = df[df['is_anomaly'] == 1]
    plt.scatter(real_anomalies['timestamp'], real_anomalies['cpu_temp'], 
                color='red', marker='o', label='Rzeczywiste anomalie')
    
    # Zaznaczone przewidziane anomalie
    predicted_anomalies = df[df['predicted_anomaly'] == 1]
    plt.scatter(predicted_anomalies['timestamp'], predicted_anomalies['cpu_temp'], 
                color='purple', marker='x', s=100, label='Wykryte anomalie')
    
    plt.title('Wykrywanie anomalii - dane treningowe')
    plt.xlabel('Czas')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Wykres wyników anomaly score
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['anomaly_score'], label='Anomaly Score', color='blue')
    plt.axhline(y=0, color='red', linestyle='--', label='Próg anomalii')
    
    plt.title('Anomaly Scores')
    plt.xlabel('Czas')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/training_results.png')
    plt.show()
    
    # Ocena modelu
    true_anomalies = df['is_anomaly'].sum()
    detected_anomalies = df['predicted_anomaly'].sum()
    correctly_detected = ((df['is_anomaly'] == 1) & (df['predicted_anomaly'] == 1)).sum()
    
    print(f"\nWyniki treningu:")
    print(f"Rzeczywistych anomalii: {true_anomalies}")
    print(f"Wykrytych anomalii: {detected_anomalies}")
    print(f"Prawidłowo wykrytych anomalii: {correctly_detected}")
    print(f"Dokładność wykrywania: {correctly_detected/true_anomalies:.2%}")

if __name__ == "__main__":
    train_anomaly_detection_model()
