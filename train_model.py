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
    sensor_stats = {
    'cpu_mean': df['cpu_temp'].mean(),
    'cpu_std': df['cpu_temp'].std(),
    'gpu_mean': df['gpu_temp'].mean(),
    'gpu_std': df['gpu_temp'].std(),
    'mb_mean': df['mb_temp'].mean(),
    'mb_std': df['mb_temp'].std()
}

    joblib.dump({'model': model, 'scaler': scaler, 'features': features, 'sensor_stats': sensor_stats}, model_path)
    print(f"Model zapisany do {model_path}")
    
    # Wizualizacja wyników treningu
    plt.figure(figsize=(14, 8))
    
    # Wykres z wykrytymi anomaliami
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['cpu_temp'], label='CPU', alpha=0.7, color='blue')
    plt.plot(df['timestamp'], df['gpu_temp'], label='GPU', alpha=0.7, color='orange')
    plt.plot(df['timestamp'], df['mb_temp'], label='Płyta główna', alpha=0.7, color='green')
    
    # Zaznaczenie rzeczywistych anomalii dla każdego czujnika osobno
    real_anomalies = df[df['is_anomaly'] == 1]
    if not real_anomalies.empty:
        # Filtrowanie po typie anomalii
        cpu_anom = real_anomalies[real_anomalies['anomaly_type'] == 'cpu']
        gpu_anom = real_anomalies[real_anomalies['anomaly_type'] == 'gpu']
        mb_anom = real_anomalies[real_anomalies['anomaly_type'] == 'mb']
        
        plt.scatter(cpu_anom['timestamp'], cpu_anom['cpu_temp'], 
                   color='red', marker='o', s=60, label='Rzeczywiste anomalie CPU', alpha=0.8)
        plt.scatter(gpu_anom['timestamp'], gpu_anom['gpu_temp'], 
                   color='red', marker='s', s=60, label='Rzeczywiste anomalie GPU', alpha=0.8)
        plt.scatter(mb_anom['timestamp'], mb_anom['mb_temp'], 
                   color='red', marker='^', s=60, label='Rzeczywiste anomalie MB', alpha=0.8)
    
    # Zaznaczenie przewidzianych anomalii dla każdego czujnika osobno
    predicted_anomalies = df[df['predicted_anomaly'] == 1]
    if not predicted_anomalies.empty:
        # Oblicz progi dla każdego czujnika
        cpu_threshold = df['cpu_temp'].mean() + 3*df['cpu_temp'].std()
        gpu_threshold = df['gpu_temp'].mean() + 3*df['gpu_temp'].std()
        mb_threshold = df['mb_temp'].mean() + 3*df['mb_temp'].std()
        
        # Filtruj anomalie dla każdego czujnika
        cpu_anom = predicted_anomalies[predicted_anomalies['cpu_temp'] > cpu_threshold]
        gpu_anom = predicted_anomalies[predicted_anomalies['gpu_temp'] > gpu_threshold]
        mb_anom = predicted_anomalies[predicted_anomalies['mb_temp'] > mb_threshold]
        
        plt.scatter(cpu_anom['timestamp'], cpu_anom['cpu_temp'], 
                   color='purple', marker='x', s=100, label='Wykryte anomalie CPU', alpha=0.9)
        plt.scatter(gpu_anom['timestamp'], gpu_anom['gpu_temp'], 
                   color='purple', marker='P', s=100, label='Wykryte anomalie GPU', alpha=0.9)
        plt.scatter(mb_anom['timestamp'], mb_anom['mb_temp'], 
                   color='purple', marker='*', s=100, label='Wykryte anomalie MB', alpha=0.9)
    
    plt.title('Wykrywanie anomalii - dane treningowe')
    plt.xlabel('Czas')
    plt.ylabel('Temperatura (°C)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
    plt.savefig('data/training_results.png', bbox_inches='tight', dpi=300)
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
