import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def detect_anomalies(data_path, model_path="models/temperature_anomaly_model.joblib", output_dir="results"):
    """
    Wykrywa anomalie w nowych danych temperaturowych
    
    Parametry:
    - data_path: ścieżka do pliku z danymi do analizy
    - model_path: ścieżka do wytrenowanego modelu
    - output_dir: katalog do zapisania wyników
    """
    # Wczytanie modelu
    print(f"Wczytywanie modelu z {model_path}")
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    # Wczytanie danych
    print(f"Wczytywanie danych z {data_path}")
    df = pd.read_csv(data_path)
    
    # Konwersja timestamp do datetime (jeśli jest string)
    if isinstance(df['timestamp'][0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ekstrakcja cech czasowych
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Przygotowanie danych
    X = df[features].values
    X_scaled = scaler.transform(X)
    
    # Wykrywanie anomalii
    print("Wykrywanie anomalii...")
    df['anomaly_score'] = model.decision_function(X_scaled)
    df['is_anomaly'] = model.predict(X_scaled)
    
    # Konwersja oznaczeń (-1 = anomalia, 1 = normalne) do (1 = anomalia, 0 = normalne)
    df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})
    
    # Liczenie wykrytych anomalii
    anomaly_count = df['is_anomaly'].sum()
    print(f"Wykryto {anomaly_count} anomalii w danych")
    
    # Utworzenie katalogu na wyniki
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Zapis danych z wynikami
    result_path = f"{output_dir}/results_{timestamp}.csv"
    df.to_csv(result_path, index=False)
    print(f"Wyniki zapisane do {result_path}")
    
    # Wizualizacja wyników
    plt.figure(figsize=(16, 10))
    
    # Wykres temperatur z oznaczonymi anomaliami
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['cpu_temp'], label='CPU', alpha=0.7)
    plt.plot(df['timestamp'], df['gpu_temp'], label='GPU', alpha=0.7)
    plt.plot(df['timestamp'], df['mb_temp'], label='Płyta główna', alpha=0.7)
    
    # Zaznaczenie wykrytych anomalii
    anomalies = df[df['is_anomaly'] == 1]
    if not anomalies.empty:
        plt.scatter(anomalies['timestamp'], anomalies['cpu_temp'], 
                   color='red', marker='x', s=100, label='Anomalie CPU')
        plt.scatter(anomalies['timestamp'], anomalies['gpu_temp'], 
                   color='red', marker='x', s=100, label='Anomalie GPU')
        plt.scatter(anomalies['timestamp'], anomalies['mb_temp'], 
                   color='red', marker='x', s=100, label='Anomalie MB')
    
    plt.title('Temperatura z wykrytymi anomaliami')
    plt.xlabel('Czas')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Wykres anomaly score
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['anomaly_score'], label='Anomaly Score', color='blue')
    plt.axhline(y=0, color='red', linestyle='--', label='Próg anomalii')
    
    plt.title('Anomaly Scores')
    plt.xlabel('Czas')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Zapis wykresu
    plot_path = f"{output_dir}/anomalies_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Wykres zapisany do {plot_path}")
    plt.show()
    
    return df[df['is_anomaly'] == 1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wykrywanie anomalii w danych temperaturowych')
    parser.add_argument('data_path', type=str, help='Ścieżka do pliku CSV z danymi temperaturowymi')
    parser.add_argument('--model', type=str, default='models/temperature_anomaly_model.joblib',
                        help='Ścieżka do wytrenowanego modelu (domyślnie: models/temperature_anomaly_model.joblib)')
    parser.add_argument('--output', type=str, default='results',
                        help='Katalog do zapisania wyników (domyślnie: results)')
    
    args = parser.parse_args()
    anomalies = detect_anomalies(args.data_path, args.model, args.output)
    
    if not anomalies.empty:
        print("\nWykryte anomalie:")
        print(anomalies[['timestamp', 'cpu_temp', 'gpu_temp', 'mb_temp', 'anomaly_score']].to_string(index=False))
    else:
        print("\nNie wykryto żadnych anomalii w danych.")
