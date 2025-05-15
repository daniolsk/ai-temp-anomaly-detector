import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_temperature_data(n_samples=1000, n_anomalies=50, save_path="data/synthetic_temperatures.csv"):
    """
    Generuje syntetyczne dane temperaturowe z anomaliami
    
    Parametry:
    - n_samples: liczba punktów danych do wygenerowania
    - n_anomalies: liczba anomalii do wstawienia
    - save_path: ścieżka do zapisania danych
    """
    # Ustawienie losowego ziarna dla reprodukowalności
    np.random.seed(42)
    
    # Generowanie czasu startowego i końcowego (co 1 minuta)
    start_time = datetime.now() - timedelta(minutes=n_samples)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generowanie normalnych temperatur CPU (35°C - 65°C z dziennym cyklem)
    time_of_day = np.array([(t.hour * 60 + t.minute) / (24 * 60) for t in timestamps])
    base_temp = 45 + 10 * np.sin(2 * np.pi * time_of_day)  # dzienne wahania
    noise = np.random.normal(0, 1, n_samples)  # normalny szum
    cpu_temp = base_temp + noise
    
    # Generowanie temperatur GPU (zazwyczaj wyższe, 40°C - 75°C)
    gpu_base_temp = 50 + 15 * np.sin(2 * np.pi * time_of_day)
    gpu_noise = np.random.normal(0, 1.5, n_samples)
    gpu_temp = gpu_base_temp + gpu_noise
    
    # Generowanie temperatur płyty głównej (30°C - 50°C)
    mb_base_temp = 38 + 5 * np.sin(2 * np.pi * time_of_day)
    mb_noise = np.random.normal(0, 0.8, n_samples)
    mb_temp = mb_base_temp + mb_noise
    
    # Tworzenie DataFrame z normalnymi danymi
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_temp': cpu_temp,
        'gpu_temp': gpu_temp,
        'mb_temp': mb_temp,
        'is_anomaly': 0  # 0 = normalne dane
    })
    
    # Dodawanie anomalii
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.randint(0, 3)
        if anomaly_type == 0:  # Nagły skok CPU
            df.loc[idx, 'cpu_temp'] = np.random.uniform(80, 95)
        elif anomaly_type == 1:  # Nagły skok GPU
            df.loc[idx, 'gpu_temp'] = np.random.uniform(85, 100)
        else:  # Nagły skok płyty głównej
            df.loc[idx, 'mb_temp'] = np.random.uniform(60, 70)
        
        df.loc[idx, 'is_anomaly'] = 1  # 1 = anomalia
    
    # Zapis danych do pliku CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    # Wyświetlenie podstawowych statystyk
    print(f"Wygenerowano {n_samples} próbek z {n_anomalies} anomaliami")
    print(f"Dane zapisane do {save_path}")
    
    # Wizualizacja danych
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['cpu_temp'], label='CPU', alpha=0.7)
    plt.plot(df['timestamp'], df['gpu_temp'], label='GPU', alpha=0.7)
    plt.plot(df['timestamp'], df['mb_temp'], label='Płyta główna', alpha=0.7)
    
    anomalies = df[df['is_anomaly'] == 1]
    plt.scatter(anomalies['timestamp'], anomalies['cpu_temp'], color='red', marker='o', label='Anomalie CPU')
    plt.scatter(anomalies['timestamp'], anomalies['gpu_temp'], color='red', marker='s', label='Anomalie GPU')
    plt.scatter(anomalies['timestamp'], anomalies['mb_temp'], color='red', marker='^', label='Anomalie płyty')
    
    plt.title('Syntetyczne dane temperaturowe z anomaliami')
    plt.xlabel('Czas')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/synthetic_temperatures_plot.png')
    plt.show()

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    generate_temperature_data()
