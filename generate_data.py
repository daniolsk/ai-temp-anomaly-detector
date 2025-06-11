import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def generate_temperature_data(n_samples=1000, n_anomalies=50, save_path="data/synthetic_temperatures.csv", seed=42, plot_title="Syntetyczne dane temperaturowe z anomaliami"):
    """
    Generuje syntetyczne dane temperaturowe z anomaliami
    """
    np.random.seed(seed)
    
    # Generowanie czasu
    start_time = datetime.now() - timedelta(minutes=n_samples)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generowanie normalnych temperatur
    time_of_day = np.array([(t.hour * 60 + t.minute) / (24 * 60) for t in timestamps])
    
    # CPU
    base_temp = 45 + 10 * np.sin(2 * np.pi * time_of_day)
    cpu_temp = base_temp + np.random.normal(0, 1, n_samples)
    
    # GPU
    gpu_base_temp = 50 + 15 * np.sin(2 * np.pi * time_of_day)
    gpu_temp = gpu_base_temp + np.random.normal(0, 1.5, n_samples)
    
    # Płyta główna
    mb_base_temp = 38 + 5 * np.sin(2 * np.pi * time_of_day)
    mb_temp = mb_base_temp + np.random.normal(0, 0.8, n_samples)

    # Tworzenie DataFrame z nową kolumną anomaly_type
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_temp': cpu_temp,
        'gpu_temp': gpu_temp,
        'mb_temp': mb_temp,
        'is_anomaly': 0,
        'anomaly_type': None
    })

    # Generowanie anomalii
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['cpu', 'gpu', 'mb'])
        direction = np.random.choice(['up', 'down'])  # losowo wzrost lub spadek
        
        if anomaly_type == 'cpu':
            if direction == 'up':
                df.loc[idx, 'cpu_temp'] = np.random.uniform(80, 95)  
            else:
                df.loc[idx, 'cpu_temp'] = np.random.uniform(10, 25)  
        elif anomaly_type == 'gpu':
            if direction == 'up':
                df.loc[idx, 'gpu_temp'] = np.random.uniform(85, 100)
            else:
                df.loc[idx, 'gpu_temp'] = np.random.uniform(15, 30)
        else:
            if direction == 'up':
                df.loc[idx, 'mb_temp'] = np.random.uniform(60, 70)
            else:
                df.loc[idx, 'mb_temp'] = np.random.uniform(10, 20)
        
        df.loc[idx, 'is_anomaly'] = 1
        df.loc[idx, 'anomaly_type'] = anomaly_type

    # Zapis danych
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    # Wizualizacja z poprawionym zaznaczaniem anomalii
    plt.figure(figsize=(15, 8))
    
    # Normalne dane
    plt.plot(df['timestamp'], df['cpu_temp'], label='CPU', alpha=0.7)
    plt.plot(df['timestamp'], df['gpu_temp'], label='GPU', alpha=0.7)
    plt.plot(df['timestamp'], df['mb_temp'], label='Płyta główna', alpha=0.7)
    
    # Anomalie z podziałem na typy
    anomalies_cpu = df[df['anomaly_type'] == 'cpu']
    anomalies_gpu = df[df['anomaly_type'] == 'gpu']
    anomalies_mb = df[df['anomaly_type'] == 'mb']
    
    plt.scatter(anomalies_cpu['timestamp'], anomalies_cpu['cpu_temp'], 
                color='red', marker='o', s=60, label='Anomalie CPU')
    plt.scatter(anomalies_gpu['timestamp'], anomalies_gpu['gpu_temp'], 
                color='red', marker='s', s=60, label='Anomalie GPU')
    plt.scatter(anomalies_mb['timestamp'], anomalies_mb['mb_temp'], 
                color='red', marker='^', s=60, label='Anomalie MB')
    
    plt.title(plot_title)
    plt.xlabel('Czas')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/' + str(plot_title) + '.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    generate_temperature_data(seed=42, save_path="data/synthetic_temperatures.csv")
    generate_temperature_data(
        n_samples=1000,
        n_anomalies=50,
        save_path="data/new_temperatures.csv",
        seed=123,
        plot_title="Wykres nowych danych do testowania modelu"
    )
