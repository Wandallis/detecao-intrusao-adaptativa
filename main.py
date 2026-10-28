# main.py
import os
import pandas as pd
from src.preprocessing import clean_and_normalize_data
from src.drift_detector import run_temporal_simulation

def main():
    print("=== INICIANDO EXPERIMENTO DE TCC: DETECÇÃO DE DDoS & CONCEPT DRIFT ===")
    
    # 1. Definindo os arquivos reais da base CSE-CIC-IDS2018
    train_file_name = "02-15-2018.csv"  # Usado para treinar os modelos iniciais (Baseline)
    streaming_file_name = "02-21-2018.csv"  # Usado para simular o tráfego futuro (Concept Drift)
    
    train_path = os.path.join("data", "raw", train_file_name)
    streaming_path = os.path.join("data", "raw", streaming_file_name)
    
    # Verificar se ambos os arquivos existem na pasta data/raw/
    if not os.path.exists(train_path) or not os.path.exists(streaming_path):
        print(f"[ERRO] Um ou ambos os arquivos não foram encontrados em data/raw/!")
        print(f"Certifique-se de que '{train_file_name}' e '{streaming_file_name}' estão na pasta correta.")
        return

    # 2. Pré-processamento dos dados de Treinamento (Dia 15-02)
    print(f"\n[1/3] Processando dados de treino inicial: {train_file_name}")
    train_data = clean_and_normalize_data(train_path)
    
    # 3. Pré-processamento dos dados de Streaming/Avaliação (Dia 21-02)
    print(f"\n[2/3] Processando dados de avaliação temporal: {streaming_file_name}")
    streaming_data_full = clean_and_normalize_data(streaming_path)
    
    # 4. Dividir o arquivo de streaming em janelas temporais sequenciais (ex: blocos de 5.000 linhas)
    window_size = 5000
    streaming_windows = [
        streaming_data_full.iloc[i:i + window_size] 
        for i in range(0, len(streaming_data_full), window_size)
    ]
    
    print(f"\n[3/3] Configuração pronta!")
    print(f"-> Amostras de Treino Inicial: {len(train_data)}")
    print(f"-> Total de janelas temporais de streaming: {len(streaming_windows)}")

    # 5. Executar simulação para o Random Forest
    print("\n--- Executando simulação com Random Forest ---")
    run_temporal_simulation(train_data, streaming_windows, model_type='rf')
    
    # 6. Executar simulação para o XGBoost
    print("\n--- Executando simulação com XGBoost ---")
    run_temporal_simulation(train_data, streaming_windows, model_type='xgb')
    
    print("\n=== EXPERIMENTO CONCLUÍDO COM SUCESSO! ===")
    print("Verifique os resultados e logs gerados na pasta 'logs/'.")

if __name__ == "__main__":
    main()