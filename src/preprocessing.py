# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_and_normalize_data(file_path):
    """Carrega o CSV bruto, limpa colunas, trata infinitos/nulos e normaliza."""
    print(f"Carregando e processando: {file_path}")
    df = pd.read_csv(file_path)
    
    # Remover espaços em branco nos nomes das colunas
    df.columns = df.columns.str.strip()
    
    # Substituir infinitos por NaN e remover nulos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Separar features (X) e rótulo (y) - assumindo 'Label' como coluna alvo
    # Mapeando Benign para 0 e ataques (DDoS) para 1
    if 'Label' in df.columns:
        y = df['Label'].apply(lambda x: 0 if str(x).strip().lower() == 'benign' else 1)
        X = df.drop(columns=['Label'])
    else:
        raise ValueError("A coluna 'Label' não foi encontrada no dataset.")
    
    # Remover colunas não numéricas se houver (ex: Timestamp, IP de origem/destino se presentes)
    X = X.select_dtypes(include=[np.number])
    
    # Normalização das características (StandardScaler)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_scaled['Label'] = y.values
    return X_scaled