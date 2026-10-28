# plot_shap.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from src.preprocessing import clean_and_normalize_data
from src.models import get_random_forest_model, get_xgboost_model
from src.explainability import compute_shap_explanation

def generate_shap_plots():
    print("=== GERANDO GRÁFICOS DE EXPLICABILIDADE SHAP PARA O TCC ===")
    
    train_path = os.path.join("data", "raw", "02-15-2018.csv")
    streaming_path = os.path.join("data", "raw", "02-21-2018.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(streaming_path):
        print("[ERRO] Arquivos de dados não encontrados em data/raw/.")
        return

    # 1. Carregar amostras dos dados de treino e streaming
    print("\n[1/3] Carregando amostras do tráfego de rede...")
    df_train = clean_and_normalize_data(train_path).sample(n=1000, random_state=42)
    df_stream = clean_and_normalize_data(streaming_path).sample(n=1000, random_state=42)
    
    X_train = df_train.drop(columns=['Label'])
    y_train = df_train['Label']
    X_stream = df_stream.drop(columns=['Label'])
    y_stream = df_stream['Label']
    
    # 2. Treinar o modelo inicial e calcular o SHAP Pré-Drift
    print("\n[2/3] Calculando SHAP para o Modelo Inicial (Dia 15-02)...")
    model_rf = get_random_forest_model()
    model_rf.fit(X_train, y_train)
    
    shap_vals_train, _ = compute_shap_explanation(model_rf, X_train)
    
    # Tratamento para formatos multidimensionais do SHAP
    if isinstance(shap_vals_train, list):
        shap_vals_train = shap_vals_train[1]
    elif isinstance(shap_vals_train, np.ndarray) and shap_vals_train.ndim == 3:
        shap_vals_train = shap_vals_train[:, :, 1]

    # Gerar Gráfico SHAP Pré-Drift
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_train, X_train, plot_type="bar", show=False)
    plt.title("Importância das Características (SHAP) - Treino Inicial (02-15-2018)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('logs', exist_ok=True)
    output_pre = os.path.join('logs', 'shap_importance_pre_drift.png')
    plt.savefig(output_pre, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Gráfico SHAP Pré-Drift salvo em: {output_pre}")

    # 3. Retreinar o modelo adaptativo com os novos dados e calcular SHAP Pós-Drift
    print("\n[3/3] Calculando SHAP para o Modelo Adaptativo Pós-Retreinamento (Dia 21-02)...")
    model_rf_adapt = get_random_forest_model()
    model_rf_adapt.fit(X_stream, y_stream)
    
    shap_vals_stream, _ = compute_shap_explanation(model_rf_adapt, X_stream)
    
    if isinstance(shap_vals_stream, list):
        shap_vals_stream = shap_vals_stream[1]
    elif isinstance(shap_vals_stream, np.ndarray) and shap_vals_stream.ndim == 3:
        shap_vals_stream = shap_vals_stream[:, :, 1]

    # Gerar Gráfico SHAP Pós-Drift
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_stream, X_stream, plot_type="bar", show=False)
    plt.title("Importância das Características (SHAP) - Pós-Adaptação (02-21-2018)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    output_post = os.path.join('logs', 'shap_importance_post_drift.png')
    plt.savefig(output_post, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCESSO] Gráfico SHAP Pós-Drift salvo em: {output_post}")

if __name__ == "__main__":
    generate_shap_plots()