# src/explainability.py
import shap
import pandas as pd
import numpy as np
import time

def compute_shap_explanation(model, X_sample):
    """Calcula os valores SHAP para uma amostra de tráfego e mede o tempo."""
    start_time = time.time()
    
    # Usando TreeExplainer otimizado para modelos baseados em árvores (RF e XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    elapsed_time = time.time() - start_time
    return shap_values, elapsed_time

def get_top_features_shap(shap_values, feature_names, top_n=3):
    """Extrai as principais features ranqueadas pelo impacto médio absoluto do SHAP."""
    
    # Tratamento para quando shap_values é uma lista (comum em classificação binária no scikit-learn)
    if isinstance(shap_values, list):
        # Pega a matriz correspondente à classe positiva (índice 1)
        vals = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Se for um array 3D (ex: [amostras, features, classes])
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    # Garante que vals é bidimensional antes de tirar a média por coluna
    if vals.ndim == 2:
        mean_abs_shap = np.abs(vals).mean(axis=0)
    else:
        mean_abs_shap = np.abs(vals)

    # Força a conversão para array 1D para evitar o erro do Pandas
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()

    feature_importance = pd.DataFrame({
        'Feature': list(feature_names),
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=False)
    
    return feature_importance.head(top_n)