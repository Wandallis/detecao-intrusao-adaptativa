# src/drift_detector.py
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.models import get_random_forest_model, get_xgboost_model
from src.explainability import compute_shap_explanation, get_top_features_shap

def run_temporal_simulation(train_df, streaming_windows, model_type='rf'):
    """Simula a execução temporal comparando o modelo estático e o adaptativo, registrando todas as métricas."""
    
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']
    
    # Definir proporção para scale_pos_weight se for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Instanciar modelos Estático (Congelado) e Adaptativo (Retreinável)
    if model_type == 'rf':
        model_static = get_random_forest_model()
        model_adaptive = get_random_forest_model()
    else:
        model_static = get_xgboost_model(scale_pos_weight=scale_weight)
        model_adaptive = get_xgboost_model(scale_pos_weight=scale_weight)
        
    # Treinamento inicial (Baseline)
    print(f"Treinando modelos iniciais ({model_type.upper()})...")
    model_static.fit(X_train, y_train)
    model_adaptive.fit(X_train, y_train)
    
    logs = []
    
    # Processamento em Janelas Temporais
    for window_id, window_df in enumerate(streaming_windows):
        X_w = window_df.drop(columns=['Label'])
        y_w = window_df['Label']
        
        # --- 1. Avaliação Modelo Estático (Baseline) ---
        t_start = time.time()
        preds_static = model_static.predict(X_w)
        time_static_inf = time.time() - t_start
        
        prec_stat = precision_score(y_w, preds_static, zero_division=0)
        rec_stat = recall_score(y_w, preds_static, zero_division=0)
        f1_stat = f1_score(y_w, preds_static, zero_division=0)
        
        cm_stat = confusion_matrix(y_w, preds_static, labels=[0, 1])
        # FPR = Falsos Positivos / (Falsos Positivos + Verdadeiros Negativos)
        tn_s, fp_s, fn_s, tp_s = cm_stat.ravel() if cm_stat.size == 4 else (0, 0, 0, 0)
        fpr_stat = fp_s / (fp_s + tn_s) if (fp_s + tn_s) > 0 else 0.0
        
        # --- 2. Avaliação Modelo Adaptativo (Proposta) ---
        t_start = time.time()
        preds_adapt = model_adaptive.predict(X_w)
        time_adapt_inf = time.time() - t_start
        
        prec_adapt = precision_score(y_w, preds_adapt, zero_division=0)
        rec_adapt = recall_score(y_w, preds_adapt, zero_division=0)
        f1_adapt = f1_score(y_w, preds_adapt, zero_division=0)
        
        cm_adapt = confusion_matrix(y_w, preds_adapt, labels=[0, 1])
        tn_a, fp_a, fn_a, tp_a = cm_adapt.ravel() if cm_adapt.size == 4 else (0, 0, 0, 0)
        fpr_adapt = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0.0
        
        # --- Gatilho de Detecção de Drift e Retreinamento ---
        retrained = False
        retrain_time = 0.0
        if f1_adapt < 0.85: # Limiar de queda de desempenho simulando drift
            t_retrain = time.time()
            model_adaptive.fit(X_w, y_w) # Retreinamento adaptativo
            retrain_time = time.time() - t_retrain
            retrained = True
            
            # Reavaliar após retreino
            preds_adapt = model_adaptive.predict(X_w)
            prec_adapt = precision_score(y_w, preds_adapt, zero_division=0)
            rec_adapt = recall_score(y_w, preds_adapt, zero_division=0)
            f1_adapt = f1_score(y_w, preds_adapt, zero_division=0)
            
            cm_adapt = confusion_matrix(y_w, preds_adapt, labels=[0, 1])
            tn_a, fp_a, fn_a, tp_a = cm_adapt.ravel() if cm_adapt.size == 4 else (0, 0, 0, 0)
            fpr_adapt = fp_a / (fp_a + tn_a) if (fp_a + tn_a) > 0 else 0.0

        # --- 3. Explicabilidade com SHAP ---
        shap_vals, time_shap = compute_shap_explanation(model_adaptive, X_w.head(50))
        top_feats = get_top_features_shap(shap_vals, X_w.columns, top_n=3)['Feature'].tolist()
        
        # Armazenar todas as métricas detalhadas no log
        logs.append({
            'window_id': window_id,
            'model': model_type,
            # Métricas Modelo Estático
            'precision_static': prec_stat,
            'recall_static': rec_stat,
            'f1_static': f1_stat,
            'fpr_static': fpr_stat,
            # Métricas Modelo Adaptativo
            'precision_adaptive': prec_adapt,
            'recall_adaptive': rec_adapt,
            'f1_adaptive': f1_adapt,
            'fpr_adaptive': fpr_adapt,
            # Custos e XAI
            'inference_time': time_adapt_inf,
            'retrained': retrained,
            'retrain_time': retrain_time,
            'shap_time': time_shap,
            'top_feature_1': top_feats[0] if len(top_feats) > 0 else '',
            'top_feature_2': top_feats[1] if len(top_feats) > 1 else '',
        })
        print(f"Janela {window_id:03d} | F1 Estático: {f1_stat:.4f} | F1 Adapt.: {f1_adapt:.4f} | FPR Adapt.: {fpr_adapt:.4f} | Retreinado: {retrained}")

    # Salvar logs completos em CSV
    os.makedirs('logs', exist_ok=True)
    logs_df = pd.DataFrame(logs)
    output_csv = os.path.join('logs', f'metrics_detailed_{model_type}.csv')
    logs_df.to_csv(output_csv, index=False)
    print(f"Logs detalhados salvos com sucesso em {output_csv}")