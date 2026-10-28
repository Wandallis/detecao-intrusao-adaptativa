# plot_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots():
    print("=== GERANDO PAINEL GRÁFICO COMPLETO PARA O TCC ===")
    
    logs_dir = "logs"
    files = {
        'Random Forest': os.path.join(logs_dir, 'metrics_detailed_rf.csv'),
        'XGBoost': os.path.join(logs_dir, 'metrics_detailed_xgb.csv')
    }
    
    for model_name, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"[AVISO] Arquivo de log para {model_name} não encontrado em {file_path}. Execute o main.py primeiro.")
            continue
            
        df = pd.read_csv(file_path)
        
        # Criando uma figura com 4 subgráficos (Grid 2x2)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Avaliação Temporal de Desempenho - {model_name}', fontsize=16, fontweight='bold')
        
        metrics_config = [
            ('f1_static', 'f1_adaptive', 'F1-Score (Equilíbrio Geral)', (0.0, 1.05), axs[0, 0]),
            ('recall_static', 'recall_adaptive', 'Recall (Capacidade de Detecção de Ataques)', (0.0, 1.05), axs[0, 1]),
            ('precision_static', 'precision_adaptive', 'Precisão (Confiabilidade dos Alertas)', (0.0, 1.05), axs[1, 0]),
            ('fpr_static', 'fpr_adaptive', 'Taxa de Falsos Alarmes - FPR (Falsos Positivos)', (0.0, 0.20), axs[1, 1])
        ]
        
        for stat_col, adapt_col, title, ylim, ax in metrics_config:
            ax.plot(df['window_id'], df[stat_col], label='Estático (Baseline)', color='red', linestyle='--', linewidth=1.5)
            ax.plot(df['window_id'], df[adapt_col], label='Adaptativo (Proposta)', color='blue', linestyle='-', linewidth=1.5)
            
            # Destacar pontos de retreinamento no gráfico de F1-Score
            retrains = df[df['retrained'] == True]
            if not retrains.empty and 'F1-Score' in title:
                ax.scatter(retrains['window_id'], df.loc[retrains.index, adapt_col], color='orange', s=40, zorder=5, label='Retreinamento')
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Janelas Temporais (Sequência de Tráfego)', fontsize=9)
            ax.set_ylabel('Valor da Métrica', fontsize=9)
            ax.set_ylim(ylim)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best', fontsize=8)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Salvar a imagem consolidada
        output_image = os.path.join(logs_dir, f'full_metrics_grid_{model_name.lower().replace(" ", "_")}.png')
        plt.savefig(output_image, dpi=300)
        print(f"[SUCESSO] Painel completo salvo em: {output_image}")
        plt.close()

if __name__ == "__main__":
    generate_plots()