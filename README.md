# Detecção Explicável de Intrusões em Fluxos de Rede sob Mudanças de Comportamento

Projeto experimental do Trabalho de Redes de Computadores da Universidade Federal do Ceará (UFC) — Campus Russas.

O projeto compara modelos de classificação de tráfego de rede com e sem adaptação a mudanças no fluxo de dados. Também calcula explicações SHAP para identificar quais características tiveram maior influência nas predições.

## Requisitos

- Python 3.10 ou superior
- Git
- As bases de dados descritas na seção [Bases de dados](#bases-de-dados)

## Clonar o repositório

```bash
git clone https://github.com/Wandallis/detecao-intrusao-adaptativa.git
cd detecao-intrusao-adaptativa
```

## Bases de dados

Os arquivos CSV **não são incluídos no repositório**, devido ao seu tamanho. Depois de clonar, baixe e extraia a base de dados CSE-CIC-IDS2018. O Kaggle disponibiliza conjuntos relacionados a ela nesta  [IDS Intrusion CSV](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)..

Para executar o experimento principal, coloque estes arquivos na pasta `data/raw/`:

| Arquivo | Dia do tráfego | Uso no projeto |
|---|---|---|
| `02-15-2018.csv` | 15 de fevereiro de 2018 | Treinamento inicial dos modelos |
| `02-21-2018.csv` | 21 de fevereiro de 2018 | Simulação do fluxo temporal e avaliação |

Após extrair o arquivo baixado, crie a pasta `data/raw/`, se necessário, e copie os dois CSVs para ela. A estrutura esperada é:

```text
detecao-intrusao-adaptativa/
├── data/
│   └── raw/
│       ├── 02-15-2018.csv
│       └── 02-21-2018.csv
├── src/
├── main.py
└── requirements.txt
```

Os CSVs são ignorados pelo Git e devem ser baixados separadamente em cada instalação do projeto. Consulte a licença e as condições de uso da base de dados na página de origem.

## Preparar o ambiente

### Windows — CMD

```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows — PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux ou macOS

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Executar o experimento

Com o ambiente virtual ativado e os dois arquivos CSV em `data/raw/`, execute:

```bash
python main.py
```

O programa processa os dados, treina modelos Random Forest e XGBoost e simula o fluxo do dia 21 de fevereiro em janelas sequenciais. Os resultados detalhados são salvos em:

```text
logs/metrics_detailed_rf.csv
logs/metrics_detailed_xgb.csv
```

A primeira execução pode levar algum tempo, dependendo do computador e do tamanho das bases.

## Gerar gráficos

Depois de executar `main.py`, gere os gráficos comparativos das métricas com:

```bash
python plot_results.py
```

Os painéis são salvos na pasta `logs/`.

Para gerar os gráficos de importância das características com SHAP, execute:

```bash
python plot_shap.py
```

Os gráficos SHAP também são salvos em `logs/`. Esse script usa amostras dos dados dos dois dias e treina um modelo Random Forest para comparar as explicações.

## Resolução de problemas

- **Arquivo de dados não encontrado:** confirme se os CSVs estão em `data/raw/` e se os nomes estão exatamente como indicados acima.
- **Erro ao importar uma biblioteca:** confirme que o ambiente virtual está ativado e instale as dependências com `pip install -r requirements.txt`.
- **Pasta `logs/` ainda não existe:** ela é criada automaticamente durante a execução.

## Estrutura do projeto

```text
.
├── data/raw/          # Bases CSV baixadas pelo usuário; não versionadas
├── src/
│   ├── drift_detector.py
│   ├── explainability.py
│   ├── models.py
│   └── preprocessing.py
├── main.py            # Executa o experimento
├── plot_results.py    # Gera painéis de métricas
├── plot_shap.py       # Gera gráficos SHAP
└── requirements.txt   # Dependências Python
```