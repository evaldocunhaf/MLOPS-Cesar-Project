# MLOPS-Cesar-Project

Pipeline de classificação para prever o desempenho acadêmico (`Low / Medium / High`) com base em hábitos de jogos e sono. Dataset: [Gaming and Mental Health](https://www.kaggle.com/datasets/shaistashahid/gaming-and-mental-health).

---

## Estrutura

```
├── data/
│   ├── raw/                    # CSV baixado do Kaggle
│   └── processed/              # Parquet após transformação
├── extract/                    # Módulo de extração (Kaggle API)
├── transformer/                # Módulo de pré-processamento
├── trainer/                    # Módulo de treinamento (sklearn pipeline)
├── models/                     # Modelo serializado (.joblib)
├── metrics/                    # Métricas DVC (metrics.json)
├── notebooks/                  # EDA e experimentos exploratórios
├── mlflow-docker/              # Volumes do servidor MLflow (Docker)
├── extract.py                  # Entrypoint estágio extract
├── transform.py                # Entrypoint estágio preprocess
├── train.py                    # Entrypoint estágio train
├── serve.py                    # API FastAPI para inferência
├── params.yaml                 # Hiperparâmetros e configurações
├── dvc.yaml                    # Definição do pipeline DVC
└── docker-compose.yml          # Servidor MLflow
```

---

## Pipeline

```
extract  →  preprocess  →  train
```

Gerenciado pelo DVC. Cada estágio só roda novamente se suas dependências mudarem.

---

## Setup

```bash
# Criar ambiente virtual e instalar dependências
python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Configurar credenciais do Kaggle (~/.kaggle/kaggle.json)
```

---

## Uso

```bash
# Subir servidor MLflow
make mlflow-up

# Rodar pipeline completo
make pipeline

# Rodar apenas o treinamento
make train

# Ver métricas
dvc metrics show

# Comparar métricas entre commits
dvc metrics diff

# Servir modelo
uvicorn serve:app --port 8000
```

---

## Trocar modelo

Em `params.yaml`, alterar:

```yaml
train:
  model: random_forest   # opções: random_forest | knn | xgboost
```

Depois rodar `make train`. Um novo run é registrado no MLflow automaticamente.

---

## MLflow

Interface disponível em `http://localhost:5050` após `make mlflow-up`.  
Cada execução de `train.py` registra parâmetros, métricas e o artefato do modelo.
