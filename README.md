![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![MLflow](https://img.shields.io/badge/MLflow-DagsHub-orange)

# Gaming & Mental Health MLOps

Projeto de classificação de desempenho acadêmico/profissional (`High`, `Medium`, `Low`) com base em hábitos de jogo e sono. Implementa um pipeline completo de Machine Learning com práticas de MLOps: pipeline reproduzível com DVC, tracking remoto com MLflow no DagsHub, API REST de inferência com FastAPI e frontend HTML para demonstração.

---

## O que o projeto faz

Dado o perfil de um jogador (idade, gênero, horas de jogo, qualidade do sono, etc.), o modelo prediz a classe de performance acadêmica/profissional do indivíduo entre `High`, `Medium` e `Low`. O treino suporta três algoritmos intercambiáveis: KNN, Random Forest e XGBoost.

### Pipeline end-to-end

```
   Kaggle dataset
         │
         ▼
   extract.py  ── data/raw/*.csv
         │
         ▼
   transform.py ── data/processed/data.parquet
         │
         ▼
   train.py
   ├─► models/model_pipeline.joblib  (fallback local)
   ├─► metrics/metrics.json          (DVC metric)
   └─► DagsHub MLflow:
         ├─ run com params/metrics
         └─ Model Registry: gaming-mental-health v1, v2, v3...
                                                       │
                                                       │ load_model("models:/.../latest")
                                                       ▼
                              ┌─────────────────────────────────┐
                              │ gaming_api (FastAPI :8000)      │
                              │ serve.py                        │
                              └─────────────┬───────────────────┘
                                            │ POST /predict
                                            ▼
                              ┌─────────────────────────────────┐
                              │ gaming_web (nginx :8501)        │
                              │ frontend HTML + JS              │
                              └─────────────────────────────────┘
```


## Quick start

```bash
# 1. Clone
git clone git@github.com:evaldocunhaf/MLOPS-Cesar-Project.git
cd MLOPS-Cesar-Project

# 2. Criar arquivo .env com credenciais (a partir do template)
cp .env.sample .env
# Editar .env e preencher as três variáveis:
#   KAGGLE_API_TOKEN=<seu token Kaggle>
#   MLFLOW_TRACKING_USERNAME=<seu usuário DagsHub>
#   MLFLOW_TRACKING_PASSWORD=<token DagsHub gerado em Settings > Tokens>

# 3. Subir a stack (api + frontend)
make stack-up

# 4. Abrir no browser
#    Frontend:  http://localhost:8501
#    Swagger:   http://localhost:8000/docs
#    Health:    http://localhost:8000/health
```

O container `gaming_api` baixa automaticamente o modelo do **MLflow Registry no DagsHub** durante o startup. O primeiro boot leva ~30 segundos por causa desse download.

Para parar tudo: `make stack-down`.

---

## Arquitetura

A stack roda em dois containers locais (`api` e `web`). O tracking de experimentos e o Model Registry são serviços remotos hospedados no DagsHub.

| Componente | Onde roda | Função |
|---|---|---|
| `gaming_api` | Container local (porta 8000) | FastAPI servindo `POST /predict`, carrega modelo do MLflow Registry no startup |
| `gaming_web` | Container local (porta 8501) | nginx servindo o `web/index.html` (form, exemplos, debugger) |
| MLflow tracking | DagsHub (https://dagshub.com/evaldocunhaf/MLOPs-Cesar.mlflow) | Runs, métricas, params, artifacts |
| Model Registry | DagsHub | `gaming-mental-health` versionado (v1, v2, ...) |
| Dataset raw | Kaggle (`shaistashahid/gaming-and-mental-health`) | Origem dos dados |

### Configuração da API via env vars (no `docker-compose.yml`)

| Variável | Default | Descrição |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/evaldocunhaf/MLOPs-Cesar.mlflow` | Endpoint do MLflow no DagsHub |
| `MODEL_URI` | `models:/gaming-mental-health/latest` | Versão a carregar (use `models:/.../1` para pinned) |
| `MLFLOW_TRACKING_USERNAME` | (do `.env`) | Usuário DagsHub |
| `MLFLOW_TRACKING_PASSWORD` | (do `.env`) | Token DagsHub |

### Fallback local

Se o DagsHub estiver inacessível (sem internet, credenciais inválidas), a API tenta carregar o `models/model_pipeline.joblib` local. O caminho está montado como volume read-only (`./models:/app/models:ro`). O campo `model_source` em `/health` indica qual fonte foi usada.

---

## Stack técnica

| Ferramenta | Função |
|---|---|
| Python 3.13 | Linguagem principal |
| pandas, pyarrow | Manipulação de dados e formato Parquet |
| scikit-learn 1.8 | Pré-processamento (OneHotEncoder) e modelos (KNN, RandomForest) |
| XGBoost 3.2 | Modelo Gradient Boosting opcional |
| kagglehub | Download automático do dataset do Kaggle |
| DVC 3.67 | Versionamento de pipeline e dados |
| MLflow 3.10 | Tracking de experimentos e Model Registry |
| DagsHub | Hospedagem remota do MLflow Registry |
| FastAPI 0.135 + Uvicorn | API REST de inferência |
| Pydantic 2.12 | Validação de payload de entrada |
| nginx (alpine) | Servidor estático do frontend |
| Docker Compose v2 | Orquestração local |

---

## Estrutura do projeto

```
MLOPS-Cesar-Project/
├── extract/                       Módulo de extração (Kaggle API)
│   ├── baseApiExtractor.py
│   └── kaggleExtract.py
│
├── transformer/                   Módulo de pré-processamento
│   ├── baseTransformer.py
│   └── stepTransformer.py
│
├── trainer/                       Classes de treino e avaliação
│   ├── baseTrainer.py
│   └── sklearnTrainer.py
│
├── data/
│   ├── raw/                       Dados brutos (DVC-tracked, gitignored)
│   └── processed/                 Dados limpos (DVC-tracked, gitignored)
│
├── models/                        Modelo treinado .joblib (DVC-tracked, gitignored)
├── metrics/                       Métricas do pipeline DVC
│
├── web/                           Frontend
│   ├── index.html                 Form + exemplos + debugger
│   └── Dockerfile                 nginx:alpine
│
├── extract.py                     Stage 1: download do Kaggle
├── transform.py                   Stage 2: limpeza e mapeamento
├── train.py                       Stage 3: treino + log MLflow + registro
├── serve.py                       FastAPI: GET /health, POST /predict
│
├── dvc.yaml                       Definição das stages do pipeline
├── dvc.lock                       Hashes dos artefatos versionados
├── params.yaml                    Hiperparâmetros centralizados
│
├── requirements.txt               Deps de treino/desenvolvimento
├── requirements-api.txt           Deps enxutas só para o container da API
├── Dockerfile.api                 Imagem do gaming_api
├── docker-compose.yml             Orquestração api + web
├── Makefile                       Atalhos de comandos
│
├── .env.sample                    Template de variáveis de ambiente
└── .env                           Credenciais reais (NÃO comitado)
```

---

## Configuração

### `params.yaml`

Arquivo central de hiperparâmetros. O DVC monitora este arquivo e re-executa o pipeline quando algo muda. Para trocar o algoritmo, alterar a chave `train.model`:

```yaml
train:
  model: knn          # opções: knn, random_forest, xgboost
  experiment_name: gaming-mental-health

mlflow:
  tracking_uri: https://dagshub.com/evaldocunhaf/MLOPs-Cesar.mlflow
  model_artifact_path: model
```

### `.env`

Não é versionado (`*.env` no gitignore). Criar a partir de `.env.sample`:

```dotenv
KAGGLE_API_TOKEN=<token Kaggle, pega em kaggle.com/settings/account>
MLFLOW_TRACKING_USERNAME=<usuário DagsHub>
MLFLOW_TRACKING_PASSWORD=<token DagsHub, pega em dagshub.com/user/settings/tokens>
```

O `train.py` lê o `.env` via `python-dotenv`. O cliente MLflow autentica automaticamente com `MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD`.

---

## Comandos via Makefile

```
make help          Lista todos os targets disponíveis

Setup:
  make setup       Cria .venv e instala requirements.txt
  make install     Reinstala dependências num venv existente

Treino (loga e registra no DagsHub):
  make pipeline    Pipeline completo: extract -> preprocess -> train
  make extract     Apenas baixa o dataset do Kaggle
  make preprocess  Apenas pré-processa
  make train       Apenas treina (uso mais comum)

Stack Docker:
  make stack-up    Sobe api + web (docker compose up -d --build)
  make stack-down  Derruba os containers
  make stack-ps    Lista containers
  make stack-logs  Tail dos logs da api
  make restart-api Reinicia a API (puxa modelo mais recente do DagsHub)

Outros:
  make clean       Apaga models/*.joblib e metrics/metrics.json
```

---

## Fluxo de novo treinamento

Para treinar um modelo, registrar no DagsHub e fazer a API servir a nova versão:

```bash
# 1. (Opcional) Trocar modelo ou hiperparams em params.yaml
#    Ex: editar a linha "model: knn" para "model: random_forest"

# 2. Treinar e registrar nova versão no DagsHub Model Registry
make train
# Equivalente a: .venv/bin/python -m dvc repro train

# 3. Verificar no DagsHub
#    Abrir https://dagshub.com/evaldocunhaf/MLOPs-Cesar.mlflow
#    Aba "Models" -> gaming-mental-health -> nova versão aparece (v2, v3...)

# 4. Reiniciar a API para puxar a nova versão
make restart-api

# 5. Confirmar a fonte do modelo
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"model_source":"mlflow:models:/gaming-mental-health/latest"}
```

Como `MODEL_URI` está configurado como `models:/gaming-mental-health/latest`, a API sempre puxa a versão mais recente registrada. Para fixar uma versão específica, alterar essa env var no `docker-compose.yml` para `models:/gaming-mental-health/1` (ou outro número).

---

## API

### `GET /health`

Retorna o estado da API e a fonte do modelo carregado.

```bash
curl http://localhost:8000/health
```

Resposta:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_source": "mlflow:models:/gaming-mental-health/latest"
}
```

- `model_source` com prefixo `mlflow:` indica que veio do Registry DagsHub.
- `model_source` com prefixo `local:` indica que caiu no fallback do `.joblib` em disco.

### `POST /predict`

Recebe um JSON com 10 features e retorna a classe predita.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 22,
    "gender": "Male",
    "daily_gaming_hours": 4.5,
    "game_genre": "FPS",
    "primary_game": "Valorant",
    "gaming_platform": "PC",
    "sleep_hours": 6.0,
    "sleep_quality": "Fair",
    "sleep_disruption_frequency": "Sometimes",
    "face_to_face_social_hours_weekly": 8.0
  }'
```

Resposta:
```json
{
  "prediction": "Medium",
  "model": "mlflow:models:/gaming-mental-health/latest"
}
```

Payloads inválidos (ex: `age: -5`, `sleep_hours: 50`) retornam HTTP 422 com detalhes do erro de validação Pydantic.

A documentação interativa do Swagger está em http://localhost:8000/docs com payload de exemplo pré-preenchido (botão "Try it out").

---

## Frontend

`http://localhost:8501` mostra:

- Formulário com os 10 campos (selects pré-populados com valores válidos do dataset).
- Painel de 10 exemplos clicáveis (de "Healthy student" a "Insomniac") que preenchem o form com presets.
- Painel "Debugger" mostrando o payload JSON enviado, o status HTTP da resposta, o tempo decorrido e o JSON de resposta.
- Barra superior com indicador `Model: mlflow:models:/...` lido em tempo real de `/health`, em verde se a origem é o DagsHub e em amarelo se for o fallback local.
- Links rápidos para Swagger, /health, DagsHub repo e MLflow UI no DagsHub.

---

## Trocando de modelo

A escolha do algoritmo é controlada por `params.yaml`:

```yaml
train:
  model: knn          # knn | random_forest | xgboost
```

Cada algoritmo tem sua própria seção de hiperparâmetros (`knn:`, `random_forest:`, `xgboost:`) que pode ser ajustada independentemente. Depois de trocar, basta rodar `make train`.

Resultados aproximados no dataset atual (1000 linhas, seed=42, mesmo split estratificado 80/20):

| Modelo | Accuracy | F1 (weighted) |
|---|---|---|
| KNN (n=7) | 0.5350 | 0.5334 |
| Random Forest (n_estimators=100) | 0.5350 | 0.5265 |
| XGBoost (default params) | 0.5050 | 0.4999 |

KNN está marginalmente à frente em F1. XGBoost provavelmente precisa de tuning para se destacar.

---

## Métricas via DVC

```bash
# Métricas da execução atual
.venv/bin/python -m dvc metrics show

# Diff entre métricas atuais e o último commit
.venv/bin/python -m dvc metrics diff
```

Para comparações visuais e por run, usar a UI do MLflow no DagsHub: https://dagshub.com/evaldocunhaf/MLOPs-Cesar.mlflow

---

## DagsHub

DagsHub hospeda um servidor MLflow gratuito para projetos públicos. Em vez de manter um MLflow local rodando em Docker, apontamos `tracking_uri` para o endpoint deles e:

- Todos os runs ficam acessíveis numa URL pública (compartilhável).
- O Model Registry funciona como qualquer MLflow padrão (`mlflow.sklearn.log_model(..., registered_model_name=...)`).
- A API consome o modelo com `mlflow.sklearn.load_model("models:/<nome>/<versão>")` autenticando via env vars.
- Não precisamos versionar o `.joblib` em git nem manter remote DVC adicional.

Acesso ao repositório: https://dagshub.com/evaldocunhaf/MLOPs-Cesar