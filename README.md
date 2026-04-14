# Gaming & Mental Health MLOps — DEV

Projeto de classificação de desempenho acadêmico/profissional (`High`, `Medium`, `Low`) com base em hábitos de jogos e sono, construído como um pipeline de Machine Learning com boas práticas de MLOps.

## 🚀 O que este projeto faz
Dado o perfil de um jogador (idade, gênero, horas de jogo, qualidade do sono, etc.), o modelo prediz o nível de performance acadêmica/trabalho do indivíduo utilizando algoritmos de classificação (XGBoost, Random Forest ou KNN).

### Pipeline:
**Kaggle Dataset** → **Limpeza e Mapeamento** → **One-Hot Encoding** → **Classificador** → **MLflow Tracker**

---

## 🛠️ Tecnologias Utilizadas

| Ferramenta | Para que serve |
|---|---|
| **Python 3.13** | Linguagem principal |
| **pandas / pyarrow** | Manipulação de dados e formato Parquet |
| **scikit-learn** | Pré-processamento e modelos de ML |
| **XGBoost** | Algoritmo de Gradient Boosting de alta performance |
| **kagglehub** | Download automático do dataset do Kaggle |
| **MLflow** | Rastreamento de experimentos (métricas, parâmetros, modelos) |
| **DVC** | Versionamento de dados e pipeline reproduzível |
| **FastAPI / Uvicorn** | API de inferência em tempo real |
| **Docker Compose** | Infraestrutura local para o MLflow |

---

## 📁 Estrutura do Projeto

```text
projeto-1/
│
├── extract/                    # Módulo de extração (Kaggle API)
├── transformer/                # Módulo de pré-processamento e limpeza
├── trainer/                    # Classes de treino e avaliação
│
├── notebooks/                  # Experimentação (EDA, testes de modelos)
│
├── data/
│   ├── raw/                    # Dados brutos (CSV) — Rastreado pelo DVC
│   └── processed/              # Dados limpos (Parquet/CSV) — Rastreado pelo DVC
│
├── models/                     # Modelos treinados (.joblib) — Rastreado pelo DVC
├── metrics/                    # Métricas do pipeline (metrics.json)
├── mlflow-docker/              # Configuração e volumes do MLflow (Docker)
│
├── extract.py                  # Script da Stage 1 (Extract)
├── transform.py                # Script da Stage 2 (Preprocess)
├── train.py                    # Script da Stage 3 (Train)
├── serve.py                    # API de inferência (FastAPI)
│
├── dvc.yaml                    # Definição das stages do pipeline DVC
├── params.yaml                 # Hiperparâmetros centralizados
├── requirements.txt            # Dependências fixadas
└── docker-compose.yml          # Orquestração do MLflow
```

---

## ⚙️ Configuração e Infraestrutura

### `params.yaml`
Arquivo central de hiperparâmetros. O DVC monitora este arquivo e reexecuta apenas as etapas afetadas por mudanças.
```yaml
train:
  model: random_forest  # opções: xgboost, random_forest, knn
```

### `dvc.yaml`
Define as etapas do pipeline. Garante que o treinamento só ocorra se os dados processados ou o código de treino mudarem.

---

## 🏃 Como Executar o Projeto

### 1. Preparar Ambiente
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Subir Infraestrutura (MLflow)
```powershell
docker compose up -d
```
*Acesse a UI em: [http://localhost:5050](http://localhost:5050)*

### 3. Executar o Pipeline
O DVC gerencia a execução automática na ordem correta:
```powershell
python -m dvc repro
```

### 4. Verificar Resultados
```powershell
# Ver métricas no terminal
python -m dvc metrics show

# Comparar com a execução anterior
python -m dvc metrics diff
```

### 5. Iniciar API de Inferência
```powershell
uvicorn serve:app --reload --port 8000
```
*Documentação interativa: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)*

---

## 📊 Resultados e Comparação
O projeto permite comparar modelos trocando o `model` no `params.yaml`. Atualmente, o **Random Forest** apresenta melhoria de ~3% em acurácia em relação ao XGBoost padrão neste dataset.

---

## 🧪 Como Experimentar e Trocar Modelos

O coração deste projeto é a experimentação rápida. Para testar um novo modelo:

1.  Abra o arquivo `params.yaml`.
2.  Localize a chave `train -> model` e altere para uma das opções:
    - `xgboost`
    - `random_forest`
    - `knn`
3.  (Opcional) Ajuste os hiperparâmetros nas seções específicas de cada modelo no `params.yaml`.
4.  No terminal, execute:
    ```powershell
    python -m dvc repro
    ```
    *O DVC detectará apenas a mudança nos parâmetros e rodará apenas a etapa de treino.*

---

## 📈 Visualizando e Comparando Métricas

Existem três formas de analisar os resultados do seu modelo:

### 1. Via Terminal (Métricas Atuais)
Para ver os resultados da última execução:
```powershell
python -m dvc metrics show
```

### 2. Comparação de Performance (Diff)
Se quiser comparar o modelo atual com a última versão que você deu `git commit`:
```powershell
python -m dvc metrics diff
```
*Isso mostrará exatamente quanto a acurácia ou o F1-score subiu ou desceu.*

### 3. Interface Gráfica (MLflow)
Abra [http://localhost:5050](http://localhost:5050) no seu navegador. Lá você pode:
- Ver gráficos de importância de features.
- Comparar múltiplas execuções lado a lado.
- Baixar o modelo (`.joblib`) gerado em cada run.

---

## 📊 Resultados Obtidos
Atualmente, o **Random Forest** apresenta os melhores resultados para este dataset (~53% de acurácia), superando o XGBoost básico.

**Próximos Passos Sugeridos:**
- Tuning automático de hiperparâmetros via Optuna.
- Adicionar novas features (como proporção de sono por horas de jogo).
- Implementar testes automatizados para validar o pré-processamento.
