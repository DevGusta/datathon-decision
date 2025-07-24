# 🤖 Datathon Decision - Recrutamento Inteligente

## 🎯 Objetivo do Projeto
Este projeto demonstra como treinar e disponibilizar um modelo de Machine Learning para apoiar o time de recrutamento da Decision. A aplicação foi pensada para prever o "match" de candidatos(as) com as vagas abertas utilizando uma API em FastAPI.

## Visão geral
- Leitura dos JSONs em `data_source/` para montar um dataset unindo candidatos e vagas;
- Endpoint `/predict` que recebe características de um candidato e retorna a probabilidade de aderência à vaga
- Registro das previsões realizadas em `predictions.log`

## 🚀 Estrutura do Projeto
```
.
├── data_source/                # Dados de exemplo em JSON
│   ├── applicants.json
│   ├── prospects.json
│   ├── vagas.json
├── data/
│   └── features.csv            # Gerado após o treino
├── notebooks/
│   └── treinamento_modelo.ipynb
├── src/                        # Código fonte
│   ├── app.py                  # API FastAPI
│   ├── data_loader.py          # Utilitários de leitura
│   ├── feature_store.py        # Armazenamento das features
│   ├── dashboard.py            # Painel de monitoramento
│   ├── model.py                # Funções de modelagem
│   └── monitor.py              # Registro das previsões
├── tests/                      # Testes unitários
│   ├── test_api.py
│   ├── test_feature_store.py
│   └── test_model.py
├── model.joblib                # Modelo treinado
├── predictions.log             # Log de previsões
├── requirements.txt
├── Dockerfile
├── pytest.ini
└── README.md
```

## ⚙️ Como Rodar o Projeto (Docker)
Certifique-se de ter o Docker instalado e execute na raiz do projeto:
```bash
docker build -t decision-api .
docker run -p 8000:8000 decision-api
```
- **API disponível em:** `http://localhost:8000`

## 🛠️ Tecnologias Utilizadas
- Python 3.12
- FastAPI
- scikit-learn
- Pandas
- Docker
- pytest

## 🏋️‍♂️ Treinamento
O notebook `notebooks/treinamento_modelo.ipynb` carrega os JSONs de candidatos, vagas e prospects em `data_source/`. Essas informacoes sao combinadas para criar features como nivel da vaga, ingles exigido e diferenca entre titulos. O modelo treinado é salvo como `model.joblib` na raiz do projeto.

## 📊 Sobre o Modelo de ML
- **Modelo:** Regressão Logística com `scikit-learn`.
- **Pipeline de treino:**
  1. **Feature engineering** cria atributos derivados do texto e valores numéricos extraídos dos JSONs.
  2. **Pré-processamento** com `StandardScaler`.
- **Validação (dados de exemplo):** acurácia ~0.90.
- **Modelo salvo em:** `model.joblib`, carregado automaticamente pela API.

## Testes
Execute os testes automatizados com:
```bash
pytest -q
```
