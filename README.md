# 📄 Documentação do Projeto: PneumoAI

## 1. Visão Geral

**Componentes:**

* Eunice Cristina de Araújo Silva
* João Lucas Gomes de Souza
* René Rufino de Figueiredo Junior

**Tecnologias Utilizadas:**

* Python
* FastAPI
* Uvicorn
* PyTorch
* `efficientnet_pytorch`
* EfficientNet

**Descrição:**
API de análise de imagens médicas com foco na detecção de pneumonia e classificação do tipo (viral, bacteriana, etc.) a partir de imagens de raio-x.

**Objetivo:**
Detectar anomalias ou características em imagens médicas utilizando Inteligência Artificial, auxiliando profissionais da saúde no diagnóstico precoce da pneumonia.


## 2. Descrição Detalhada do Projeto

**O que é o projeto?**
O projeto consiste no desenvolvimento de uma API baseada em Inteligência Artificial, cujo objetivo é auxiliar médicos e outros profissionais da área da saúde na análise de imagens médicas — como radiografias do tórax — para detectar sinais de pneumonia. A solução visa identificar não apenas a presença da doença, mas também o tipo específico de pneumonia (bacteriana, viral, etc.).

### 2.1 Funcionalidades Principais

| **Funcionalidade** | **Rota**        | **Descrição**                                                     |
| ------------------ | --------------- | ----------------------------------------------------------------- |
| Upload de imagem   | `POST /upload`  | Recebe o arquivo da imagem para análise posterior.                |
| Diagnóstico        | `POST /predict` | Retorna um diagnóstico sobre a presença ou ausência de pneumonia. |
| Tipo da doença     | `POST /predict` | Caso positivo, retorna o tipo: viral, bacteriana, etc.            |
| Resultado completo | `POST /predict` | Retorna uma resposta em JSON com o resultado da inferência.       |
| Health check       | `GET /health`   | Retorna status 200 e mensagem indicando que a API está online.    |

> 🔁 As funcionalidades 2, 3 e 4 são unificadas na rota `/predict`, com respostas estruturadas.



### 2.2 Arquitetura do Código


PneumoAI/
├── main.py               # Lógica principal da API
├── train.py              # Script de treinamento do modelo IA
├── dataset_pneumonia/    # Base de dados utilizada no treinamento
├── training/
│   ├── PNEUMONIA/        # Imagens com diagnóstico de pneumonia (treinamento)
│   └── NORMAL/           # Imagens normais (treinamento)
└── validate/
    ├── PNEUMONIA/        # Imagens com pneumonia (validação)
    └── NORMAL/           # Imagens normais (validação)


## 3. Etapas de Entrega (Cronograma Detalhado)

| **Etapa** | **Data**   | **Descrição**                                                           |
| --------- | ---------- | ----------------------------------------------------------------------- |
| Etapa 1   | 17/05/2025 | Levantamento de requisitos e definição do escopo do projeto.            |
| Etapa 2   | 24/05/2025 | Configuração do ambiente de desenvolvimento e estrutura inicial da API. |
| Etapa 3   | 08/06/2025 | Implementação das rotas principais da API e integração com o modelo IA. |
| Etapa 4   | 21/06/2025 | Testes com dados reais e ajustes no modelo e na API.                    |
| Etapa 5   | 19/07/2025 | Implementação de testes, documentação e refinamento do projeto.         |
| Etapa 6   | 27/07/2025 | Entrega final com deploy, relatório e apresentação do projeto.          |

