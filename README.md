### README - Projeto BertRanking

---

## Visão Geral

O **BertRanking** compara modelos de linguagem, como **BERTimbau** e **RoBERTa**, para ranquear tópicos de artigos científicos. Utiliza embeddings de cada modelo para medir a relevância dos tópicos com base na similaridade semântica.

---

## Estrutura do Projeto

- **Classe BertRanking**: Carrega, limpa, tokeniza, treina os modelos, gera embeddings e ranqueia tópicos.
- **Processamento de Dados**: Limpeza e preparação de dados de artigos científicos.
- **Comparação de Modelos**: Avalia os modelos com base na similaridade semântica.

---

## Funcionalidades

- **Importação e Preprocessamento**: Carrega e limpa dados do Excel.
- **Tokenização**: Tokeniza o texto com os tokenizadores do BERT e RoBERTa.
- **Treinamento**: Treina modelos com MLM (Masked Language Modeling).
- **Geração de Embeddings**: Gera embeddings de textos e tópicos.
- **Ranqueamento de Tópicos**: Ranqueia tópicos com base na similaridade de cosseno.
- **Avaliação**: Compara modelos com precisão, recall, NDCG e similaridade semântica.

---

## Requisitos

- Python 3.7+
- Instalar dependências:

```bash
poetry install
```

---

## Execução

1. **Configurar Caminhos**: Definir os caminhos de dados, modelos e saídas.
2. **Treinar Modelos**: Caso não estejam treinados, os modelos serão treinados.
3. **Gerar Embeddings**: Embeddings dos textos e tópicos serão gerados.
4. **Ranquear Tópicos**: Tópicos serão ranqueados pela similaridade.
5. **Rodar o Projeto**:

```bash
python caminho_para_o_script.py
```

---

## Estrutura de Arquivos

```
BertRanking/
├───data
│   ├───internal
│   │   ├───fapesp_forms
│   │   └───fapesp_projects
│   └───processed
│       └───fapesp_projects
├───models
│   ├───BERTimbau
│   │   └───results
│   └───RoBERTa
│       └───results
├───notebooks
├───src
│   ├───etl
│   │   └───__pycache__
│   ├───nlp_pipeline
│   ├───util
│   └───visualization
│       ├───dashboard
│       └───static
└───tokenizers
    ├───BERTimbau
    └───RoBERTa
```

---

## Métodos Principais

- **`import_data`**: Carrega os dados de artigos.
- **`clean_text`**: Limpa e prepara os textos.
- **`tokenize_function`**: Tokeniza textos.
- **`train_model`**: Treina ou ajusta os modelos.
- **`get_embeddings`**: Gera embeddings de textos e tópicos.
- **`rank_topics_by_relevance`**: Ranqueia tópicos por relevância.
- **`compare_models_by_semantic_similarity`**: Compara similaridade entre os modelos.

---

## Resultados

- **Similaridade de Cosseno**: Média e desvio padrão entre textos e tópicos.
- **Ranqueamento de Tópicos**: Métricas de precisão, recall e NDCG (caso haja validação).
- **Visualização**: Gráficos de similaridade para comparação entre os modelos.

---

## Autor

- Desenvolvido por [Claudiomar Mendes]. 

--- 

Este README apresenta o projeto **BertRanking**, que ranqueia tópicos de artigos científicos usando os modelos **BERTimbau** e **RoBERTa**.