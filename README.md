### README - Projeto BertRanking

---

## Visão Geral

O **BertRanking** é um sistema projetado para comparar modelos de linguagem como **BERTimbau** e **BERT** utilizando embeddings gerados a partir dos textos e tópicos de artigos científicos. O sistema permite o ranqueamento de artigos com base na similaridade semântica entre um artigo de consulta e os artigos da base.

---

## Estrutura do Projeto

- **Classe BertRanking**: Realiza a importação, preparação e tokenização dos dados, treina os modelos, gera embeddings e ranqueia artigos com base em similaridade semântica.
- **Processamento de Dados**: Realiza a limpeza, padronização e preparação dos textos e tópicos de artigos científicos.
- **Comparação de Modelos**: Gera embeddings usando diferentes modelos de linguagem e compara seus desempenhos com base na similaridade de cosseno.

---

## Funcionalidades

- **Importação de Dados**: Carrega e prepara dados de artigos científicos a partir de um arquivo Excel.
- **Tokenização**: Converte textos em tokens utilizando modelos BERT ou BERTimbau.
- **Treinamento de Modelos**: Permite o treinamento de modelos usando **Masked Language Modeling (MLM)**.
- **Geração de Embeddings**: Gera representações vetoriais (embeddings) para textos e tópicos.
- **Ranqueamento de Artigos**: Ranqueia artigos com base na similaridade de cosseno entre o texto de consulta e os artigos da base.
- **Comparação de Modelos**: Compara a performance dos modelos com base na similaridade semântica calculada.

---

## Requisitos

- **Python 3.7+**
- Instalar dependências:

```bash
poetry install
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
│   └───BERT
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
    └───BERT
```

---

## Métodos Principais

- **`import_data`**: Carrega os dados de artigos a partir de um arquivo Excel.
- **`clean_text`**: Realiza a limpeza dos textos, removendo caracteres especiais e padronizando o conteúdo.
- **`tokenize_function`**: Aplica a tokenização dos textos utilizando os modelos de linguagem.
- **`train_model`**: Treina ou ajusta os modelos de linguagem utilizando **Masked Language Modeling**.
- **`get_embeddings`**: Gera embeddings dos textos e tópicos utilizando os modelos BERT ou BERTimbau.
- **`rank_texts`**: Ranqueia artigos da base com base na similaridade semântica do artigo de consulta.
- **`compare_models`**: Compara a performance dos modelos com base na similaridade semântica calculada entre os embeddings.

---

## Resultados

- **Similaridade Semântica**: Métricas de similaridade entre o artigo de consulta e os artigos da base, utilizando a média e desvio padrão das similaridades.
- **Ranqueamento de Artigos**: Lista de artigos ranqueados pela similaridade semântica com o artigo de consulta.
- **Comparação de Modelos**: Compara os resultados entre os modelos BERT e BERTimbau com base na similaridade semântica.

---

## Autor

Desenvolvido por **Claudiomar Mendes** como parte de um projeto para ranqueamento de artigos científicos utilizando técnicas de **Processamento de Linguagem Natural (NLP)**.

---

Este README documenta o projeto **BertRanking**, que ranqueia artigos científicos usando os modelos **BERTimbau** e **BERT**, com foco em comparação de similaridade semântica.