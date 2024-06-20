# Projeto de Modelagem de Tópicos de Resumos de Projetos Científicos Financiados pela FAPESP

## Visão Geral

Este projeto tem como objetivo realizar a modelagem de tópicos em resumos de projetos científicos financiados pela FAPESP. O intuito é mapear os assuntos abordados nos textos com base nos tópicos encontrados pela modelagem de tópicos. Esses tópicos serão utilizados para buscar os assuntos em um dicionário controlado da USP. 

Para atingir esse objetivo, serão comparados dois modelos de BERTopic:
1. Um BERTopic padrão ajustado aos dados.
2. Um BERTopic com embeddings de um BERTimbau fine-tunado utilizando uma tarefa de Masked Language Modeling (MLM).

## Estrutura do Projeto

### 1. Preparação dos Dados

Os dados são extraídos de um arquivo Excel contendo informações sobre os projetos financiados. As principais variáveis extraídas incluem:
- Número do Processo
- Data de Início
- Título (Português)
- Grande Área do Conhecimento
- Área do Conhecimento
- Subárea do Conhecimento
- Palavras-Chave do Processo
- Assuntos
- Resumo (Português)

Os dados são pré-processados e filtrados para garantir a qualidade dos textos que serão utilizados na modelagem de tópicos.

### 2. Limpeza e Pré-Processamento de Texto

Utilizamos a biblioteca spaCy para realizar a limpeza e pré-processamento dos textos, que inclui:
- Remoção de caracteres especiais
- Tokenização e remoção de stop words
- Lematização dos textos

### 3. Treinamento do Modelo BERTimbau para MLM

O modelo BERTimbau é fine-tunado utilizando uma tarefa de Masked Language Modeling (MLM). O conjunto de dados é tokenizado e dividido em conjuntos de treino e teste para realizar o treinamento do modelo.

### 4. Extração de Embeddings

As embeddings dos textos são extraídas utilizando o modelo BERTimbau fine-tunado. Essas embeddings serão utilizadas posteriormente na modelagem de tópicos.

### 5. Modelagem de Tópicos com BERTopic

A modelagem de tópicos é realizada utilizando a biblioteca BERTopic. São treinados dois modelos:
1. BERTopic padrão utilizando os textos limpos.
2. BERTopic utilizando embeddings extraídas pelo BERTimbau fine-tunado.

### 6. Visualização e Análise

Os tópicos encontrados são visualizados e analisados para entender melhor os assuntos abordados nos resumos dos projetos científicos.

## Estrutura de Diretórios

- `data/`: Contém os dados utilizados no projeto.
- `models/`: Contém os modelos treinados.
- `tokenizers/`: Contém os tokenizers utilizados.
- `results/`: Contém os resultados das avaliações dos modelos.

## Requisitos

Para executar este projeto, você precisará das seguintes bibliotecas:
- pandas
- numpy
- polars
- spacy
- sklearn
- bertopic
- transformers
- torch
- datasets
- umap-learn
- plotly
- wordcloud
- matplotlib

## Como Executar

1. **Preparação do Ambiente**: Instale todas as bibliotecas necessárias.
2. **Extração e Pré-Processamento dos Dados**: Carregue e pré-processe os dados conforme descrito.
3. **Treinamento do Modelo BERTimbau**: Fine-tune o modelo BERTimbau utilizando MLM.
4. **Extração de Embeddings**: Extraia as embeddings dos textos limpos.
5. **Modelagem de Tópicos**: Treine os modelos BERTopic e visualize os tópicos encontrados.
6. **Análise dos Resultados**: Compare os tópicos encontrados e analise os resultados.