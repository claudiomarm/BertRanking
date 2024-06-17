# Modelagem de Tópicos de Resumos de Projetos FAPESP

Este projeto tem como objetivo a modelagem de tópicos dos resumos de projetos financiados pela FAPESP. Os tópicos gerados serão utilizados para buscar no vocabulário controlado da USP os assuntos abordados em cada resumo.

## Estrutura do Projeto

O projeto é composto pelas seguintes etapas:

1. **Importação de Bibliotecas**
2. **Carregamento e Pré-processamento dos Dados**
3. **Divisão dos Dados em Treino e Teste**
4. **Tokenização e Criação de Embeddings com XLM-RoBERTa**
5. **Treinamento do Modelo XLM-RoBERTa**
6. **Extração de Embeddings**
7. **Treinamento do BERTopic**
8. **Treinamento do BERTopic com Embeddings de XLM-RoBERTa**

### 1. Importação de Bibliotecas

Nesta etapa, as bibliotecas necessárias para o projeto são importadas, incluindo bibliotecas para manipulação de dados, processamento de linguagem natural, treinamento de modelos e visualização.

### 2. Carregamento e Pré-processamento dos Dados

#### Definição da Raiz do Projeto

Definimos a raiz do projeto para facilitar o gerenciamento dos caminhos dos arquivos.

#### Extração dos Dados

Os dados dos projetos financiados pela FAPESP são extraídos de um arquivo Excel.

#### Filtragem dos Dados

Os dados são filtrados para garantir que apenas os resumos com assuntos não nulos e da área de Medicina sejam utilizados.

#### Carregamento do Modelo SpaCy

Carregamos o modelo de linguagem em português do SpaCy para facilitar o pré-processamento dos textos.

#### Limpeza do Texto

Os textos dos resumos são limpos, removendo caracteres especiais, stop words e realizando a lematização.

### 3. Divisão dos Dados em Treino e Teste

Os dados limpos são divididos em conjuntos de treino e teste para posterior treinamento e avaliação dos modelos.

### 4. Tokenização e Criação de Embeddings com XLM-RoBERTa

Utilizamos o tokenizer do XLM-RoBERTa para tokenizar os textos limpos e criar embeddings que serão utilizados no treinamento do modelo de classificação.

### 5. Treinamento do Modelo XLM-RoBERTa

#### Criação do Dataset Customizado

Criamos uma classe customizada para o Dataset, que facilita a manipulação dos dados durante o treinamento.

#### Configuração do Modelo XLM-RoBERTa

Carregamos o modelo XLM-RoBERTa pré-treinado e ajustamos a camada de classificação para multi-rótulo.

#### Congelamento e Descongelamento de Camadas

Congelamos todas as camadas do BERT e descongelamos as últimas 4 camadas para permitir o fine-tuning.

#### Definição dos Hiperparâmetros

Configuramos os hiperparâmetros, como taxa de aprendizado, batch size e número de épocas.

#### Treinamento e Avaliação

Treinamos o modelo XLM-RoBERTa e avaliamos seu desempenho utilizando métricas como acurácia, F1-score, recall e precisão.

### 6. Extração de Embeddings

Extraímos os embeddings dos textos utilizando o modelo XLM-RoBERTa treinado.

### 7. Treinamento do BERTopic

#### Configuração do BERTopic

Configuramos o modelo BERTopic com UMAP e CountVectorizer para realizar a modelagem de tópicos nos textos limpos.

#### Treinamento do BERTopic

Treinamos o modelo BERTopic nos textos limpos.

### 8. Treinamento do BERTopic com Embeddings de XLM-RoBERTa

#### Extração de Embeddings em Lotes

Utilizamos o modelo XLM-RoBERTa para extrair embeddings dos textos em lotes.

#### Treinamento do BERTopic com Embeddings

Treinamos o modelo BERTopic utilizando os embeddings extraídos dos textos.

## Dependências

As seguintes bibliotecas são necessárias para a execução do projeto:

- `os`
- `sys`
- `pandas`
- `numpy`
- `polars`
- `re`
- `spacy`
- `sklearn`
- `bertopic`
- `transformers`
- `torch`
- `umap`

## Resultados Esperados

Ao final da execução deste projeto, espera-se obter:

### Modelos de Tópicos

- Modelo BERTopic treinado sem embeddings.
- Modelo BERTopic treinado com embeddings de XLM-RoBERTa.

### Embeddings

- Embeddings dos textos extraídos utilizando o modelo XLM-RoBERTa.

### Métricas de Avaliação

- Acurácia, F1-score, recall e precisão do modelo XLM