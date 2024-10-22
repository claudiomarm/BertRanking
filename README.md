---

# BertRanking

Este projeto implementa um sistema de ranqueamento semântico de projetos científicos utilizando modelos de linguagem BERT. Foram comparados o BERT multilíngue e o BERTimbau, especializado em português, com o objetivo de identificar qual modelo apresenta melhor desempenho em tarefas de similaridade semântica.

## Estrutura do Projeto

O projeto é implementado em Python 3.7+ e utiliza as seguintes bibliotecas principais:
- `transformers` (Hugging Face)
- `datasets`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `plotly`

### Métodos Principais da Classe `BertRanking`:

1. **`import_data()`**: Importa os dados de projetos científicos da FAPESP a partir de um arquivo Excel.
2. **`prepare_data()`**: Prepara os dados de entrada, filtrando colunas, convertendo datas e limpando textos.
3. **`tokenize_function()`**: Tokeniza os textos utilizando o `BertTokenizer`.
4. **`train_model()`**: Treina os modelos BERT com base nos resumos dos projetos científicos.
5. **`get_embeddings()`**: Gera embeddings para os textos e palavras-chave.
6. **`calculate_similarities()`**: Calcula as similaridades entre os textos e queries.
7. **`rank_texts()`**: Rankeia os textos com base na similaridade de cosseno entre os embeddings dos textos e das queries.
8. **`visualize_results_rank()`**: Executa a visualização dos resultados de similaridade com base no ranqueamento de textos, gerando gráficos de dispersão, distribuição, boxplots e sumários estatísticos para comparar os modelos BERT e BERTimbau.

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

## Resultados

Os resultados indicam que o BERT multilíngue apresentou maior consistência nos valores de similaridade semântica, enquanto o BERTimbau demonstrou maior variação. Ambos os modelos são capazes de identificar os resumos mais relevantes, mas o BERT multilíngue se destacou por sua robustez em contextos variados.

## Como Executar

1. **Instalar as dependências**: 
   ```
   poetry install
   ```

2. **Executar a pipeline**:
   ```
   python src/nlp_pipeline/bertranking.py
   ```
--- 