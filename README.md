# Extração de Termos Representativos em Resumos de Artigos Científicos

## Descrição
Este projeto utiliza técnicas de Processamento de Linguagem Natural (PLN) e Modelagem de Tópicos para extrair termos representativos de resumos de artigos científicos.

## Estrutura

1. **Coleta e Preparação dos Dados**
   - Coletar e estruturar resumos de artigos científicos.

2. **Pré-processamento dos Textos**
   - Limpar e normalizar textos usando `nltk` ou `spaCy`.

3. **Modelagem de Tópicos com LDA**
   - Vetorizar textos, aplicar LDA e extrair termos principais.

4. **Geração de Embeddings com BERT (DistilBERT)**
   - Carregar modelo BERT, tokenizar textos e gerar embeddings ([CLS]).

5. **Extração de Termos Representativos com BERT**
   - Calcular embeddings médios, medir similaridade e selecionar palavras representativas.

6. **Combinação dos Resultados de LDA e BERT**
   - Combinar termos de LDA e BERT, removendo duplicados.

7. **Análise e Validação dos Resultados**
   - Revisar termos, ajustar parâmetros e validar resultados.

8. **Implementação de Visualizações**
   - Criar gráficos e dashboards interativos com Plotly e Dash.

## Tecnologias

- **Linguagem:** Python
- **Bibliotecas:** `nltk`, `spaCy`, `sklearn`, `gensim`, `transformers`, `polars`, `Plotly`, `Dash`
- **Web:** Flask ou Django