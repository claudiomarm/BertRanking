---

# BertRanking
![IMG_5773](https://github.com/user-attachments/assets/2ec70286-5b8f-40b8-b806-d6930c55fe6f)

This project implements a semantic ranking system for scientific projects using BERT language models. The multilingual BERT and BERTimbau, specialized in Portuguese, were compared to identify which model performs better in semantic similarity tasks.

## Project Structure

The project is implemented in Python 3.7+ and uses the following key libraries:
- `transformers` (Hugging Face)
- `datasets`
- `torch`
- `scikit-learn`
- `pandas`
- `numpy`
- `plotly`

### Main Methods of the `BertRanking` Class:

1. **`import_data()`**: Imports FAPESP scientific project data from an Excel file.
2. **`prepare_data()`**: Prepares the input data by filtering columns, converting dates, and cleaning text.
3. **`tokenize_function()`**: Tokenizes the texts using `BertTokenizer`.
4. **`train_model()`**: Trains the BERT models based on the project summaries.
5. **`get_embeddings()`**: Generates embeddings for texts and keywords.
6. **`calculate_similarities()`**: Calculates the similarities between the texts and queries.
7. **`rank_texts()`**: Ranks the texts based on the cosine similarity between the text and query embeddings.
8. **`visualize_results_rank()`**: Visualizes similarity results based on text ranking, generating scatter plots, distributions, boxplots, and statistical summaries to compare BERT and BERTimbau models.

## File Structure

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

## Results

The results indicate that the multilingual BERT showed greater consistency in semantic similarity values, while BERTimbau demonstrated more variation. Both models are capable of identifying the most relevant summaries, but the multilingual BERT stood out for its robustness in diverse contexts.

## How to Run

1. **Install dependencies**:
   ```
   poetry install
   ```

2. **Run the pipeline**:
   ```
   python src/nlp_pipeline/bertranking.py
   ```
---
