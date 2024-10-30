# Importação de Bibliotecas
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertModel
import torch
from datasets import Dataset
import time
import re
import logging
import logging.config
from functools import reduce
import locale
import plotly.graph_objects as go
import warnings
import random

warnings.filterwarnings('ignore')

# Definicao da raiz do projeto
current_file_path = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(current_file_path, '..', '..'))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

class BertRanking():
    def __init__(self, file_path, dict_models, processed_data_path, area='Medicina'):
        self.file_path = file_path
        self.dict_models = dict_models
        self.processed_data_path = processed_data_path
        self.area = area
        
        self.variables = {
            'N. Processo_B.V': 'n_processo',
            'Data de Início': 'data',
            'Título (Português)': 'titulo',
            'Grande Área do Conhecimento': 'grande_area',
            'Área do Conhecimento': 'area',
            'Subárea do Conhecimento': 'subarea',
            'Palavras-Chave do Processo': 'palavras_chave',
            'Assuntos': 'assuntos',
            'Resumo (Português)': 'resumo'}
        
        self.charts_path = os.path.join(PROJECT_ROOT, 'charts')

        self.fig_config = {
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': 'BertRanking',
                'height': 450,
                'width': 1500,
                'scale': 6
            },
            'displayModeBar': True,
            'displaylogo': False,
        }
        
    def import_data(self, sheet_name='Sheet1'):
        try:
            logging.info('Importando dados da planilha...')

            return pd.read_excel(self.file_path, sheet_name=sheet_name)
        
        except Exception as e:
            logging.error(f'Erro ao importar dados: {str(e)}')
    
    def save_dataset(self, dataset, path):
        try:
            logging.info('Salvando dataset...')

            df = dataset.to_pandas()

            df.to_parquet(path, index=False)
        
        except Exception as e:
            logging.error(f'Erro ao salvar dataset: {str(e)}')

    def load_dataset(self, path, load_as='datasets'):
        try:
            logging.info('Carregando dataset salvo...')

            df = pd.read_parquet(path)

            if load_as == 'pandas':
                return df
            elif load_as == 'datasets':
                return Dataset.from_pandas(df)
            else:
                raise ValueError("O argumento 'load_as' deve ser 'pandas' ou 'datasets'.")
        
        except Exception as e:
            logging.error(f'Erro ao carregar dataset: {str(e)}')
    
    def clean_text(self, text):
        try:
            if not isinstance(text, str):
                raise ValueError("O argumento 'text' deve ser uma string.")
            
            text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s]', '', text)

            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        except Exception as e:
            logging.error(f'Erro ao limpar texto: {str(e)}')

    def prepare_data(self, data):
        try:
            logging.info('Preparando dados...')

            full_data = data.rename(columns=self.variables)

            full_data = full_data[list(self.variables.values())]

            full_data = full_data[
                full_data['n_processo'].notnull() & full_data['resumo'].notnull() & (full_data['resumo'] != '')
            ]

            full_data['data'] = pd.to_datetime(full_data['data'], format='%m-%d-%y', errors='coerce')

            full_data['ano'] = full_data['data'].dt.year
            full_data['mes'] = full_data['data'].dt.month

            full_data = full_data.drop(columns=['data'])

            data_train_test = full_data[full_data['assuntos'].notnull() & full_data['palavras_chave'].notnull() & (full_data['area'] == self.area)]

            data_train_test['titulo'] = data_train_test['titulo'].astype(str)
            data_train_test['palavras_chave'] = data_train_test['palavras_chave'].astype(str)
            
            data_train_test['keywords'] = data_train_test['palavras_chave'].apply(self.clean_text)
            data_train_test['cleaned_text'] = data_train_test['titulo'].apply(self.clean_text) + '. ' + data_train_test['resumo'].apply(self.clean_text) + '. Palavras-chave: ' + data_train_test['keywords']
            
            return data_train_test
        
        except Exception as e:
            logging.error(f'Erro ao preparar dados: {str(e)}')

    def tokenize_function(self, dataset, tokenizer, col_text='cleaned_text'):
        try:
            logging.info('Tokenizando textos...')
            
            return tokenizer(dataset[col_text], padding="max_length", truncation=True, max_length=512)
        
        except Exception as e:
            logging.error(f'Erro na tokenização: {str(e)}')

    def train_model(self, bert_model, tokenizer, train_dataset, test_dataset, data_collator, model_path, tokenizer_path, output_dir, overwrite_output_dir=True, save_steps=10_000, save_total_limit=2, prediction_loss_only=True, num_train_epochs=3, per_device_train_batch_size=8):
        try:
            logging.info('Treinando o modelo...') 
        
            os.makedirs(model_path, exist_ok=True)
            os.makedirs(tokenizer_path, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=overwrite_output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                prediction_loss_only=prediction_loss_only,
            )

            trainer = Trainer(
                model=bert_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )

            trainer.train()

            if model_path:
                trainer.save_model(model_path)
            
            if tokenizer_path:
                tokenizer.save_pretrained(tokenizer_path)
            
            return trainer
        
        except Exception as e:
            logging.error(f'Erro no treinamento do modelo: {str(e)}')

    def evaluate_model(self, trainer, test_dataset):
        try:
            logging.info('Avaliando o modelo...')

            eval_results = trainer.evaluate(eval_dataset=test_dataset)
            loss = eval_results['eval_loss']
            perplexity = np.exp(loss)
            
            metrics = {
                'loss': loss,
                'perplexity': perplexity
            }
            
            return metrics
    
        except Exception as e:
                logging.error(f'Erro na avaliação do modelo: {str(e)}')

    def get_embeddings(self, texts, tokenizer, bert_model, max_length=512, batch_size=8):
        try:
            logging.info('Gerando embeddings...')

            if not hasattr(bert_model, "is_on_device"):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                bert_model.to(device)
                bert_model.is_on_device = True

            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(bert_model.device)

                with torch.no_grad():
                    outputs = bert_model(**inputs, output_hidden_states=True)

                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]

                batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(batch_embeddings)

            all_embeddings = np.vstack(all_embeddings)

            return all_embeddings

        except Exception as e:
            logging.error(f'Erro ao gerar embeddings: {str(e)}')

    def generate_embeddings(self, dataset, tokenizer, bert_model, text_col='cleaned_text', keywords_col=None, batch_size=8, model_name='bertimbal'):
        try:
            logging.info('Gerando embeddings do dataset...')

            texts = dataset[text_col]
            dataset[f'text_embedding_{model_name}'] = self.get_embeddings(texts=texts, tokenizer=tokenizer, bert_model=bert_model, batch_size=batch_size)
            
            if keywords_col:
                all_keywords = dataset[keywords_col]
                all_keywords_embeddings = []
                for subjects in all_keywords:
                    subjects_embeddings = self.get_embeddings(texts=subjects, tokenizer=tokenizer, bert_model=bert_model, batch_size=batch_size)
                    all_keywords_embeddings.append(subjects_embeddings)
                
                dataset[f'queries_embeddings_{model_name}'] = all_keywords_embeddings

            return dataset
        
        except Exception as e:
            logging.error(f'Erro ao gerar embeddings do dataset: {str(e)}')

    def get_query_embedding(self, query_text, tokenizer, bert_model):
        query_text = self.clean_text(query_text)
        query_embedding = self.get_embeddings([query_text], tokenizer, bert_model)

        return query_embedding
    
    def rank_similarity(self, query_embedding, texts_embeddings):
        similarities = cosine_similarity(query_embedding, texts_embeddings)

        return np.argsort(similarities[0])[::-1], similarities[0]
    
    def rank_texts(self, query_text, tokenizer=None, bert_model=None, data=None, text_embedding_col='text_embedding_BERT', model_name='BERT', top_n=5, return_col='titulo', show_info=True):
        try:
            logging.info(f'Ranqueando textos para o modelo {model_name}...')
            if data is None:
                data = self.embeddings_test_dataset
            
            if tokenizer is None or bert_model is None:
                tokenizer, bert_model = self.model_dict[model_name]['tokenizer'], self.model_dict[model_name]['model']

            query_embedding = self.get_query_embedding(query_text, tokenizer, bert_model)
            
            texts_embeddings = np.vstack(data[text_embedding_col].values)
            
            ranked_indices, similarities = self.rank_similarity(query_embedding, texts_embeddings)
            
            ranked_indices = ranked_indices[:top_n]
            ranked_similarities = similarities[ranked_indices]
            
            ranked_values = data.iloc[ranked_indices][return_col].values
            
            if show_info:
                for i, (value, sim) in enumerate(zip(ranked_values, ranked_similarities)):
                    print(f"Rank {i+1}: Similaridade {sim:.4f}")
                    print(f"Valor da coluna '{return_col}': {value}\n")
            
            return ranked_values, ranked_similarities
        
        except Exception as e:
            logging.error(f'Erro ao ranquear textos: {str(e)}')
    
    def add_similarity_column(self, data=None, text_embedding_col='text_embedding_BERT', query_embedding_col='queries_embeddings_BERT', similarity_col='similarity_text_query_BERT'):
        try:
            logging.info(f'Adicionando coluna de similaridade "{similarity_col}"...')

            if data is None:
                data = self.embeddings_test_dataset
            
            def calculate_similarity(text_emb, queries_emb_list):
                queries_emb = np.vstack(queries_emb_list)
                similarities = cosine_similarity([text_emb], queries_emb)[0] 
                return similarities.mean()

            data[similarity_col] = data.apply(
                lambda row: calculate_similarity(row[text_embedding_col], row[query_embedding_col]), axis=1
            )

            logging.info(f'Coluna "{similarity_col}" adicionada com sucesso.')

            return data
            
        except Exception as e:
            logging.error(f'Erro ao calcular a similaridade: {str(e)}')

    def generate_random_queries(self, data=None, n_queries=30, text_col='cleaned_text'):
        try:
            logging.info(f'Gerando {n_queries} queries aleatórias...')
            if data is None:
                data = self.embeddings_test_dataset

            random.seed(42)
            return random.sample(list(data[text_col]), n_queries)
        except Exception as e:
            logging.error(f'Erro ao gerar queries aleatórias: {str(e)}')

    def get_embeddings_queries(self, queries, tokenizer=None, bert_model=None, data=None, text_embedding_col='text_embedding_BERT', model_name='BERT', top_n=5, return_col='titulo', show_info=False):
        try:
            if data is None:
                data = self.embeddings_test_dataset
            
            if tokenizer is None or bert_model is None:
                tokenizer, bert_model = self.model_dict[model_name]['tokenizer'], self.model_dict[model_name]['model']
            
            top_n_embeddings = {i: [] for i in range(1, top_n+1)}
            queries_embeddings = []

            for query in queries:
                query_embedding = self.get_query_embedding(query, tokenizer, bert_model)
                queries_embeddings.append(query_embedding[0])

                ranked_values, _ = self.rank_texts(query_text=query, tokenizer=tokenizer, bert_model=bert_model, data=data, text_embedding_col=text_embedding_col, model_name=model_name, top_n=top_n, return_col=return_col, show_info=show_info)
                
                for i in range(top_n):
                    embedding = data.loc[data[return_col] == ranked_values[i], text_embedding_col].values[0]
                    top_n_embeddings[i+1].append(embedding)
            
            return top_n_embeddings, queries_embeddings
        
        except Exception as e:
            logging.error(f'Erro ao obter embeddings do top-n: {str(e)}')
    
    def calculate_similarities(self, top_n_embeddings, queries_embeddings, top_n=5):
        try:
            similarities_dict = {rank: [] for rank in range(1, top_n+1)}
            
            for rank in range(1, top_n+1):
                for i, doc_embedding in enumerate(top_n_embeddings[rank]):
                    query_embedding = queries_embeddings[i]
                    
                    similarity = cosine_similarity([doc_embedding], [query_embedding])[0][0]
                    similarities_dict[rank].append(similarity)
            
            return similarities_dict
        except Exception as e:
            logging.error(f'Erro ao calcular as similaridades: {str(e)}')

    def aggregate_similarities(self, similarities_dict, top_n=5):
        try:
            aggregated_similarities = []
            
            for rank in range(1, top_n+1):
                aggregated_similarities.extend(similarities_dict[rank])
            
            return aggregated_similarities
        except Exception as e:
            logging.error(f'Erro ao agregar as similaridades: {str(e)}')

    def plot_scatter_similarity(self, title, aggregated_similarities_dict):
        try:
            fig = go.Figure()

            for model_name, aggregated_similarities in aggregated_similarities_dict.items():
                fig.add_trace(go.Scatter(
                    x=list(range(len(aggregated_similarities))),
                    y=aggregated_similarities,
                    mode='markers',
                    name=model_name
                ))

            # Layout do gráfico
            fig.update_layout(
                title=title,
                xaxis_title='Amostras',
                yaxis_title='Similaridade',
                showlegend=True
            )
            
            fig.show(config=self.fig_config)

        except Exception as e:
            logging.error(f'Erro ao plotar dispersão: {str(e)}')

    def plot_similarity_distribution(self, title, aggregated_similarities_dict):
        try:
            fig = go.Figure()

            for model_name, aggregated_similarities in aggregated_similarities_dict.items():
                fig.add_trace(go.Histogram(
                    x=aggregated_similarities,
                    name=model_name,
                    nbinsx=20,
                    opacity=0.75
                ))

            fig.update_layout(
                title=title,
                xaxis_title='Similaridade Semântica',
                yaxis_title='Frequência',
                barmode='overlay',
                showlegend=True
            )

            fig.update_traces(opacity=0.6)
            
            fig.show(config=self.fig_config)

        except Exception as e:
            logging.error(f'Erro ao plotar distribuição: {str(e)}')

    def plot_boxplot_similarity(self, title, aggregated_similarities_dict):
        try:
            fig = go.Figure()

            for model_name, aggregated_similarities in aggregated_similarities_dict.items():
                fig.add_trace(go.Box(
                    y=aggregated_similarities,
                    name=model_name
                ))

            fig.update_layout(
                title=title,
                yaxis_title='Similaridade'
            )
            
            fig.show(config=self.fig_config)

        except Exception as e:
            logging.error(f'Erro ao plotar boxplot: {str(e)}')
    
    def plot_statistical_summary(self, title, aggregated_similarities_dict):
        try:
            summaries = {}

            metric_rename = {
                'count': 'Contagem',
                'mean': 'Média',
                'std': 'Desvio Padrão',
                'min': 'Mínimo',
                '25%': '1º Quartil (25%)',
                '50%': 'Mediana (50%)',
                '75%': '3º Quartil (75%)',
                'max': 'Máximo'
            }

            for model_name, aggregated_similarities in aggregated_similarities_dict.items():
                summary = pd.Series(aggregated_similarities).describe()
            
                summary['count'] = int(summary['count'])

                for metric in summary.index:
                    if metric != 'count':
                        summary[metric] = f'{round(summary[metric], 4):.4f}'
                
                summaries[model_name] = summary
            
                summary.rename(index=metric_rename, inplace=True)

            metrics = summaries[next(iter(summaries))].index
            values = [summaries[model_name].values for model_name in aggregated_similarities_dict.keys()]

            fig = go.Figure(data=[go.Table(
                header=dict(values=['Métrica'] + list(aggregated_similarities_dict.keys()),  # Nome das colunas
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(
                    values=[metrics] + values,
                    fill_color='lavender',
                    align='left'
                ))
            ])

            fig.update_layout(
                title=title,
            )

            fig.show(config=self.fig_config)

        except Exception as e:
            logging.error(f'Erro ao plotar resumo estatístico: {str(e)}')

    def plot_boxplot_embeddings(self, title, top_n_embeddings, queries_embeddings, top_n=5):
        try:
            similarities = {rank: [] for rank in range(1, top_n+1)}

            for rank in range(1, top_n+1):
                for i, doc_embedding in enumerate(top_n_embeddings[rank]):
                    query_embedding = queries_embeddings[i]
                    
                    similarity = cosine_similarity([doc_embedding], [query_embedding])[0][0]
                    similarities[rank].append(similarity)

            fig = go.Figure()

            for rank in range(1, top_n+1):
                fig.add_trace(go.Box(
                    y=similarities[rank],
                    name=f'Posição {rank}',
                    boxmean=True
                ))

            fig.update_layout(
                title=title,
                yaxis_title='Similaridade de Cosseno',
                xaxis_title='Posição no Ranking'
            )

            fig.show(config=self.fig_config)

        except Exception as e:
            logging.error(f'Erro ao plotar Boxplot para o top-n: {str(e)}')

    def visualize_results_rank(self, tokenizer=None, bert_model=None, data=None, text_embedding_dict={'BERT': 'text_embedding_BERT', 'BERTimbau': 'text_embedding_BERTimbau'}, top_n=5, n_queries=30, text_col='cleaned_text', return_col='titulo'):
        try:
            if data is None:
                data = self.embeddings_test_dataset
            
            aggregated_similarities_dict = {}
            for model_name, text_embedding_col in text_embedding_dict.items():
                if tokenizer is None or bert_model is None:
                    tokenizer, bert_model = self.model_dict[model_name]['tokenizer'], self.model_dict[model_name]['model']
                
                queries = self.generate_random_queries(data=data, n_queries=n_queries, text_col=text_col)
                top_n_embeddings, queries_embeddings = self.get_embeddings_queries(queries=queries, tokenizer=tokenizer, bert_model=bert_model, data=data, text_embedding_col=text_embedding_col, model_name=model_name, top_n=top_n, return_col=return_col)
                
                boxplot_embeddings_title = f'<b>Distribuição das similaridades semânticas dos documentos ranqueados em relação às queries</b><br>Modelo: {model_name}, top {top_n} posições, número de queries: {n_queries}'
                self.plot_boxplot_embeddings(title=boxplot_embeddings_title, top_n_embeddings=top_n_embeddings, queries_embeddings=queries_embeddings, top_n=top_n)
                
                similarities_dict = self.calculate_similarities(top_n_embeddings, queries_embeddings, top_n=top_n)
                
                aggregated_similarities = self.aggregate_similarities(similarities_dict, top_n=top_n)

                aggregated_similarities_dict[model_name] = aggregated_similarities

            def generate_title(prefix, model_name=model_name, n_queries=n_queries):
                return f'<b>{prefix} das similaridades semânticas dos documentos ranqueados em relação às queries</b><br>Modelo: {model_name}, número de queries: {n_queries}'

            titles = ('Dispersão', 'Resumo estatístico', 'Distribuição', 'Boxplot')
            scatter_similarity_title, statistical_summary_title, similarity_distribution_title, boxplot_similarity_title = (
                generate_title(title) for title in titles
            )
            
            self.plot_scatter_similarity(scatter_similarity_title, aggregated_similarities_dict)
            self.plot_similarity_distribution(similarity_distribution_title, aggregated_similarities_dict)
            self.plot_boxplot_similarity(boxplot_similarity_title, aggregated_similarities_dict)
            self.plot_statistical_summary(statistical_summary_title, aggregated_similarities_dict)

        except Exception as e:
            logging.error(f'Erro ao visualizar resultados para o top-n: {str(e)}')

    def execute(self, test_size=0.2):
        try:
            start = time.time()
            logging.info('Iniciando o pipeline de execução...')
            os.makedirs(self.processed_data_path, exist_ok=True)

            data_list = []
            for model_name in self.dict_models.keys():
                data_list.append(os.path.join(self.processed_data_path, f'tokenized_test_dataset_{model_name}.parquet'))

            check = [os.path.isfile(data) for data in data_list]
            
            if not all(check):
                s1 = time.time()
                data = self.import_data()
                logging.info(f'Dados carregados com sucesso.')

                processed_data = self.prepare_data(data)
                dataset = Dataset.from_pandas(processed_data)

                train_test_split = dataset.train_test_split(test_size=test_size)
                train_dataset, test_dataset = train_test_split['train'], train_test_split['test']

                end1 = time.time() - s1
                logging.info(f'Importação e divisão dos dados concluídas em {end1:.2f} segundos.')

                for model_name, opt in self.dict_models.items():
                    s3 = time.time()
                    logging.info(f'Iniciando tokenização para {model_name}...')
                    tokenized_test_dataset_model_path = os.path.join(self.processed_data_path, f'tokenized_test_dataset_{model_name}.parquet')
                    
                    model, model_path, tokenizer_path, results_path = (
                        opt['model'], opt['model_path'], opt['tokenizer_path'], opt['results_path']
                    )

                    os.makedirs(model_path, exist_ok=True)
                    os.makedirs(tokenizer_path, exist_ok=True)
                    os.makedirs(results_path, exist_ok=True)
                    
                    tokenizer, bert_model = (
                        BertTokenizer.from_pretrained(model), BertForMaskedLM.from_pretrained(model)
                    )

                    tokenized_train_dataset = train_dataset.map(self.tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
                    tokenized_test_dataset = test_dataset.map(self.tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})

                    end3 = time.time() - s3
                    logging.info(f'Tokenização para {model_name} concluída em {end3:.2f} segundos.')
                    
                    s4 = time.time()
                    logging.info(f'Iniciando treinamento|importação do modelo {model_name}...')

                    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

                    model_exists, tokenizer_exists = (
                        os.path.isfile(os.path.join(model_path, 'model.safetensors')) and os.path.isfile(os.path.join(model_path, 'config.json')),
                        os.path.isfile(os.path.join(tokenizer_path, 'vocab.txt'))
                    )

                    check_model = [model_exists, tokenizer_exists]

                    if not all(check_model):
                        logging.info(f'Modelo {model_name} não encontrado. Iniciando o treinamento...')
                        trainer = self.train_model(
                            bert_model=bert_model,
                            tokenizer=tokenizer,
                            train_dataset=tokenized_train_dataset,
                            test_dataset=tokenized_test_dataset,
                            data_collator=data_collator,
                            model_path=model_path,
                            tokenizer_path=tokenizer_path,
                            output_dir=results_path
                        )

                        metrics = self.evaluate_model(trainer, tokenized_test_dataset)
                        logging.info(f'Metrics for {model_name}: {metrics}')

                    self.tokenizer, self.bert_model = (
                        BertTokenizer.from_pretrained(tokenizer_path), BertModel.from_pretrained(model_path)
                    )

                    end4 = time.time() - s4
                    logging.info(f'Treinamento|importação do modelo {model_name} concluído em {end4:.2f} segundos.')

                    s5 = time.time()
                    
                    logging.info(f'Iniciando geração de embeddings para {model_name}...')
                    BATCH_SIZE = 8
                    tokenized_test_dataset = tokenized_test_dataset.map(
                        self.generate_embeddings, batched=True, batch_size=BATCH_SIZE, fn_kwargs={'model_name': model_name, 'tokenizer': self.tokenizer, 'bert_model': self.bert_model})
                    
                    end5 = time.time() - s5
                    logging.info(f'Geração de embeddings para {model_name} concluídos em {end5:.2f} segundos.')

                    self.save_dataset(tokenized_test_dataset, tokenized_test_dataset_model_path)
            
            data_dict, self.model_dict, model_name_list = {}, {}, []

            rename_cols = ['input_ids', 'token_type_ids', 'attention_mask']
            for model_name, opt in self.dict_models.items():
                data = self.load_dataset(os.path.join(self.processed_data_path, f'tokenized_test_dataset_{model_name}.parquet'), load_as='pandas')
                
                data_dict[model_name] = data.rename(columns={col: f'{col}_{model_name}' for col in rename_cols})

                model_name_list.append(model_name)

                model_path, tokenizer_path = (opt['model_path'], opt['tokenizer_path'])
                if model_name not in self.model_dict:
                    self.model_dict[model_name] = {}
                    
                self.model_dict[model_name]['model'] = BertModel.from_pretrained(model_path)
                self.model_dict[model_name]['tokenizer'] = BertTokenizer.from_pretrained(tokenizer_path)

            cols_in_common = list(
                set.intersection(
                    *[set(data_dict[model_name].columns) for model_name in model_name_list]
                )
            )

            self.embeddings_test_dataset = reduce(
                lambda left, right: left.merge(right, on=cols_in_common, how='inner'), 
                [data_dict[model_name] for model_name in model_name_list]
            )

            end = time.time() - start
            logging.info(f'Execução completa em {end:.2f} segundos.')

        except Exception as e:
            logging.error(f'{str(e)}')

if __name__ == '__main__':
    file_path = os.path.join(PROJECT_ROOT, 'data', 'internal', 'fapesp_projects', 'all_process.xlsx')
    processed_data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'fapesp_projects')

    model_path = os.path.join(PROJECT_ROOT, 'models')
    tokenizer_path = os.path.join(PROJECT_ROOT, 'tokenizers')
    results_path = os.path.join(model_path, 'results')

    dict_models = {
        'BERTimbau': {
            'model': 'neuralmind/bert-base-portuguese-cased',
            'model_path': os.path.join(model_path, 'BERTimbau'),
            'tokenizer_path': os.path.join(tokenizer_path, 'BERTimbau'),
            'results_path': os.path.join(results_path, 'BERTimbau'),
        },
        'BERT': {
            'model': 'bert-base-multilingual-cased',
            'model_path': os.path.join(model_path, 'BERT'),
            'tokenizer_path': os.path.join(tokenizer_path, 'BERT'),
            'results_path': os.path.join(results_path, 'BERT'),
        }
    }

    # Configuracao do logger
    logging.basicConfig(filename='BertRanking.log', level=logging.DEBUG, format='%(levelname)s: %(asctime)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console_handler)

    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
    
    BertRanking = BertRanking(file_path=file_path, dict_models=dict_models, processed_data_path=processed_data_path)
    BertRanking.execute()