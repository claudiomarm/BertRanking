# Importação de Bibliotecas
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertModel
import torch
from datasets import Dataset
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import re
import logging
import logging.config
import threading

# Definicao da raiz do projeto
PROJECT_ROOT = 'G:/Csouza/nlp/topic_modeling'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

class BertRanking():
    def __init__(self, model_path, tokenizer_path, results_path, embedding_path, file_name, pretrained_model='neuralmind/bert-base-portuguese-cased'):
        self.file_name = file_name
        self.pretrained_model = pretrained_model
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.results_path = results_path
        self.embedding_path = embedding_path

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
        
        self.tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.bert_model = BertForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')
    
    def import_data(self, sheet_name='Sheet1'):
        try:
            logging.info('Importando dados da planilha...')

            return pd.read_excel(self.file_name, sheet_name=sheet_name)
        
        except Exception as e:
            logging.error(f'Erro ao importar dados: {str(e)}')
    
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

            data_train_test = full_data[full_data['assuntos'].notnull() & (full_data['area'] == 'Medicina')]

            data_train_test['titulo'] = data_train_test['titulo'].astype(str)
            data_train_test['palavras_chave'] = data_train_test['palavras_chave'].astype(str)

            data_train_test['cleaned_text'] = data_train_test['titulo'].apply(self.clean_text) + '. ' + data_train_test['resumo'].apply(self.clean_text) + '. Palavras-chave: ' + data_train_test['palavras_chave'].apply(self.clean_text)

            data_train_test['topics'] = data_train_test['assuntos'].apply(lambda x: [s.strip() for s in str(x).split(':')])

            return data_train_test
        
        except Exception as e:
            logging.error(f'Erro ao preparar dados: {str(e)}')

    def tokenize_function(self, dataset):
        try:
            logging.info('Tokenizando textos...')
            
            return self.tokenizer(dataset['cleaned_text'], padding="max_length", truncation=True, max_length=512)
        
        except Exception as e:
            logging.error(f'Erro na tokenização: {str(e)}')

    def train_model(self, train_dataset, test_dataset, data_collator, model_path, tokenizer_path, output_dir, overwrite_output_dir=True, save_steps=10_000, save_total_limit=2, prediction_loss_only=True, num_train_epochs=3, per_device_train_batch_size=8):
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
                model=self.bert_model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )

            trainer.train()

            if model_path:
                trainer.save_model(model_path)
            
            if tokenizer_path:
                self.tokenizer.save_pretrained(tokenizer_path)
            
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

    def get_embeddings(self, texts, max_length=512, batch_size=8):
        try:
            logging.info('Gerando embeddings...')

            if not hasattr(self.bert_model, "is_on_device"):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.bert_model.to(device)
                self.bert_model.is_on_device = True

            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(self.bert_model.device)

                with torch.no_grad():
                    outputs = self.bert_model(**inputs, output_hidden_states=True)  # Habilitar a saída dos hidden_states

                # Obtenha a última camada oculta dos hidden_states
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]  # A última camada da saída oculta

                batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()  # Calcular a média ao longo das tokens
                all_embeddings.append(batch_embeddings)

            all_embeddings = np.vstack(all_embeddings)

            return all_embeddings

        except Exception as e:
            logging.error(f'Erro ao gerar embeddings: {str(e)}')

    def generate_embeddings(self, dataset, text_col='cleaned_text', topic_col='topics', batch_size=8):
        try:
            logging.info('Gerando embeddings do dataset...')

            texts = dataset[text_col]
            dataset['text_embedding'] = self.get_embeddings(texts, batch_size=batch_size)
            
            all_topics = dataset[topic_col]
            all_topics_embeddings = []
            for subjects in all_topics:
                subjects_embeddings = self.get_embeddings(subjects, batch_size=batch_size)
                all_topics_embeddings.append(subjects_embeddings)
            
            dataset['topics_embeddings'] = all_topics_embeddings

            return dataset
        
        except Exception as e:
            logging.error(f'Erro ao gerar embeddings do dataset: {str(e)}')
    
    def save_dataset(self, dataset, path):
        try:
            logging.info('Salvando dataset...')

            df = dataset.to_pandas()

            df.to_parquet(path, index=False)
        
        except Exception as e:
            logging.error(f'Erro ao salvar dataset: {str(e)}')

    def load_dataset(self, path):
        try:
            logging.info('Carregando dataset salvo...')

            df = pd.read_parquet(path)

            return Dataset.from_pandas(df)
        
        except Exception as e:
            logging.error(f'Erro ao carregar dataset: {str(e)}')
    
    def rank_topics_by_relevance(self, text_embedding, topics_embeddings, topics):
        try:
            logging.info('Ranqueando tópicos por relevância...')

            text_embedding = np.array(text_embedding)
            topics_embeddings = [np.array(topic_emb) for topic_emb in topics_embeddings]
            
            # Calcular similaridades de cosseno entre o embedding do texto e cada embedding dos tópicos
            similarities = [cosine_similarity(text_embedding.reshape(1, -1), topic_emb.reshape(1, -1))[0, 0] for topic_emb in topics_embeddings]
            
            # Classificar os tópicos de acordo com a similaridade, do maior para o menor
            ranked_topics = sorted(zip(topics, similarities), key=lambda x: x[1], reverse=True)
            
            # Retornar apenas os tópicos ranqueados
            return [topic for topic, _ in ranked_topics]
        
        except Exception as e:
            logging.error(f'Erro ao ranquear tópicos: {str(e)}')

    def rank_topics(self, dataset, text_embedding_col='text_embedding', topics_embeddings_col='topics_embeddings', topics_col='topics'):
        try:
            logging.info('Aplicando ranqueamento de tópicos ao dataset...')
            
            dataset['ranked_topics'] = self.rank_topics_by_relevance(dataset[text_embedding_col], dataset[topics_embeddings_col], dataset[topics_col])
            
            return dataset
        
        except Exception as e:
            logging.error(f'Erro ao aplicar ranqueamento de tópicos: {str(e)}')
    
    def evaluate_ranking(self, test_dataset, k=3):
        try:
            logging.info('Avaliando precisão, recall e NDCG no top-k...')

            def precision_at_k(true_labels, predicted_labels, k):
                correct_predictions = 0
                
                for true, predicted in zip(true_labels, predicted_labels):
                    predicted_top_k = predicted[:k]
                    if any(subject in predicted_top_k for subject in true):
                        correct_predictions += 1
                
                return correct_predictions / len(true_labels)

            def recall_at_k(true_labels, predicted_labels, k):
                correct_predictions = 0
                
                for true, predicted in zip(true_labels, predicted_labels):
                    predicted_top_k = predicted[:k]
                    correct_in_top_k = len(set(true) & set(predicted_top_k))
                    total_relevant = len(true)
                    
                    if total_relevant > 0:
                        correct_predictions += correct_in_top_k / total_relevant
                
                return correct_predictions / len(true_labels)

            def dcg_at_k(relevances, k):
                relevances = np.array(relevances)[:k]
                return np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))

            def ndcg_at_k(true_labels, predicted_labels, k):
                total_ndcg = 0.0
                
                for true, predicted in zip(true_labels, predicted_labels):
                    # Atribuir relevância: 1 para tópicos verdadeiros, 0 para os outros
                    relevances = [1 if topic in true else 0 for topic in predicted[:k]]
                    dcg = dcg_at_k(relevances, k)
                    ideal_relevances = sorted(relevances, reverse=True)
                    idcg = dcg_at_k(ideal_relevances, k)
                    
                    if idcg > 0:
                        total_ndcg += dcg / idcg
                
                return total_ndcg / len(true_labels)
                    
            true_labels, predicted_labels = (test_dataset['topics'], test_dataset['ranked_topics'])

            precision, recall, ndcg = (
                precision_at_k(true_labels, predicted_labels, k), recall_at_k(true_labels, predicted_labels, k), ndcg_at_k(true_labels, predicted_labels, k)
            )

            return precision, recall, ndcg
        
        except Exception as e:
            logging.error(f'Erro na avaliação de ranqueamento: {str(e)}')
    
    def calculate_mean_cosine_similarity(self, text_embeddings, topics_embeddings):
        try:
            logging.info('Calculando similaridade média de cosseno...')

            all_similarities = []
            for text_emb, topic_embs in zip(text_embeddings, topics_embeddings):
                similarities = [cosine_similarity(text_emb.reshape(1, -1), topic_emb.reshape(1, -1))[0, 0]
                                for topic_emb in topic_embs]
                all_similarities.append(np.mean(similarities))

            return np.array(all_similarities)
        
        except Exception as e:
            logging.error(f'Erro ao calcular similaridade de cosseno: {str(e)}')
    
    def execute(self, test_size=0.2):
        try:
            start = time.time()
            logging.info('Iniciando o pipeline de execução...')

            s1 = time.time()
            data = self.import_data()
            logging.info(f'Dados carregados com sucesso.')

            end1 = time.time() - s1
            logging.info(f'Importação dos dados concluída em {end1:.2f} segundos.')
            

            s2 = time.time()
            processed_data = self.prepare_data(data)

            dataset = Dataset.from_pandas(processed_data)

            tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

            train_test_split = tokenized_datasets.train_test_split(test_size=test_size)
            train_dataset, test_dataset = (
                train_test_split['train'],
                train_test_split['test']
            )

            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

            end2 = time.time() - s2
            logging.info(f'Preparação dos dados concluída em {end2:.2f} segundos.')
            
            s3 = time.time()
            model_exists, tokenizer_exists = (
                os.path.isfile(os.path.join(self.model_path, 'model.safetensors')) and os.path.isfile(os.path.join(self.model_path, 'config.json')),
                os.path.isfile(os.path.join(self.tokenizer_path, 'vocab.txt'))
            )

            # Treinar o modelo se ele não existir
            if not model_exists or not tokenizer_exists:
                logging.info('Modelo treinado não encontrado. Iniciando o treinamento...')
                trainer = self.train_model(
                    model=bert_model,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    data_collator=data_collator,
                    model_path=self.model_path,
                    tokenizer_path=self.tokenizer_path,
                    output_dir=self.results_path
                )

                metrics = self.evaluate_model(trainer, test_dataset)
                logging.info(f'Metrics: {metrics}')

            tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
            bert_model = BertModel.from_pretrained(self.model_path)

            end3 = time.time() - s3
            logging.info(f'Tokenização concluída em {end3:.2f} segundos.')

            s4 = time.time()
            if os.path.exists(self.embedding_path):
                logging.info('Carregando test_dataset do arquivo salvo...')
                test_dataset = self.load_dataset(self.embedding_path)
            else:
                logging.info('Gerando embeddings e salvando test_dataset...')
                
                BATCH_SIZE = 8
                test_dataset = test_dataset.map(self.generate_embeddings, batched=True, batch_size=BATCH_SIZE)
                test_dataset = test_dataset.map(self.rank_topics, batched=False)

                self.save_dataset(test_dataset, self.embedding_path)
                logging.info('test_dataset salvo com sucesso.')

            self.test_dataset = test_dataset

            end4 = time.time() - s4
            logging.info(f'Treinamento do modelo concluído em {end4:.2f} segundos.')

            s5 = time.time()
            self.bert_similarities = self.calculate_mean_cosine_similarity(test_dataset['text_embedding'], test_dataset['topics_embeddings'])

            end5 = time.time() - s5
            logging.info(f'Cálculo da similaridade semântica concluída em {end5:.2f} segundos.')

            end = time.time() - start
            logging.info(f'Execução concluída em {end:.2f} segundos.')

        except Exception as e:
            logging.error(f'{str(e)}')
            
if __name__ == '__main__':
    file_name = os.path.join(PROJECT_ROOT, 'data', 'internal', 'fapesp_projects', 'all_process.xlsx')
    embedding_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'fapesp_projects', 'test_dataset_with_embeddings.parquet')

    model_path = os.path.join(PROJECT_ROOT, 'models')
    tokenizer_path = os.path.join(PROJECT_ROOT, 'tokenizers')
    results_path = os.path.join(model_path, 'results')

    # Configuracao do logger
    logging.basicConfig(filename='BertRanking.log', level=logging.DEBUG, format='%(levelname)s: %(asctime)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console_handler)

    os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.WE8ISO8859P1'
    
    BertRanking = BertRanking(model_path=model_path, tokenizer_path=tokenizer_path, results_path=results_path, embedding_path=embedding_path, file_name=file_name)
    BertRanking.execute()