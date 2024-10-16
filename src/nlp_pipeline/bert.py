# Importação de Bibliotecas
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertModel
import torch
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import logging
import logging.config
import json

# Definicao da raiz do projeto
PROJECT_ROOT = 'G:/Csouza/nlp/topic_modeling'
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

class BertRanking():
    def __init__(self, file_path, dict_models, processed_data_path):
        self.file_path = file_path
        self.dict_models = dict_models
        self.processed_data_path = processed_data_path
        
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
        
    def import_data(self, sheet_name='Sheet1'):
        try:
            logging.info('Importando dados da planilha...')

            return pd.read_excel(self.file_path, sheet_name=sheet_name)
        
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

    def tokenize_function(self, dataset, tokenizer):
        try:
            logging.info('Tokenizando textos...')
            
            return tokenizer(dataset['cleaned_text'], padding="max_length", truncation=True, max_length=512)
        
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
                    outputs = bert_model(**inputs, output_hidden_states=True)  # Habilitar a saída dos hidden_states

                # Obtenha a última camada oculta dos hidden_states
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]  # A última camada da saída oculta

                batch_embeddings = last_hidden_state.mean(dim=1).cpu().numpy()  # Calcular a média ao longo das tokens
                all_embeddings.append(batch_embeddings)

            all_embeddings = np.vstack(all_embeddings)

            return all_embeddings

        except Exception as e:
            logging.error(f'Erro ao gerar embeddings: {str(e)}')

    def generate_embeddings(self, dataset, tokenizer, bert_model, text_col='cleaned_text', topic_col='topics', batch_size=8, model_name='bertimbal'):
        try:
            logging.info('Gerando embeddings do dataset...')

            texts = dataset[text_col]
            dataset[f'text_embedding_{model_name}'] = self.get_embeddings(texts=texts, tokenizer=tokenizer, bert_model=bert_model, batch_size=batch_size)
            
            all_topics = dataset[topic_col]
            all_topics_embeddings = []
            for subjects in all_topics:
                subjects_embeddings = self.get_embeddings(texts=subjects, tokenizer=tokenizer, bert_model=bert_model, batch_size=batch_size)
                all_topics_embeddings.append(subjects_embeddings)
            
            dataset[f'topics_embeddings_{model_name}'] = all_topics_embeddings

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
    
    def compare_models_by_semantic_similarity(self, m1_similarities, m1_name, m2_similarities, m2_name):
        mean_m1_similarity = np.mean(m1_similarities)
        mean_m2_similarity = np.mean(m2_similarities)

        std_m1_similarity = np.std(m1_similarities)
        std_m2_similarity = np.std(m2_similarities)

        logging.info(f'Média Similaridade ({m1_name}): {mean_m1_similarity:.2f}')
        logging.info(f'Média Similaridade ({m2_name}): {mean_m2_similarity:.2f}')
        logging.info(f'Desvio Padrão Similaridade ({m1_name}): {std_m1_similarity:.2f}')
        logging.info(f'Desvio Padrão Similaridade ({m2_name}): {std_m2_similarity:.2f}')
        
        # Plotando a Distribuição das Similaridades
        plt.figure(figsize=(10, 6))
        sns.histplot(m1_similarities, color='blue', label='BERTimbau', kde=True, bins=20)
        sns.histplot(m2_similarities, color='green', label='RoBERTa', kde=True, bins=20)
        plt.title('Distribuição das Similaridades de Cosseno: BERTimbau vs RoBERTa')
        plt.xlabel('Similaridade de Cosseno')
        plt.ylabel('Frequência')
        plt.legend()
        plt.show()

        # Boxplot para Comparação de Similaridades
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=[m1_similarities, m2_similarities], palette='Set2')
        plt.xticks([0, 1], ['BERTimbau', 'RoBERTa'])
        plt.title('Comparação da Similaridade de Cosseno: BERTimbau vs RoBERTa')
        plt.ylabel('Similaridade de Cosseno')
        plt.show()
    
    def execute(self, test_size=0.2):
        try:
            start = time.time()
            logging.info('Iniciando o pipeline de execução...')
            os.makedirs(self.processed_data_path, exist_ok=True)

            tokenized_test_dataset_path = os.path.join(self.processed_data_path, 'tokenized_test_dataset.parquet')
            models_similarities_path = os.path.join(self.processed_data_path, 'bert_similarities.json')

            models_similarities = {}
            if not os.path.exists(models_similarities_path):
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
                    models_similarities_model_path = os.path.join(self.processed_data_path, f'bert_similarities_{model_name}.json')

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

                    if not model_exists or not tokenizer_exists:
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

                    tokenizer, bert_model = (
                        BertTokenizer.from_pretrained(tokenizer_path), BertModel.from_pretrained(model_path)
                    )

                    end4 = time.time() - s4
                    logging.info(f'Treinamento|importação do modelo {model_name} concluído em {end4:.2f} segundos.')

                    s4 = time.time()
                    
                    BATCH_SIZE = 8
                    tokenized_test_dataset = tokenized_test_dataset.map(
                        self.generate_embeddings, batched=True, batch_size=BATCH_SIZE, fn_kwargs={'model_name': model_name, 'tokenizer': tokenizer, 'bert_model': bert_model})
                    
                    tokenized_test_dataset = tokenized_test_dataset.map(self.rank_topics, batched=False, fn_kwargs={'text_embedding_col': f'text_embedding_{model_name}', 'topics_embeddings_col': f'topics_embeddings_{model_name}'})
                    
                    end4 = time.time() - s4
                    logging.info(f'Geração de embeddings e ranqueamento de tópicos para {model_name} concluídos em {end4:.2f} segundos.')

                    s5 = time.time()
                    bert_similarities = self.calculate_mean_cosine_similarity(
                        tokenized_test_dataset[f'text_embedding_{model_name}'], tokenized_test_dataset[f'topics_embeddings_{model_name}']
                    )

                    models_similarities[model_name] = bert_similarities
                    end5 = time.time() - s5
                    logging.info(f'Cálculo da similaridade semântica para o modelo {model_name} concluída em {end5:.2f} segundos.')

                    self.save_dataset(tokenized_test_dataset, tokenized_test_dataset_model_path)

                    with open(models_similarities_model_path, 'w') as f:
                        json.dump(models_similarities, f)

                self.save_dataset(tokenized_test_dataset, tokenized_test_dataset_path)

                with open(models_similarities_path, 'w') as f:
                    json.dump(models_similarities, f)
            
            models_similarities = json.load(open(models_similarities_path, 'r'))

            model_names = list(models_similarities.keys())

            m1_name, m2_name = model_names[0], model_names[1]
            m1_similarities, m2_similarities = models_similarities[m1_name], models_similarities[m2_name]

            self.compare_models_by_semantic_similarity(m1_similarities=m1_similarities, m1_name=m1_name, m2_similarities=m2_similarities, m2_name=m2_name)

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

    os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.WE8ISO8859P1'
    
    BertRanking = BertRanking(file_path=file_path, dict_models=dict_models, processed_data_path=processed_data_path)
    BertRanking.execute()