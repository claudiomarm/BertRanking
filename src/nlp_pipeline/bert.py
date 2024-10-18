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

    def generate_embeddings(self, dataset, tokenizer, bert_model, text_col='cleaned_text', topic_col='topics', batch_size=8, model_name='bertimbal'):
        try:
            logging.info('Gerando embeddings do dataset...')

            texts = dataset[text_col]
            dataset[f'text_embedding_{model_name}'] = self.get_embeddings(texts=texts, tokenizer=tokenizer, bert_model=bert_model, batch_size=batch_size)
            
            if topic_col:
                all_topics = dataset[topic_col]
                all_topics_embeddings = []
                for subjects in all_topics:
                    subjects_embeddings = self.get_embeddings(texts=subjects, tokenizer=tokenizer, bert_model=bert_model, batch_size=batch_size)
                    all_topics_embeddings.append(subjects_embeddings)
                
                dataset[f'topics_embeddings_{model_name}'] = all_topics_embeddings

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
    
    def rank_texts(self, query_text, tokenizer=None, bert_model=None, data=None, text_embedding_col='text_embedding_BERT', model_name='BERT', top_n=5, return_col='titulo'):
        """
        Ranqueia os artigos com base na similaridade semântica do artigo de consulta com os artigos da base.
        Utiliza o DataFrame contendo os embeddings dos artigos. Controla o número máximo de artigos no ranking com `top_n`
        e permite selecionar a coluna a ser retornada (ex: título, número do processo).
        
        Args:
        query_text (str): Texto do artigo de consulta.
        tokenizer: Tokenizador do modelo.
        bert_model: Modelo BERT ou BERTimbau.
        data (pd.DataFrame): DataFrame contendo os textos e embeddings dos artigos.
        top_n (int): Número máximo de artigos a serem exibidos no ranking.
        return_col (str): Nome da coluna que será retornada no ranking (ex: 'titulo', 'n_processo').
        
        Returns:
        ranked_values (list): Lista de valores da coluna selecionada ranqueados.
        ranked_similarities (list): Lista de similaridades dos artigos ranqueados.
        """
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
        
        for i, (value, sim) in enumerate(zip(ranked_values, ranked_similarities)):
            print(f"Rank {i+1}: Similaridade {sim:.4f}")
            print(f"Valor da coluna '{return_col}': {value}\n")
        
        return ranked_values, ranked_similarities

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

                    self.tokenizer, self.bert_model = (
                        BertTokenizer.from_pretrained(tokenizer_path), BertModel.from_pretrained(model_path)
                    )

                    end4 = time.time() - s4
                    logging.info(f'Treinamento|importação do modelo {model_name} concluído em {end4:.2f} segundos.')

                    s4 = time.time()
                    
                    BATCH_SIZE = 8
                    tokenized_test_dataset = tokenized_test_dataset.map(
                        self.generate_embeddings, batched=True, batch_size=BATCH_SIZE, fn_kwargs={'model_name': model_name, 'tokenizer': self.tokenizer, 'bert_model': self.bert_model})
                    
                    end4 = time.time() - s4
                    logging.info(f'Geração de embeddings e ranqueamento de tópicos para {model_name} concluídos em {end4:.2f} segundos.')

                    self.save_dataset(tokenized_test_dataset, tokenized_test_dataset_model_path)
            
            data_dict, self.model_dict, model_name_list = {}, {}, []

            rename_cols = ['input_ids', 'token_type_ids', 'attention_mask', 'ranked_topics']
            for model_name, opt in self.dict_models.items():
                data = self.load_dataset(os.path.join(self.processed_data_path, f'tokenized_test_dataset_{model_name}.parquet'), load_as='pandas')
                
                if 'topics' in data.columns:
                    data['topics'] = data['topics'].apply(lambda x: tuple(x))
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