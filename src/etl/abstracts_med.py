# Importação de bibliotecas
import polars as pl
import logging
import logging.config
import requests
from spacy.lang.en.stop_words import STOP_WORDS
from xml.etree import ElementTree as ET

# Classe ETL
class AbstractOpenAlexPubMedETL():
    def __init__(self, open_alex_concept_params={'search': 'health'}, open_alex_concept='Health psychology', open_alex_language ='en', open_alex_from_publication_date='2022-01-01', open_alex_to_publication_date='2024-12-31', open_alex_per_page=200):
        self.concept_params = open_alex_concept_params
        self.concept = open_alex_concept
        self.language = open_alex_language
        self.from_publication_date = open_alex_from_publication_date
        self.to_publication_date = open_alex_to_publication_date
        self.per_page = open_alex_per_page
        
    def search_openalex(self, entity='works', **kwargs):
        base_url = f'https://api.openalex.org/{entity}'
        try:
            response = requests.get(base_url, params=kwargs)

            return response.json()
        
        except requests.exceptions.RequestException as e:
            logging.error(f'Request failed: {str(e)}')
            raise e

    def search_pubmed(self, doi):
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
        try:
            params = {
                'db': 'pubmed',
                'term': doi,
                'retmode': 'json'
            }
            response = requests.get(url, params=params)
            data = response.json()
            idlist = data['esearchresult']['idlist']
            
            return idlist[0]
        
        except requests.exceptions.RequestException as e:
            logging.error(f'Request failed: {str(e)}')
            return None

    def fetch_abstract(self, pmid):
        url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
        try:
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml'
            }

            response = requests.get(url, params=params)
            root = ET.fromstring(response.content)

            abstract_text = ''
            for abstract in root.findall(".//AbstractText"):
                abstract_text += abstract.text + ' '
            return abstract_text.strip()
        
        except requests.exceptions.RequestException as e:
            logging.error(f'Request failed: {str(e)}')
            return None
    
    def fetch_concepts(self, concept_params):
        try:
            concept_result = self.search_openalex(entity='concepts', **concept_params)

            concepts = {}
            for concept in concept_result['results']:
                concepts[concept['display_name']] = concept['id'].split('/')[-1]

            return concepts
        
        except requests.exceptions.RequestException as e:
            logging.error(f'Error {str(e)}')
            raise e
    
    def fetch_doi(self, open_alex_works):
        try:
            doi_dict = {}
            for work in open_alex_works['results']:
                open_alex_id = work.get('id').split('/')[-1]
                doi = work.get('doi')
                doi = doi.split('org/')[-1]
                
                if open_alex_id and doi:
                    doi_dict[open_alex_id] = doi

            return doi_dict
        
        except requests.exceptions.RequestException as e:
            logging.error(f'Error: {str(e)}')
            raise e
        
    def abstracts(self):
        try:
            concepts = self.fetch_concepts(self.concept_params)
            concept_id = concepts[self.concept]

            work_params = {
                'filter': f'language:{self.language},from_publication_date:{self.from_publication_date},to_publication_date:{self.to_publication_date},concepts.id:{concept_id}',
                'per_page': self.per_page,
            }

            works = self.search_openalex(entity='works', **work_params)

            doi_dict = self.fetch_doi(works)

            articles = {}
            for article_id, doi in doi_dict.items():
                pmid = self.search_pubmed(doi)
                
                if pmid:
                    abstract = self.fetch_abstract(pmid)
                    articles[article_id] = abstract
            
            return articles
        
        except requests.exceptions.RequestException as e:
            logging.error(f'Error: {str(e)}')
            raise e
    
    def load_abstracts(self, df=None, load_path=None, file_name=None, **kwargs):
        try:
            abstracts = self.abstracts()

            if df is None:
                df = pl.DataFrame(abstracts, **kwargs)

            df.write_parquet(f'{load_path}/{file_name}.parquet')

        except requests.exceptions.RequestException as e:
            logging.error(f'Error: {str(e)}')
            raise e