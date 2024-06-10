import requests
import xml.etree.ElementTree as ET
import pandas as pd
import json

class USPVocabularyETL:
    def __init__(self, load_path, file_name='vocabulario_usp_hierarchy'):
        self.url_top_terms = 'https://vocabulario.abcd.usp.br/pt-br/services.php?task=fetchTopTerms'
        self.df_top_terms = pd.DataFrame()
        self.terms_hierarchy = []
        self.load_path = load_path
        self.file_name = file_name

    def fetch_top_terms(self):
        response = requests.get(self.url_top_terms, verify=False)
        if response.status_code == 200:
            # Processar o XML diretamente da resposta
            root = ET.fromstring(response.content)
            self.df_top_terms = pd.DataFrame([self.xml_to_dict(child) for child in root.findall('.//term')])
        else:
            print("Falha ao baixar os termos principais do vocabulário:", response.status_code)
    
    @staticmethod
    def xml_to_dict(element):
        data_dict = {}
        for child in element:
            data_dict[child.tag] = child.text
        return data_dict

    def fetch_down(self, term_id):
        url = f'https://vocabulario.abcd.usp.br/pt-br/services.php?task=fetchDown&arg={term_id}'
        response = requests.get(url, verify=False)
        if response.status_code == 200:
            tree = ET.fromstring(response.content)
            return [self.xml_to_dict(child) for child in tree.findall('.//term')]
        return []

    def fetch_hierarchy(self, term):
        term_id = term['term_id']
        sub_terms = self.fetch_down(term_id)
        term['narrower_terms'] = sub_terms
        for sub_term in sub_terms:
            self.fetch_hierarchy(sub_term)

    def build_hierarchy(self):
        for _, row in self.df_top_terms.iterrows():
            term = row.to_dict()
            self.fetch_hierarchy(term)
            self.terms_hierarchy.append(term)

    def structure_tree(self, term):
        term_info = {
            'term_id': term['term_id'],
            'string': term['string'],
            'subterms': []
        }
        for sub_term in term.get('narrower_terms', []):
            term_info['subterms'].append(self.structure_tree(sub_term))
        return term_info

    def save_hierarchy(self):
        structured_hierarchy = [self.structure_tree(term) for term in self.terms_hierarchy]
        hierarchy_json = json.dumps(structured_hierarchy, indent=4, ensure_ascii=False)
        with open(f'{self.load_path}/{self.file_name}.json', 'w', encoding='utf-8') as f:
            f.write(hierarchy_json)
        print(f'JSON hierárquico salvo em: {self.load_path}')

    def run_etl(self):
        self.fetch_top_terms()
        self.build_hierarchy()
        self.save_hierarchy()

