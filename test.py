import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import umap
from bertopic import BERTopic
from gensim.models import Word2Vec
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import Levenshtein
from datetime import datetime
import hdbscan
import warnings
warnings.filterwarnings('ignore')

class EnterpriseLogClustering:
    def __init__(self, custom_patterns=None):
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.domain_bert = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
        self.topic_model = BERTopic(language="english")
        
        self.vectorizers = {}
        self.reducers = {}
        self.scalers = {}
        
        self.clustering_models = {}
        self.best_model = None
        
        self.technical_patterns = {
            'error_codes': r'(?i)(error|err|fail|exception)[:\s]+[A-Z0-9\-_]+',
            'hex_codes': r'0x[0-9A-Fa-f]+',
            'ip_addresses': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'timestamps': r'\d{2}:\d{2}:\d{2}(?:\.\d{3})?',
            'versions': r'\b\d+\.\d+\.\d+(?:\.\d+)?\b',
            'memory_addresses': r'\b0x[0-9a-fA-F]+\b',
            'urls': r'https?://\S+',
            'file_paths': r'/?(?:[\w\-. ]+/)*[\w\-. ]+\.\w+',
            'port_numbers': r'\b\d{1,5}/(?:tcp|udp)\b',
            'mac_addresses': r'(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})',
            'process_ids': r'\bPID[\s:]?[0-9]+\b'
        }
        
        if custom_patterns:
            self.technical_patterns.update(custom_patterns)
            
        self.load_domain_knowledge()

    def load_domain_knowledge(self):
        """Load and maintain domain-specific knowledge."""
        self.domain_terms = {
            'hardware': [
                'cpu', 'memory', 'disk', 'network', 'hardware', 'device', 'port',
                'interface', 'controller', 'buffer', 'cache', 'stack', 'queue'
            ],
            'software': [
                'driver', 'firmware', 'software', 'application', 'service', 'process',
                'thread', 'daemon', 'kernel', 'module', 'package'
            ],
            'errors': [
                'error', 'failure', 'crash', 'exception', 'timeout', 'overflow',
                'underflow', 'corruption', 'leak', 'deadlock', 'bottleneck'
            ],
            'actions': [
                'restart', 'reboot', 'shutdown', 'initialize', 'configure', 'update',
                'upgrade', 'downgrade', 'install', 'uninstall', 'modify'
            ]
        }
        
        self.term_weights = {}
        for category, terms in self.domain_terms.items():
            weight = 2.0 if category in ['errors', 'hardware'] else 1.5
            for term in terms:
                self.term_weights[term] = weight

    def extract_temporal_patterns(self, text):
        """Extract and analyze temporal patterns in logs."""
        temporal_info = {
            'timestamps': [],
            'durations': [],
            'sequences': []
        }
        
        timestamps = re.findall(r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{3})?\b', text)
        if timestamps:
            temporal_info['timestamps'] = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]
            
        duration_patterns = re.findall(r'\b(\d+)\s*(ms|seconds|minutes|hours|days)\b', text, re.IGNORECASE)
        temporal_info['durations'] = [(int(value), unit) for value, unit in duration_patterns]
        
        return temporal_info

    def create_hybrid_embeddings(self, texts):
        """Create sophisticated hybrid embeddings."""
        embeddings = {
            'bert': self.bert_model.encode(texts),
            'domain_bert': self.domain_bert.encode(texts),
            'tfidf': None,
            'nmf': None
        }
        
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            token_pattern=r'(?u)\b[a-zA-Z0-9\-_./:%]+\b',
            preprocessor=self._preprocess_for_tfidf
        )
        tfidf_matrix = self.vectorizers['tfidf'].fit_transform(texts)
        
        self.reducers['nmf'] = NMF(n_components=50, random_state=42)
        embeddings['nmf'] = self.reducers['nmf'].fit_transform(tfidf_matrix)
        
        self.reducers['tfidf'] = TruncatedSVD(n_components=100)
        embeddings['tfidf'] = self.reducers['tfidf'].fit_transform(tfidf_matrix)
        
        combined = np.hstack([
            embeddings['bert'],
            embeddings['domain_bert'],
            embeddings['tfidf'],
            embeddings['nmf']
        ])
        
        self.scalers['combined'] = StandardScaler()
        scaled_embeddings = self.scalers['combined'].fit_transform(combined)
        
        self.reducers['umap'] = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=100,
            random_state=42
        )
        final_embeddings = self.reducers['umap'].fit_transform(scaled_embeddings)
        
        return final_embeddings

    def dynamic_weighting(self, json_str):
        """Dynamically weight JSON components based on content."""
        try:
            data = json.loads(json_str)
            weighted_components = []
            
            def calculate_importance(text):
                score = 1.0
                for term, weight in self.term_weights.items():
                    if term in text.lower():
                        score *= weight
                for pattern in self.technical_patterns.values():
                    if re.search(pattern, text):
                        score *= 1.5
                return score

            if 'info' in data:
                importance = calculate_importance(data['info'])
                weighted_components.extend([data['info']] * int(4 * importance))
                
            if 'tags' in data:
                for item in data['tags']:
                    if 'fProof' in item:
                        importance = calculate_importance(item['fProof'])
                        weighted_components.extend([item['fProof']] * int(3 * importance))
                    if 'inf' in item:
                        importance = calculate_importance(item['inf'])
                        weighted_components.extend([item['inf']] * int(2 * importance))
                    if 'similarL' in item:
                        importance = calculate_importance(item['similarL'])
                        weighted_components.extend([item['similarL']] * int(3 * importance))
            
            return ' '.join(weighted_components)
        except json.JSONDecodeError:
            return json_str

    def find_optimal_clusters(self, embeddings):
        """Find optimal clustering parameters using multiple methods."""
        best_score = float('-inf')
        best_labels = None
        best_model = None
        
        clustering_configs = {
            'hdbscan': [
                {'min_cluster_size': size, 'min_samples': samples}
                for size in [5, 10, 15]
                for samples in [2, 3, 4]
            ],
            'optics': [
                {'min_samples': samples, 'xi': xi}
                for samples in [2, 3, 4]
                for xi in [0.01, 0.05, 0.1]
            ],
            'kmeans': [
                {'n_clusters': k}
                for k in range(max(2, len(embeddings) // 50), min(len(embeddings) // 10, 100))
            ]
        }
        
        for algo_name, configs in clustering_configs.items():
            for config in configs:
                if algo_name == 'hdbscan':
                    model = hdbscan.HDBSCAN(**config)
                elif algo_name == 'optics':
                    model = OPTICS(**config)
                else:
                    model = KMeans(**config)
                
                labels = model.fit_predict(embeddings)
                
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(embeddings, labels)
                    calinski = calinski_harabasz_score(embeddings, labels)
                    davies = davies_bouldin_score(embeddings, labels)
                    
                    combined_score = (silhouette + calinski/1000 - davies) / 3
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_labels = labels
                        best_model = model
        
        return best_labels, best_model

    def analyze_cluster_patterns(self, texts, labels):
        """Perform detailed pattern analysis for each cluster."""
        cluster_patterns = defaultdict(lambda: {
            'common_patterns': Counter(),
            'temporal_patterns': [],
            'error_sequences': [],
            'term_frequencies': Counter(),
            'similar_issues': []
        })
        
        for text, label in zip(texts, labels):
            if label == -1:
                continue
                
            for pattern_name, pattern in self.technical_patterns.items():
                matches = re.findall(pattern, text)
                cluster_patterns[label]['common_patterns'].update(matches)
            
            temporal_info = self.extract_temporal_patterns(text)
            cluster_patterns[label]['temporal_patterns'].append(temporal_info)
            
            terms = word_tokenize(text.lower())
            cluster_patterns[label]['term_frequencies'].update(terms)
            
            for other_text in texts:
                if Levenshtein.ratio(text, other_text) > 0.8:
                    cluster_patterns[label]['similar_issues'].append(other_text)
        
        return dict(cluster_patterns)

    def fit_transform(self, texts):
        """Main clustering method with comprehensive analysis."""
        weighted_texts = [self.dynamic_weighting(text) for text in texts]
        
        embeddings = self.create_hybrid_embeddings(weighted_texts)
        
        labels, best_model = self.find_optimal_clusters(embeddings)
        self.best_model = best_model
        
        cluster_patterns = self.analyze_cluster_patterns(texts, labels)
        
        return labels, cluster_patterns

def cluster_enterprise_logs(df, summary_column='TLDRCOLUMN', custom_patterns=None):
    """Main function for enterprise-grade log clustering."""
    clusterer = EnterpriseLogClustering(custom_patterns)
    
    labels, patterns = clusterer.fit_transform(df[summary_column].tolist())
    
    df['cluster_id'] = labels
    
    cluster_report = {
        'cluster_patterns': patterns,
        'model_info': {
            'type': type(clusterer.best_model).__name__,
            'params': clusterer.best_model.get_params(),
            'num_clusters': len(set(labels)) - (1 if -1 in labels else 0)
        },
        'statistics': {
            'total_logs': len(df),
            'clustered_logs': sum(1 for l in labels if l != -1),
            'noise_points': sum(1 for l in labels if l == -1),
            'avg_cluster_size': np.mean([sum(1 for l in labels if l == i) 
                                       for i in set(labels) if i != -1])
        }
    }
    
    return df, cluster_report