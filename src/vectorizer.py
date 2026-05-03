"""Custom TF-IDF Vectorizer (no sklearn!)"""
import numpy as np
from collections import defaultdict

class TFIDFVectorizer:
    """Custom TF-IDF: Transform text to numbers"""
    
    def __init__(self, max_features=5000, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}
        self.idf_weights = None
    
    def fit(self, documents):
        """Learn vocabulary and IDF weights"""
        documents = list(documents)
        term_frequencies = defaultdict(int)
        num_docs = len(documents)

        for doc in documents:
            if isinstance(doc, str):
                terms = set(doc.lower().split())
            else:
                terms = set(str(doc).lower().split())
            for term in terms:
                term_frequencies[term] += 1
        
        # Keep most frequent terms
        sorted_terms = sorted(
            [(t, f) for t, f in term_frequencies.items() if f >= self.min_df],
            key=lambda x: x[1], reverse=True
        )[:self.max_features]
        
        self.vocab = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        
        # Compute IDF weights
        self.idf_weights = np.zeros(len(self.vocab))
        for term, idx in self.vocab.items():
            df = term_frequencies[term]
            self.idf_weights[idx] = np.log((num_docs + 1) / (1 + df))
        
        return self
    
    def transform(self, documents):
        """Convert documents to TF-IDF vectors"""
        documents = list(documents)
        n_docs = len(documents)
        n_features = len(self.vocab)

        if n_features == 0:
            return np.zeros((n_docs, 1))

        matrix = np.zeros((n_docs, n_features))

        for doc_idx, doc in enumerate(documents):
            doc_str = doc if isinstance(doc, str) else str(doc)
            terms = doc_str.lower().split()
            term_counts = defaultdict(int)
            for term in terms:
                if term in self.vocab:
                    term_counts[term] += 1

            if len(terms) > 0:
                for term, count in term_counts.items():
                    idx = self.vocab[term]
                    tf = count / len(terms)
                    matrix[doc_idx, idx] = tf * self.idf_weights[idx]

                norm = np.linalg.norm(matrix[doc_idx])
                if norm > 0:
                    matrix[doc_idx] /= norm

        return matrix
    
    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)
