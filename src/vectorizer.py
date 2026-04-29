# Pseudocode structure:
class TFIDFVectorizer:
    def fit(self, documents):
        """Build vocabulary and compute IDF weights"""
        # 1. Tokenize all documents
        # 2. Build vocab (term → index mapping)
        # 3. Count document frequencies df(t)
        # 4. Compute IDF: idf[t] = log(N / (1 + df[t]))
        
    def transform(self, documents):
        """Convert documents to sparse TF-IDF matrix"""
        # For each document:
        #   1. Tokenize
        #   2. Compute TF: tf[t] = count(t) / len(doc)
        #   3. Multiply: tfidf[t] = tf[t] * idf[t]
        #   4. Normalize: divide by L2 norm
        # Output: (n_samples, n_features) sparse matrix