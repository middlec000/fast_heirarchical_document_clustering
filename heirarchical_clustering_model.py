from typing import Dict
from data_classes import \
    Vocabulary, \
    Corpus, \
    Level

class HeirarchicalClusteringModel():
    vocabulary: Vocabulary
    corpus: Corpus
    levels: Dict[int, Level] # {level_id: Level}

    def __init__(self, corpus: Corpus, vocabulary: Vocabulary):
        self.corpus = corpus
        self.vocabulary = vocabulary
    '''
    def norm(doc: Doc, use_cache=True):
        if use_cache and doc[0] in norm_cache:
            return norm_cache[doc[0]]
        freqs = doc[1]
        norm = sum(freqs[word] ** 2 for word in freqs) ** (1/2)
        if use_cache:
            norm_cache[doc[0]] = norm
        return norm

    def item_distance_dot_product(a: Doc, b: Doc, use_cache=True) -> float:
        similarity = 0
        for word in a[1]:
            if word in b[1]:
                similarity += a[1][word] * b[1][word]
        norm_a = norm(a, use_cache=use_cache)
        norm_b = norm(b, use_cache=use_cache)
        if norm_a == 0:
            print(f"doc has norm zero: {a[0]}")
            return 1
        if norm_b == 0:
            print(f"doc has norm zero: {b[0]}")
            return 1
        similarity_normalized = similarity / (norm_a * norm_b)
        return 1 - similarity_normalized
    '''

