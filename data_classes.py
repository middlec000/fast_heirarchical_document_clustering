from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, FrozenSet

num_items_to_print = 10

@dataclass
class Doc:
    doc_id: int
    # contents: Dict[int, float] # {word_id: TF-IDF}
    contents: Dict[int, int] # {word_id: frequency}

    def __str__(self):
        keys = list(self.contents.keys())[:num_items_to_print]
        values = list(self.contents.values())[:num_items_to_print]
        doc_string = f"Doc ID: {self.doc_id}\nContents: {dict(zip(keys, values))}..."
        return doc_string

@dataclass
class Corpus:
    docs: Dict[int, Doc] # {doc_id: Doc}

    def __str__(self):
        corpus_string = f"First {num_items_to_print} Documents: {list(self.docs)[:num_items_to_print]}..."
        return corpus_string

@dataclass
class Vocabulary:
    id_word: Dict[int, str] # {word_id: word}
    word_id: Dict[str, int] # {word: word_id}
    id_count: Dict[int, int] # {word_id: count}

    def __str__(self):
        # id_word
        keys = list(self.id_word.keys())[:num_items_to_print]
        values = list(self.id_word.values())[:num_items_to_print]
        vocab_string = f"ID to Word: {dict(zip(keys, values))}..."
        # word_id
        keys = list(self.word_id.keys())[:num_items_to_print]
        values = list(self.word_id.values())[:num_items_to_print]
        vocab_string += f"\nWord to ID: {dict(zip(keys, values))}..."
        # id_count
        keys = list(self.id_count.keys())[:num_items_to_print]
        values = list(self.id_count.values())[:num_items_to_print]
        vocab_string += f"\nID to Count: {dict(zip(keys, values))}..."
        return vocab_string
    # TODO: add methods for adding, removing words

class Cluster:
    cluster_id: int
    docs: List[int] # doc_ids
    contents: Dict[int, int] # {word_id: frequency}
    norm: float
    theme: List[str]

    def norm(self) -> float:
        # L2 Euclidean Norm
        # https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
        return sum([self.contents[word_id] ** 2 for word_id in self.contents]) ** (1/2)
    
    def __init__(self, cluster_id: int, docs: List[int], contents: Dict[int, int]):
        self.cluster_id = cluster_id
        self.docs = docs
        self.contents = contents
        self.norm = self.norm()
        self.theme = None # TODO: add
        return

    def __str__(self):
        cluster_string = f"Cluster ID: {self.cluster_id}\nFirst {num_items_to_print-1} Documents: {list(self.docs.keys())[:num_items_to_print]}...\nTheme: {self.theme[num_items_to_print]}..."
        return cluster_string
    
@dataclass
class Level:
    level_id: int
    clusters: Dict[int, Cluster] # {cluster_id: Cluster}
    num_clusters: int

    def __str__(self):
        level_string = f"Level: {self.level_id}\nNumber of Clusters: {self.num_clusters}\nClusters: {list(self.clusters.keys())[:num_items_to_print]}...\n"
        return level_string

@dataclass
class Similarity_Matrix:
    similarities: Dict[FrozenSet[int], float] # {(cluster_a_id, cluster_b_id): similarity}

