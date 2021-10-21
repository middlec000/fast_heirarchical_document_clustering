from dataclasses import dataclass
from typing import List, Dict

from new_code.preprocessing import preprocess

num_items_to_print = 5

@dataclass
class Doc:
    doc_id: int
    contents: Dict[int, float] # {word_id: TF-IDF}

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

@dataclass
class Cluster:
    cluster_id: int
    docs: Dict[int, Doc] # {doc_id, Doc}
    theme: List[str]

    def __str__(self):
        cluster_string = f"Cluster ID: {self.cluster_id}\nFirst {num_items_to_print-1} Documents: {list(self.docs.keys())[:num_items_to_print]}...\nTheme: {self.theme[num_items_to_print]}..."
        return cluster_string
    
@dataclass
class Level:
    level_id: int
    clusters: Dict[int, Cluster] # {cluster_id: Cluster}
    num_clusters: int

    def __str__(self):
        level_string = f"Level: {self.level_id}\nNumber of Clusters: {self.num_clusters}"
        return level_string

