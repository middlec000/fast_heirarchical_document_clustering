from dataclasses import dataclass
import numpy
from copy import deepcopy
from typing import List, Dict, FrozenSet

NUMBER_ITEMS_TO_PRINT = 10

@dataclass
class Doc:
    doc_id: int
    # contents: Dict[int, float] # {word_id: TF-IDF}
    contents: Dict[int, int] # {word_id: frequency}

    def __str__(self):
        keys = list(self.contents.keys())[:NUMBER_ITEMS_TO_PRINT]
        values = list(self.contents.values())[:NUMBER_ITEMS_TO_PRINT]
        doc_string = f"Doc ID: {self.doc_id}\nContents: {dict(zip(keys, values))}..."
        return doc_string

@dataclass
class Corpus:
    docs: Dict[int, Doc] # {doc_id: Doc}

    def __str__(self):
        corpus_string = f"First {NUMBER_ITEMS_TO_PRINT} Documents: {list(self.docs)[:NUMBER_ITEMS_TO_PRINT]}..."
        return corpus_string

@dataclass
class Vocabulary:
    id_word: Dict[int, str] # {word_id: word}
    word_id: Dict[str, int] # {word: word_id}
    id_count: Dict[int, int] # {word_id: count}

    def __str__(self):
        # id_word
        keys = list(self.id_word.keys())[:NUMBER_ITEMS_TO_PRINT]
        values = list(self.id_word.values())[:NUMBER_ITEMS_TO_PRINT]
        vocab_string = f"ID to Word: {dict(zip(keys, values))}..."
        # word_id
        keys = list(self.word_id.keys())[:NUMBER_ITEMS_TO_PRINT]
        values = list(self.word_id.values())[:NUMBER_ITEMS_TO_PRINT]
        vocab_string += f"\nWord to ID: {dict(zip(keys, values))}..."
        # id_count
        keys = list(self.id_count.keys())[:NUMBER_ITEMS_TO_PRINT]
        values = list(self.id_count.values())[:NUMBER_ITEMS_TO_PRINT]
        vocab_string += f"\nID to Count: {dict(zip(keys, values))}..."
        return vocab_string
    # TODO: add methods for adding, removing words

class Cluster:
    cluster_id: int
    docs: List[int] # doc_ids
    contents: Dict[int, float] # {word_id: TF-IDF}
    norm: float
    theme: List[str]

    def norm(self) -> float:
        # L2 Euclidean Norm
        # https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
        return sum([self.contents[word_id] ** 2 for word_id in self.contents]) ** (1/2)
    
    def word_frequencies_to_tfidf(self, word_frequencies: Dict[int, int], vocabulary: Vocabulary) -> Dict[int, float]:
        word_id_tfidf = {}
        for word_id in word_frequencies:
            word_id_tfidf[word_id] = float(word_frequencies[word_id]) / float(vocabulary.id_count[word_id])
        return word_id_tfidf
    
    def __init__(self, cluster_id: int, docs: List[int], word_frequencies: Dict[int, int], vocabulary: Vocabulary):
        self.cluster_id = cluster_id
        self.docs = docs
        self.contents = self.word_frequencies_to_tfidf(word_frequencies=word_frequencies, vocabulary=vocabulary)
        self.norm = self.norm()
        self.theme = []
        return

    def __str__(self, verbosity: int=0):
        cluster_string = f"Cluster ID: {self.cluster_id}\n"
        if verbosity == 0:
            cluster_string += f"First {NUMBER_ITEMS_TO_PRINT-1} Documents: {self.docs[:NUMBER_ITEMS_TO_PRINT]}...\nTheme: {self.theme[:NUMBER_ITEMS_TO_PRINT]}...\n"
        elif verbosity == 1:
            cluster_string += f"Documents: {self.docs}\nTheme: {self.theme}\n"
        else:
            return "Verbosity level not recognized, please choose a supported verbosity level.\n"
        return cluster_string
    
    def merge(self, another: "Cluster", cluster_id: int, vocabulary: Vocabulary) -> "Cluster":
        new_cluster_docs = self.docs + another.docs
        # New cluster contents consists of sum of tfidf scores of parent clusters
        new_cluster_contents = {}
        common_words = [word_id for word_id in self.contents if word_id in another.contents]
        self_only_words = [word_id for word_id in self.contents if word_id not in common_words]
        another_only_words = [word_id for word_id in another.contents if word_id not in common_words]
        for word_id in common_words:
            new_cluster_contents[word_id] = self.contents[word_id] + another.contents[word_id]
        for word_id in self_only_words:
            new_cluster_contents[word_id] = self.contents[word_id]
        for word_id in another_only_words:
            new_cluster_contents[word_id] = another.contents[word_id]
        # Return new Cluster
        return Cluster(cluster_id=cluster_id, docs=new_cluster_docs, word_frequencies=new_cluster_contents, vocabulary=vocabulary)
    
    def compute_theme(self, vocabulary: Vocabulary) -> None:
        # Rank words by TF-IDF scores in Cluster
        # TODO: finish
        return

@dataclass
class Level:
    level_id: int
    clusters: Dict[int, Cluster] # {cluster_id: Cluster}
    num_clusters: int

    def __str__(self, verbosity: int=0):
        level_string = f"Level: {self.level_id}\nNumber of Clusters: {self.num_clusters}\nClusters: "
        if verbosity == 0:
            level_string += f"{list(self.clusters.keys())[:NUMBER_ITEMS_TO_PRINT]}...\n"
        elif verbosity == 1:
            level_string += f"{list(self.clusters.keys())}\n"
        else:
            return "Verbosity level not recognized, please choose a supported verbosity level.\n"
        return level_string

class Distance_Matrix:
    distances: Dict[FrozenSet[int], float] # {(cluster_a_id, cluster_b_id): distance}

    def distance(self, a: Cluster, b: Cluster) -> float:
        """
        Computes and returns TF-IDF distance between two clusters.
        """
        similarity = 0
        for word_id in a.contents:
            if word_id in b.contents:
                similarity += a.contents[word_id] * b.contents[word_id]
                # == TFIDF_a * TFIDF_b
        if a.norm == 0:
            print(f"Cluster {a.cluster_id} has norm zero.")
            return 0.0
        elif b.norm == 0:
            print(f"Cluster {b.cluster_id} has norm zero.")
            return 0.0
        else:
            return 1 - (similarity / (a.norm * b.norm))

    def __init__(self, level: Level):
        '''
        Creates a distance matrix for the entire level.
        Made to be updatable.

        The distances are stored as:
        {frozenset(cluster_a_id, cluster_b_id): distance}
        '''
        self.distances = {}
        for a in range(len(level.clusters)):
            cluster_a_id = list(level.clusters.keys())[a]
            for b in range(a+1, len(level.clusters)):
                cluster_b_id = list(level.clusters.keys())[b]
                self.distances[frozenset({cluster_a_id, cluster_b_id})] = self.distance(a=level.clusters[cluster_a_id], b=level.clusters[cluster_b_id])
        return
    
    def __str__(self) -> str:
        return str(self.distances)
    
    def remove_cluster(self, cluster_to_remove: int) -> None:
        self.distances = {key:self.distances[key] for key in self.distances if cluster_to_remove not in key}
        return
    
    def add_cluster(self, new_cluster_id: int, current_level: Level) -> None:
        other_cluster_ids = list(current_level.clusters.keys())
        other_cluster_ids.remove(new_cluster_id)
        for other_cluster_id in other_cluster_ids:
            self.distances[frozenset({new_cluster_id, other_cluster_id})] = self.distance(a=current_level.clusters[new_cluster_id], b=current_level.clusters[other_cluster_id])
        return

'''
depricated - used word_frequenceis, now using tfidf scores

    def merge(self, another: "Cluster", cluster_id: int) -> "Cluster":
        new_cluster_docs = self.docs + another.docs
        # New cluster frequenceis are weighted average of previous cluster frequenceis w/weighting equal to number of documents in each
        new_cluster_contents = deepcopy(self.contents)
        weight_denominator = len(self.docs) / float(len(self.docs) + len(another.docs))
        # d.update((k,s.index(k)) for k in d.iterkeys())
        new_cluster_contents.update((key, new_cluster_contents[key] * (len(self.docs) / weight_denominator)) for key in new_cluster_contents.keys())
        # Gather word frequencies
        for word_id in another.contents:
            if word_id in new_cluster_contents:
                new_cluster_contents[word_id] += another.contents[word_id] * (len(another.docs) / weight_denominator)
            else:
                new_cluster_contents[word_id] = another.contents[word_id] * (len(another.docs) / weight_denominator)
        # Return new Cluster
        return Cluster(cluster_id=cluster_id, docs=new_cluster_docs, contents=new_cluster_contents)

    def similarity(self, a: Cluster, b: Cluster, vocabulary: Vocabulary) -> float:
        """
        Computes and returns TF-IDF similarity between two clusters.
        """
        similarity = 0
        for word_id in a.contents:
            if word_id in b.contents:
                similarity += (a.contents[word_id] * b.contents[word_id]) / (vocabulary.id_count[word_id] ** 2)
                # (a_word_freq + b_word_freq) / vocab_word_freq^2 ==
                # (a_word_freq / vocab_word_freq) * (b_word_freq / vocab_word_freq)
                # == TFIDF_a * TFIDF_b
        if a.norm == 0:
            print(f"Cluster {a.cluster_id} has norm zero.")
            return 0.0
        elif b.norm == 0:
            print(f"Cluster {b.cluster_id} has norm zero.")
            return 0.0
        else:
            # Normalized similarity
            return similarity / (a.norm * b.norm)
'''