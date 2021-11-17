from numpy import array, zeros
from dataclasses import dataclass
from typing import List, Dict, FrozenSet

NUMBER_ITEMS_TO_PRINT = 10

@dataclass
class Corpus:
    docs: Dict[int, "Cluster"] # {cluster_id: document}

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

class Cluster:
    cluster_id: int
    docs: List[int] # doc_ids
    contents: Dict[int, float] # {word_id: TF-IDF}
    norm: float
    theme: List[str]

    def norm(self) -> float:
        """Computes L2 (Euclidean) norm for a cluster. See more at:
        https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

        Returns:
            float: The L2 norm of the cluster.
        """
        return sum([self.contents[word_id] ** 2 for word_id in self.contents]) ** (1/2)
    
    def __init__(self, cluster_id: int, docs: List[int], contents: Dict[int, float]):
        self.cluster_id = cluster_id
        self.docs = docs
        self.contents = contents
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
    
    def merge(self, another: "Cluster", new_cluster_id: int) -> "Cluster":
        new_cluster_docs = self.docs + another.docs
        # Child cluster contents consists of sum of tfidf scores of parent clusters
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
        return Cluster(cluster_id=new_cluster_id, docs=new_cluster_docs, contents=new_cluster_contents)
    
    def compute_theme(self, vocabulary: Vocabulary, num_to_take: int=NUMBER_ITEMS_TO_PRINT) -> None:
        sorted_by_tfidf = {vocabulary.id_word[k]: v for k, v in sorted(self.contents.items(), key=lambda item: item[1])}
        theme_words = list(sorted_by_tfidf.keys())[:num_to_take]
        cluster_theme = []
        for theme_word in theme_words:
            cluster_theme += f'{theme_word}: {sorted_by_tfidf[theme_word]}'
        self.theme = cluster_theme
        return

class Level:
    level_id: int
    clusters: Dict[int, Cluster] # {cluster_id: Cluster}
    num_clusters: int

    def __init__(self, level_id: int, clusters: Dict[int, Cluster]):
        self.level_id = level_id
        self.clusters = clusters
        self.num_clusters = len(clusters)

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
        """Computes and returns dot product distance based on TF-IDF scores between two clusters.

        Args:
            a (Cluster): The first cluster to compute distance from.
            b (Cluster): The second cluster to compute distance to.

        Returns:
            float: Distance between clusters a and b.
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
        Distances are stored in a dictionary to be quickly updatable.

        The distances are stored as:
        {frozenset(cluster_a_id, cluster_b_id): distance}

        Args:
            level (Level): The level over which to compute distances.
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
        """Efficiently removes specified cluster from DistanceMatrix.

        Args:
            cluster_to_remove (int): The id of the cluser to remove.
        """
        self.distances = {key:self.distances[key] for key in self.distances if cluster_to_remove not in key}
        return
    
    def add_cluster(self, new_cluster_id: int, current_level: Level) -> None:
        """Efficiently add new cluster to DistanceMatrix.

        Args:
            new_cluster_id (int): The id of the new cluster.
            current_level (Level): The current level with all it's clusters.
        """
        other_cluster_ids = list(current_level.clusters.keys())
        other_cluster_ids.remove(new_cluster_id)
        for other_cluster_id in other_cluster_ids:
            self.distances[frozenset({new_cluster_id, other_cluster_id})] = self.distance(a=current_level.clusters[new_cluster_id], b=current_level.clusters[other_cluster_id])
        return

    def to_numpy(self) -> array:
        """Return a numpy array with all the distances in DistanceMatrix.

        Returns:
            array: Distance matrix.
        """
        numpy_arr = zeros(shape=(len(self.distances), len(self.distances)))
        for ids in self.distances:
            numpy_arr[ids[0], ids[1]] = self.distance[ids]
            numpy_arr[ids[1], ids[0]] = self.distance[ids]
        return numpy_arr

