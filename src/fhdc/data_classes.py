from numpy import array, zeros, inf
from dataclasses import dataclass
from typing import List, Dict, FrozenSet, Callable

@dataclass
class Vocabulary:
    id_word: Dict[int, str]
    word_id: Dict[str, int]
    id_freq: Dict[int, int] # {word_id: word_frequency}

    def __str__(self) -> str:
        return f"id_word={self.id_word}\nword_id={self.word_id}\nid_freq={self.id_freq}"

    # TODO: def to_json(self): return

@dataclass
class DocumentNameMap:
    name_id: Dict[str, int]
    id_name: Dict[int, str]

    def __str__(self) -> str:
        return f"name_id={self.name_id}\nid_name={self.id_name}"

    # TODO: def to_json(self): return

class ClusterContents:
    def __init__(self, cluster_id: int, contents: Dict[int, float]):
        self.cluster_id = cluster_id
        self.contents = contents # {word_id: TF-IDF}
        self.norm = self.l2_norm()
        return

    def __str__(self) -> str:
        return f"Cluster ID: {self.cluster_id}\nContents: {self.contents}"

    def l2_norm(self) -> float:
        """
        Computes L2 (Euclidean) norm for a cluster. See more at:
        https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

        Returns:
            float: The L2 norm of the cluster.
        """
        return sum([self.contents[word_id] ** 2 for word_id in self.contents]) ** (1/2)

    def merge_with(self, another: "ClusterContents", new_cluster_id: int) -> "ClusterContents":
        """
        Merges the current cluster with 'another' and returns the resulting cluster.

        Args:
            another: (ClusterContents): Another ClusterContents object to merge with this one.
            new_cluster_id (int): The id for the new cluster to return.

        Returns:
            (ClusterContents): New cluster that is result of merging current cluster with 'another'.
        """
        # Child cluster contents consist of sum of TF-IDF scores of parent clusters
        new_cluster_contents = {}
        set_self = set(self.contents.keys())
        set_another = set(another.contents.keys())
        for word_id in set_self.intersection(set_another):
            new_cluster_contents[word_id] = self.contents[word_id] + another.contents[word_id]
        for word_id in set_self.difference(set_another):
            new_cluster_contents[word_id] = self.contents[word_id]
        for word_id in set_another.difference(set_self):
            new_cluster_contents[word_id] = another.contents[word_id]
        return ClusterContents(cluster_id=new_cluster_id, contents=new_cluster_contents)

    # TODO: def to_json(self): return

@dataclass
class ClusterNode:
    cluster_id: int
    docs: List[int] # The original documents contained in cluster
    parents: List[int] # The two immediate parent clusters
    intermediate_ancestors: List[int] # All intermediate clusters in this cluster's history (not original docs or immediate parents)

    def __str__(self):
        return f"Cluster ID: {self.cluster_id}\nOriginal Documents: {self.docs}\nParent Clusters: {self.parents}"

    def get_theme(self, vocabulary: Vocabulary, corpus: Dict[int, ClusterContents], doc_name_map: Dict[int, str], num_words: int=10) -> List[str]:
        """
        Sorts the Cluster contents by TF-IDF score and saves the resulting string as a list of strings in Cluster.theme.

        Args:
            vocabulary (Vocabulary): Vocabulary for the corpus. Needed to translate word ids back into string words.
            corpus (Dict[int, Dict[int, float]]): Corpus of original documents. {doc_id: {word_id: TF-IDF}}
            num_words (int, Optional): Number of words to include in the cluster theme, defaults to 10.
        Returns:
            (List(str)): List of cluster theme words.
        """
        # Rebuild cluster contents from original docs
        cluster_contents = {}
        for word_id in vocabulary.id_word:
            for doc in self.docs:
                if word_id in corpus[doc].contents:
                    if word_id in cluster_contents:
                        cluster_contents[word_id] += corpus[doc].contents[word_id]
                    else:
                        cluster_contents[word_id] = corpus[doc].contents[word_id]
        for word_id in cluster_contents:
            cluster_contents[word_id] = cluster_contents[word_id] / vocabulary.id_freq[word_id]
        # Get theme words
        words_sorted_by_tfidf = {vocabulary.id_word[k]: v for k, v in sorted(cluster_contents.items(), key=lambda item: item[1], reverse=True)}
        return {'words': list(words_sorted_by_tfidf.keys())[:num_words], 'docs': [doc_name_map.id_name[doc_id] for doc_id in self.docs]}

    # TODO: def to_json(self): return

class Distance_Matrix:
    distances: Dict[FrozenSet[int], float] # {(cluster_a_id, cluster_b_id): distance}
    distance: Callable[[ClusterContents, ClusterContents], float]

    def __init__(self, cluster_contents: Dict[int, ClusterContents], distance: Callable=None):
        """
        Creates a distance matrix for the entire corpus of cluster_contents.
        Distances are stored in a dictionary to be quickly updated.

        The distances are stored as:
        {frozenset(cluster_a_id, cluster_b_id): distance}

        Args:
            cluster_contents Dict[int, ClusterContents]): The cluster_contents over which to compute distances.
            distance (function): Distance function to be used.
        """
        if not distance:
            distance = self.dot_product_distance
        self.distances = {}
        cluster_contents_keys = list(cluster_contents.keys())
        for a in range(len(cluster_contents_keys)):
            cluster_a_id = cluster_contents_keys[a]
            for b in range(a+1, len(cluster_contents_keys)):
                cluster_b_id = cluster_contents_keys[b]
                self.distances[frozenset({cluster_a_id, cluster_b_id})] = self.distance(a=cluster_contents[cluster_a_id], b=cluster_contents[cluster_b_id])
        return
    
    def __str__(self) -> str:
        return str(self.distances)
    
    def remove_cluster(self, cluster_to_remove: int) -> None:
        """Removes specified cluster from DistanceMatrix.

        Args:
            cluster_to_remove (int): The id of the cluster to remove.
        """
        self.distances = {key:self.distances[key] for key in self.distances if cluster_to_remove not in key}
        return

    def add_cluster(self, new_cluster_id: int, clusters: Dict[int, ClusterContents]) -> None:
        """
        Adds new cluster to the DistanceMatrix.

        Args:
            new_cluster_id (int): The id of the new cluster.

            clusters (Dict[int, ClusterContents]): The current clusters and their contents.
        """
        other_cluster_ids = list(clusters.keys())
        other_cluster_ids.remove(new_cluster_id)
        for other_cluster_id in other_cluster_ids:
            self.distances[frozenset({new_cluster_id, other_cluster_id})] = self.distance(a=clusters[new_cluster_id], b=clusters[other_cluster_id])
        return

    def get_min(self):
        """
        Finds and returns the minimum distance cluster pair and distance.

        Returns:
            merge_clusters (Tuple(int, int)): The cluster_id's of the two minimum distance clusters.
            merge_dist (float): Distance between the two merge_clusters.
        """
        merge_dist = inf
        merge_clusters = None
        for cluster_pair, distance in self.distances.items():
            if distance < merge_dist:
                merge_dist = distance
                merge_clusters = cluster_pair
        return merge_clusters, merge_dist

    def dot_product_distance(self, a: ClusterContents, b: ClusterContents) -> float:
        """
        Computes and returns dot product distance based on TF-IDF scores between two clusters.
        https://en.wikipedia.org/wiki/Dot_product

        Args:
            a (ClusterContents): The first cluster to compute distance from.
            b (ClusterContents): The second cluster to compute distance to.

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

    def to_numpy(self) -> array:
        """
        Return a numpy array with all the distances in DistanceMatrix.

        Returns:
            numpy_arr (numpy.array): Distance matrix.
        """
        numpy_arr = zeros(shape=(len(self.distances), len(self.distances)))
        for ids in self.distances:
            id1, id2 = ids
            numpy_arr[id1, id2] = self.distances[ids]
            numpy_arr[id2, id1] = self.distances[ids]
        return numpy_arr
    
    # TODO: def to_json(self): return