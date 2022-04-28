from numpy import array
from typing import Dict, List
from copy import copy
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from data_classes import *
from clustering import *
from preprocessing import *

class FHDC_Model():
    corpus: Dict[int, ClusterContents]
    doc_name_map: DocumentNameMap
    vocabulary: Vocabulary
    cluster_nodes: Dict[int, ClusterNode]
    linkage: array

    def __init__(self):
        self.corpus = None
        self.doc_name_map = None
        self.vocabulary = None
        self.distance_matrix = None
        return

    @classmethod
    def fromPreprocessed(cls, corpus: Dict[int, ClusterContents], doc_name_map: Dict[str, int], vocabulary: Vocabulary):
        """
        Create a model by loading preprocessed data.

        Args:
            corpus (Dict[int, ClusterContents]): Collection of preprocessed documents in specific dictionary configuration.
            doc_name_map (Dict[str, int]): Mapping of passed document name to cluster (document) id.
            vocabulary (Vocabulary): Vocabulary for corpus in specific Vocabulary data type.
            
        Returns:
            model (FDHC_Model): A new model with the preprocessed data loaded into it.
        """
        model = cls()
        model.corpus = corpus
        model.doc_name_map = doc_name_map
        model.vocabulary = vocabulary
        return model

    def preprocess(self, docs: Dict[str, str], return_processed: bool=False, **kwargs):
        """
        Preprocess input documents to efficient TF-IDF vectors (dicitonaries).

        Args:
            docs (Dict[int, str]): Collection of documents to cluster in simple data format.
            doc_min_frequency (int, optional): Minimum frequency a word must have in a document in order to be retained in the output data format. Defaults to 2.
            corpus_min_frequency (int, optional): Minimum frequency a word must have corpus wide in order to be retained in the output data format. Defaults to 2.
            return_processed (bool, optional): Specify True if the corpus and Vocabulary are to be returned from this method call. Defaults to False.

        Returns:
            corpus (Dict[int, Dict[int, float]]): Collection of documents in output data format. This is only returned if return_processed is set to True.
            vocablary (Vocabulary): Collection of all the words in the corpus, their frequencies, and a mapping of each word to it's word id. This is only returned if return_processed is set to True.
        """
        self.corpus, self.doc_name_map, self.vocabulary = preprocess(docs=copy(docs), kwargs=kwargs)
        if return_processed:
            return self.corpus, self.doc_name_map, self.vocabulary
        else:
            return None

    def cluster(self, cluster_type: str='agglomerative', stop_num_clusters: int=None) -> None:
        """
        Cluster documents in corpus using the specified cluster_type.

        Args:
            cluster_type (str, optional): Chosen clustering method. Defaults to 'agglomerative'.
            stop_num_clusters (int, optional): Controls if and when to stop clustering early. Passed on to the specific clustering method used. Defaults to None.
        """
        if self.corpus is None:
            print("You must add your documents to the model before clustering can be performed.")
            return
        if cluster_type == 'agglomerative':
            if not stop_num_clusters:
                stop_num_clusters = 1
            self.cluster_nodes, self.linkage = agglomerative_cluster(corpus=self.corpus, stop_num_clusters=stop_num_clusters)
        else:
            print('Clustering type not supported.\n')
        return

    def plot_dendrogram(self) -> None:
        """
        Plot the dendrogram diagram based on the linkage matrix created during clustering.
        """
        plt.figure()
        dendrogram(self.linkage)
        plt.show()
        return

    def get_cluster_theme(self, cluster_id: int) -> List[str]:
        """
        Get the theme for a specific cluster.
        This method calls the ClusterNode.get_theme() method.

        Args:
            cluster_id (int): ID of the cluster.

        Returns:
            (str): Cluster theme.
        """
        return self.cluster_nodes[cluster_id].get_theme(vocabulary=self.vocabulary, corpus=self.corpus, doc_name_map=self.doc_name_map)

    def get_clustering_step_clusters(self, clustering_step: int) -> List[ClusterNode]:
        """
        Find all clusters at a given clustering step.

        Args:
            clustering_step (int): Clustering step to find all clusters.

        Returns:
            List[ClusterNode]: Ids of all clusters at given clustering step.
        """
        # Find all cluster_id's at or before this clustering_step
        clustering_slice = set(self.linkage[:clustering_step+1, :2].flatten().astype(int).tolist() + list(self.corpus.keys()))
        # If last clustering step, add final root cluster
        if clustering_step >= self.linkage.shape[0]:
            clustering_slice.add(max(clustering_slice)+1)
        clustering_step_cluster_ids = []
        while len(clustering_slice) > 0:
            youngest_cluster_id = max(clustering_slice)
            youngest_cluster_node = self.cluster_nodes[youngest_cluster_id]
            clustering_slice -= set([youngest_cluster_id] + youngest_cluster_node.docs + youngest_cluster_node.parents + youngest_cluster_node.intermediate_ancestors)
            clustering_step_cluster_ids.append(youngest_cluster_id)
        return clustering_step_cluster_ids

    def get_clustering_step_themes(self, clustering_step: int) -> Dict[int, List[str]]:
        """
        Get the themes of all the clusters present at a chosen clustering step.
        This method calls the get_cluster_theme() method which calls the ClusterNode.get_theme() method.

        Args:
            clustering_step (int): Which step to get cluster themes.

        Returns:
            clustering_step_theme (Dict[int, List[str]]): Themes for each cluster at the chosen clustering step. {cluster_id: [cluster_theme_words]}
        """
        clustering_step_cluster_ids = self.get_clustering_step_clusters(clustering_step=clustering_step)
        clustering_step_theme = {}
        for cluster_id in clustering_step_cluster_ids:
            clustering_step_theme[cluster_id] = self.get_cluster_theme(cluster_id=cluster_id)
        return clustering_step_theme

    # TODO: def to_json(self): return