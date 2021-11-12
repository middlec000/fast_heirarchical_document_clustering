from typing import Dict, List
from data_classes import \
    Vocabulary, \
    Corpus, \
    Level, \
    Cluster
from clustering import agglomerative_cluster

num_items_to_print = 10

class HeirarchicalClusteringModel():
    vocabulary: Vocabulary
    corpus: Corpus
    levels: Dict[int, Level] # {level_id: Level}

    def __init__(self, corpus: Corpus, vocabulary: Vocabulary):
        self.corpus = corpus
        self.vocabulary = vocabulary
        self.levels = None
        self.measure_matrix = None
        return

    def __str__(self, verbosity: int=0) -> str:
        heirarchicalclusteringmodel_string = f"Levels:\n"
        for i in range(num_items_to_print):
            if i in self.levels:
                heirarchicalclusteringmodel_string += self.levels[i].__str__(verbosity=verbosity)
        heirarchicalclusteringmodel_string += '...\n'
        return heirarchicalclusteringmodel_string
    
    def cluster(self, cluster_type: str='agglomerative')-> None:
        if cluster_type == 'agglomerative':
            self.levels, self.measure_matrix = agglomerative_cluster(self.corpus)
        else:
            print('Clustering type not supported.\n')
        return
    
    def get_cluster(self, cluster_id: int) -> Cluster:
        for level in self.levels:
            if cluster_id in self.levels[level].clusters:
                return self.levels[level].clusters[cluster_id]

    def get_results(self, levels: List=None, clusters: List=None, verbosity: int=0) -> str:
        #TODO: add print to file
        cumulative_string = ''
        if (levels is None) and (clusters is None):
            return self.__str__(verbosity=verbosity)
        elif not (levels is None) and not (clusters is None):
            cumulative_string += 'Please choose either levels or clusters.\n'
        else:
            if not (levels is None):
                for level in levels:
                    cumulative_string += self.levels[level].__str__(verbosity=verbosity)
            elif not (clusters is None):
                for cluster_id in clusters:
                    cumulative_string += self.get_cluster(cluster_id=cluster_id).__str__(verbosity=verbosity)
        return cumulative_string

