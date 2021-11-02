from typing import Dict
from data_classes import \
    Vocabulary, \
    Corpus, \
    Level
from clustering import agglomerative_cluster

class HeirarchicalClusteringModel():
    vocabulary: Vocabulary
    corpus: Corpus
    levels: Dict[int, Level] # {level_id: Level}

    def __init__(self, corpus: Corpus, vocabulary: Vocabulary):
        self.corpus = corpus
        self.vocabulary = vocabulary
        self.levels = None
        return

    def __str__(self):
        heirarchicalclusteringmodel_string = f"Levels: {self.levels}...\n"
        return heirarchicalclusteringmodel_string
    
    def cluster(self, cluster_type:str='agglomerative'):
        if cluster_type == 'agglomerative':
            self.levels = agglomerative_cluster(self.corpus, self.vocabulary)
        else:
            print('Selected cluster_type not supported yet.')
        return
