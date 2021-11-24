from typing import Dict, List
from data_classes import \
    Vocabulary, \
    Corpus, \
    Level, \
    Cluster
from clustering import agglomerative_cluster
from preprocessing import preprocess

num_items_to_print = 10

class FHDC_Model():
    vocabulary: Vocabulary #TODO: possibly remove this
    corpus: Corpus
    levels: Dict[int, Level] # {level_id: Level}

    def __init__(self):
        self.corpus = None
        self.vocabulary = None
        self.levels = None
        self.distance_matrix = None
        return

    def __str__(self, verbosity: int=0) -> str:
        s = f"Levels:\n"
        for i in range(num_items_to_print):
            if i in self.levels:
                s += self.levels[i].__str__(vocabulary=self.vocabulary, verbosity=verbosity)
        s += '...\n'
        return s
    
    #def preprocess(self, docs: Dict[int, str], doc_min_frequency: int=2, corpus_min_frequency: int=2, return_processed: bool=False):
    def preprocess(self, docs: Dict[str, str], return_processed: bool=False, **kwargs):
        """Preprocess documents with input data format of a dictionary with keys of document ids and string values of document contents to an output data format where each document is represented by a dictionary with keys of the word ids in that document and values of the TF-IDF score for that word in that document.

        Args:
            docs (Dict[int, str]): Collection of documents to cluster in simple data format.

            doc_min_frequency (int, optional): Minimum frequency a word must have in a document in order to be retained in the output data format. Defaults to 2.

            corpus_min_frequency (int, optional): Minimum frequency a word must have corpus wide in order to be retained in the output data format. Defaults to 2.

            return_processed (bool, optional): Specify True if the Corpus adn Vocabulary are to be returned from this method call. Defaults to False.

        Returns:
            corpus (Corpus): Collection of documents in output data format. This is only returned if return_processed is set to True.
            
            vocablary (Vocabulary): Collection of all the words in the corpus, their frequencies, and a mapping of each word to it's word id. This is only returned if return_processed is set to True.
        """
        # self.corpus, self.vocabulary = preprocess(docs=docs, doc_min_frequency=kwargs['doc_min_frequency'], corpus_min_frequency=kwargs['corpus_min_frequency'], tfidf_decimals=kwargs['tfidf_decimals'], stop_words=kwargs['stop_words'], lemmatizer=kwargs['lemmatizer'])
        self.corpus, self.vocabulary = preprocess(docs=docs, kwargs=kwargs)
        if return_processed:
            return self.corpus, self.vocabulary
    
    def load_preprocessed(self, corpus: Corpus, vocabulary: Vocabulary):
        """Load preprocessed data into model.

        Args:
            corpus (Corpus): Collection of preprocessed documents in specific Corpus data type.
            vocabulary (Vocabulary): Vocabular for corpus in specific Vocabulary data type.
        """
        self.corpus = corpus
        self.vocabulary = vocabulary
        return
    
    def cluster(self, cluster_type: str='agglomerative', stop_num_clusters: int=None)-> None:
        """Cluster documents in corpus using the specified cluster_type.

        Args:
            cluster_type (str, optional): Chosen clustering method. Defaults to 'agglomerative'.
            stop_num_clusters (int, optional): Controls if and when to stop clustering early. Passed on to the specific clustering method used. Defaults to None.
        """
        if self.corpus is None:
            print("You must add your documents to the model before clustering can be performed.")
            return
        if cluster_type == 'agglomerative':
            self.levels, self.distance_matrix = agglomerative_cluster(self.corpus, stop_num_clusters=stop_num_clusters)
        else:
            print('Clustering type not supported.\n')
        return
    
    def get_cluster(self, cluster_id: int) -> Cluster:
        """Returns a cluster with the specified cluster_id in any Level

        Args:
            cluster_id (int): ID of the cluster to be located and returned.

        Returns:
            Cluster: The specified cluster.
        """
        for level in self.levels:
            if cluster_id in self.levels[level].clusters:
                return self.levels[level].clusters[cluster_id]

    def summary(self, levels: List=None, clusters: List=None, verbosity: int=0) -> str:
        """Creates and returns a summary string of the clustering history. The summary can focus on specific levels or clusters by using the optional levels and clusters variables. The detail of the summary is determined by the verbosity variable.

        Args:
            levels (List, optional): Specific levels to summarize. Defaults to None.
            clusters (List, optional): Specific clusters to summarize. Defaults to None.
            verbosity (int, optional): Level of detail to provide. Defaults to 0.
                TODO: add to verbosity description

        Returns:
            str: Summary of clustering history.
        """

        cumulative_string = ''
        if (levels is None) and (clusters is None):
            return self.__str__(verbosity=verbosity)
        elif not (levels is None) and not (clusters is None):
            cumulative_string += 'Please choose either level(s) or cluster(s) to summarize.\n'
        else:
            if not (levels is None):
                for level in levels:
                    cumulative_string += self.levels[level].__str__(vocabulary=self.vocabulary, verbosity=verbosity)
            elif not (clusters is None):
                for cluster_id in clusters:
                    cumulative_string += self.get_cluster(cluster_id=cluster_id).__str__(vocabulary=self.vocabulary, verbosity=verbosity)
        return cumulative_string
        #TODO: add print to file
        #TODO: create a summary object?

