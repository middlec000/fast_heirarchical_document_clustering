from typing import Dict
from copy import deepcopy
from data_classes import \
    Corpus, \
    Distance_Matrix, \
    Vocabulary, \
    Level, \
    Cluster

def agglomerative_cluster(corpus: Corpus, vocabulary: Vocabulary) -> Dict[int, Level]:
    levels = {}
    cluster_id = -1
    # Create initial Level
    current_level = Level(level_id=0, clusters={}, num_clusters=0)
    for doc_id in corpus.docs:
        # Incremend cluster ID
        cluster_id += 1
        # Create initial Clusters - each Doc becomes a Cluster
        current_level.clusters[cluster_id] = Cluster(cluster_id=cluster_id, docs=[doc_id], word_frequencies=corpus.docs[doc_id].contents, vocabulary=vocabulary)
    current_level.num_clusters = len(current_level.clusters)
    levels[current_level.level_id] = deepcopy(current_level)

    # Compute distance matrix
    distance_matrix = Distance_Matrix(level=current_level)

    # Loop agglomerating clusters until there is only one cluster in current_level
    while current_level.num_clusters > 1:
        # Incriment IDs for new objects
        current_level.level_id += 1
        cluster_id += 1

        print(distance_matrix)
        # Find least different clusters to combine
        cluster_a_id, cluster_b_id = min(distance_matrix.distances, key=distance_matrix.distances.get)

        a_cluster = current_level.clusters[cluster_a_id]
        b_cluster = current_level.clusters[cluster_b_id]

        # Create new cluster and add it to the level's clusters
        current_level.clusters[cluster_id] = a_cluster.merge(another=b_cluster, cluster_id=cluster_id, vocabulary=vocabulary)

        # Remove merged Clusters
        # ...from Level
        del current_level.clusters[cluster_a_id]
        del current_level.clusters[cluster_b_id]
        # and from Distance_Matrix
        distance_matrix.remove_cluster(cluster_a_id)
        distance_matrix.remove_cluster(cluster_b_id)

        # Add new cluster to distance matrix
        distance_matrix.add_cluster(new_cluster_id=cluster_id, current_level=current_level)

        # Adjust (decrement) number of Clusters
        current_level.num_clusters = len(current_level.clusters)

        # Add current_level to levels
        levels[current_level.level_id] = deepcopy(current_level)

    # return levels, distance_matrix
    return levels, distance_matrix

