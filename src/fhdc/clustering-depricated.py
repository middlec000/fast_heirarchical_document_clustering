from typing import Dict
from copy import deepcopy
from data_classes import *

def agglomerative_cluster(corpus: Corpus, stop_num_clusters: int=None) -> Dict[int, Level]:
    """
    Clusters the given documents, or corpus, using an efficient form of dot product distance (see DistanceMatrix.distance()) in an agglomerative manner. This means that initially each document is a cluster (see Cluster) and at every round of clustering the two clusters with smallest distance are combined. This process is repeated until either there is one cluster remaining or there are stop_num_clusters clusters remaining.

    Args:
        corpus (Corpus): Collection of documents to cluster.

        stop_num_clusters (int, optional): Optional argument to stop clustering early when the number of clusters reaches the passed value.

    Returns:
        Dict[int, Level]: Dictionary representing the clustering history as a series of levels (see Level), each containing the clusters at that level.
    """

    if stop_num_clusters is None:
        stop_num_clusters = 1

    levels = {}
    # Create initial Level
    current_level = Level(level_id=0, clusters=corpus.docs, merged_clusters=[], new_cluster=corpus.docs.keys())
    cluster_id = max(current_level.clusters.keys())

    levels[current_level.level_id] = deepcopy(current_level)

    # Compute distance matrix
    distance_matrix = Distance_Matrix(level=current_level)
    initial_distance_matrix = distance_matrix.to_numpy()

    # Loop agglomerating clusters until the number of clusters in current_level reaches stop_num_clusters
    while current_level.num_clusters > stop_num_clusters:
        # Incriment IDs for new objects
        current_level.level_id += 1
        cluster_id += 1

        # Find least different clusters to combine
        cluster_a_id, cluster_b_id = min(distance_matrix.distances, key=distance_matrix.distances.get)

        a_cluster = current_level.clusters[cluster_a_id]
        b_cluster = current_level.clusters[cluster_b_id]
        merged_clusters = [a_cluster.cluster_id, b_cluster.cluster_id]

        # Create new cluster and add it to the level's clusters
        current_level.clusters[cluster_id] = a_cluster.merge(another=b_cluster, new_cluster_id=cluster_id)

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
        # Update merged clusters
        current_level.merged_clusters = merged_clusters
        # Update new cluster
        current_level.new_cluster = cluster_id

        # Add current_level to levels
        levels[current_level.level_id] = deepcopy(current_level)

    # return levels, distance_matrix
    return levels, initial_distance_matrix

