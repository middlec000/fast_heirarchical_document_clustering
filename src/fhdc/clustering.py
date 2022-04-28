from numpy import zeros
from typing import Dict
from copy import deepcopy
from data_classes import *

def agglomerative_cluster(corpus: Dict[int, ClusterContents], stop_num_clusters: int=1):
    """
    Clusters the given documents, or corpus, using an efficient form of dot product distance (see DistanceMatrix.distance()) in an agglomerative manner. 
    This means that initially each document is a cluster (see Cluster) and at every round of clustering the two clusters with smallest distance are combined. 
    This process is repeated until either there is one cluster remaining or there are stop_num_clusters clusters remaining.

    Args:
        corpus (Dict[int, ClusterContents]): Collection of documents to cluster. {doc_id: {word_id: TF-IDF}}
        stop_num_clusters (int, optional): Optional argument to stop clustering early when the number of clusters reaches the passed value.
    Returns:
        cluster_nodes (Dict[int, ClusterNode]): If stop_num_clusters==1, this contains only the root node of the clustering tree. If stop_num_clusters>1, this contains the collection of remaining clusters. {cluster_id: ClusterNode}
        linkages (numpy.array): Linkage matrix as defined by scipy.cluster.heirarchy.linkage(). 
            (n-1) x 4 array with columns: [cluster a id, cluster b id, merge distance, number of original observations in newly formed cluster]
    """
    # Create working data structures
    cluster_nodes = {}
    contents_of_current_clusters = deepcopy(corpus)
    # Put each of the original documents into its own cluster initially
    for doc in corpus:
        cluster_nodes[doc] = ClusterNode(
            cluster_id=doc, 
            docs=[doc], 
            parents=[], 
            intermediate_ancestors=[])
    # Create initial distance matrix
    distance_matrix = Distance_Matrix(contents_of_current_clusters)
    linkages = zeros(shape=(len(contents_of_current_clusters)-1, 4))
    merge_step = 0
    # Loop agglomerating clusters until the number of clusters reaches stop_num_clusters
    current_cluster_id = len(contents_of_current_clusters)
    while len(contents_of_current_clusters) > stop_num_clusters:
        # Find the least different clusters to combine
        (cluster_a_id, cluster_b_id), merge_distance = distance_matrix.get_min()
        cluster_a_contents = contents_of_current_clusters[cluster_a_id]
        cluster_b_contents = contents_of_current_clusters[cluster_b_id]
        # Create a new ClusterContents object and add it to the contents_of_current_clusters
        contents_of_current_clusters[current_cluster_id] = cluster_a_contents.merge_with(another=cluster_b_contents, new_cluster_id=current_cluster_id)
        # Add a new ClusterNode to the cluster_nodes
        cluster_nodes[current_cluster_id] = ClusterNode(
            cluster_id=current_cluster_id, 
            docs=cluster_nodes[cluster_a_id].docs + cluster_nodes[cluster_b_id].docs, 
            parents=[cluster_a_id, cluster_b_id],
            intermediate_ancestors=cluster_nodes[cluster_a_id].intermediate_ancestors + cluster_nodes[cluster_a_id].parents + cluster_nodes[cluster_b_id].intermediate_ancestors + cluster_nodes[cluster_b_id].parents)
        # Remove merged ClusterContents from contents_of_current_clusters
        del contents_of_current_clusters[cluster_a_id]
        del contents_of_current_clusters[cluster_b_id]
        # Remove merged clusters from Distance_Matrix
        distance_matrix.remove_cluster(cluster_a_id)
        distance_matrix.remove_cluster(cluster_b_id)
        # Add new cluster to distance matrix
        distance_matrix.add_cluster(new_cluster_id=current_cluster_id, clusters=contents_of_current_clusters)
        # Record merging in linkages
        linkages[merge_step] = [cluster_a_id, cluster_b_id, merge_distance, len(cluster_nodes[current_cluster_id].docs)]
        # Incriment IDs for new objects
        current_cluster_id += 1
        # Inciment merge_step
        merge_step += 1
    return cluster_nodes, linkages