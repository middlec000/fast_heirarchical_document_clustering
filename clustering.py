from typing import Dict
from copy import copy
from data_classes import \
    Corpus, \
    Vocabulary, \
    Similarity_Matrix, \
    Level, \
    Cluster

def similarity(a: Cluster, b: Cluster, vocabulary: Vocabulary) -> float:
    similarity = 0
    for word_id in a.contents:
        if word_id in b.contents:
            similarity += (a.contents[word_id] * b.contents[word_id]) / (vocabulary.id_count[word_id] ** 2)
            # (a_word_freq + b_word_freq) / word_freq^2 ==
            # (a_word_freq / word_freq) * (b_word_freq / word_freq)
            # == TFIDF_a * TFIDF_b
    if a.norm == 0:
        print(f"Cluster {a.cluster_id} has norm zero.")
        return 1
    elif b.norm == 0:
        print(f"Cluster {b.cluster_id} has norm zero.")
        return 1
    else:
        # Normalized similarity
        return similarity / (a.norm * b.norm)

def get_similarity_matrix(level: Level, vocabulary: Vocabulary) -> Similarity_Matrix:
    similarity_matrix = Similarity_Matrix(similarities={})
    for a in range(len(level.clusters)):
        cluster_a_id = level.clusters.keys()[a]
        for b in range(a+1, len(level.clusters)):
            cluster_b_id = level.clusters.keys()[b]
            similarity_matrix[tuple(cluster_a_id, cluster_b_id)] = similarity(a=level.clusters[cluster_a_id], b=level.clusters[cluster_b_id], vocabulary=vocabulary)
    return similarity_matrix

def agglomerative_cluster(corpus: Corpus, vocabulary: Vocabulary) -> Dict[int, Level]:
    levels = {}
    cluster_id = 0
    # Create initial Level
    current_level = Level(level_id=0, clusters={}, num_clusters=0)
    for doc_id in corpus.docs:
        # Create initial Clusters - each Doc is a Cluster
        current_level.clusters[cluster_id] = Cluster(cluster_id=cluster_id, docs=[doc_id], contents=corpus.docs[doc_id])
        current_level.num_clusters += 1
        cluster_id += 1
    levels[current_level.level_id] = copy(current_level)

    # Loop agglomerating clusters until there is only one cluster in current_level
    while current_level.num_clusters > 1:
        # Incriment IDs for new objects
        current_level.level_id += 1
        cluster_id += 1

        # Find most similar clusters to combine
        similarity_matrix = get_similarity_matrix(level=current_level, vocabulary=vocabulary)
        cluster_a_id, cluster_b_id = max(similarity_matrix, key=similarity_matrix.get)
        a_cluster = current_level.clusters[cluster_a_id]
        b_cluster = current_level.clusters[cluster_b_id]

        # Create new Cluster
        new_cluster_docs = a_cluster.docs + b_cluster.docs
        new_cluster_contents = a_cluster.contents
        # Gather word frequencies
        for word_id in b_cluster.contents:
            if word_id in new_cluster_contents:
                new_cluster_contents[word_id] += b_cluster.contents[word_id]
            else:
                new_cluster_contents[word_id] = b_cluster.contents[word_id]
        # Add new Cluster to current_level
        current_level.clusters[cluster_id] = Cluster(cluster_id=cluster_id, docs=new_cluster_docs, contents=new_cluster_contents)

        # Remove merged Clusters
        del current_level.clusters[cluster_a_id]
        del current_level.clusters[cluster_b_id]

        # Decriment number of Clusters
        current_level.num_clusters -= 1

        # Add current_level to levels
        levels[current_level.level_id] = copy(current_level)

    return levels