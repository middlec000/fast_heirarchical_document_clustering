import os
import sys
sys.path.insert(1, 'C:/Users/colin/Documents/GitHub/fhdc/src/fhdc')
from fhdc_model import FHDC_Model

if __name__ == '__main__':
    
    docs = {
        "doc1": "one one one two", # one: 3, two: 1 -> one: 3
        "doc1.4": "it is such a it a stop word here plaster wall plaster plaster",
        "doc0": "one one one three three", # one: 3, three: 2 -> one: 3, three: 2
        "doc2": "two two three three" # two: 2, three: 2 -> two: 2, three: 2
    }
    # Vocabulary: {one: 6, two: 2, three: 3}
    # doc1 -> one: 3/6 = 0.5
    # doc0 -> one: 3/6 = 0.5, three: 2/4 = 0.5
    # doc2 -> two: 2/1 = 1, three: 2/4 = 0.5
    '''
    docs = {}
    folder_path = 'C:/Users/colin/Documents/GitHub/fhdc/data/DataScienceCoverLetterExamples/'
    for filename in os.listdir(folder_path):
        if filename != 'source_info.txt':
            with open(folder_path + filename, 'r') as f:
                docs[filename] = f.read()
    '''

    model = FHDC_Model()
    corpus, name_map, vocabulary = model.preprocess(docs=docs, return_processed=True)

    print(f'Name Mapping:\n{name_map}')
    print(f'\nVocabulary:\n{vocabulary}')
    print('\nCorpus:')
    for cluster_id in corpus:
        print(corpus[cluster_id])

    model.cluster()

    print(f'\nCluster Nodes:')
    for cluster_id in model.cluster_nodes:
        print(model.cluster_nodes[cluster_id])
    print(f'\nLinkage Matrix:\n{model.linkage}')

    clustering_step = 2
    cluster_themes = model.get_clustering_step_themes(clustering_step=clustering_step)
    print(f'\nCluster Themes at cluster step {clustering_step}:')
    for cluster_id in cluster_themes:
        print(f'Cluster ID: {cluster_id}\nTheme Words: {cluster_themes[cluster_id]["words"]}\nCluster Documents: {cluster_themes[cluster_id]["docs"]}')

    model.plot_dendrogram()