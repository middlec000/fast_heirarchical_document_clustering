from preprocessing import preprocess
from heirarchical_clustering_model import HeirarchicalClusteringModel

def main():
    docs = {
        0: 'Hello my name is Colin. Did I tell you my name?', 
        2:"I'm fun and    cool, hello!", 
        7:'Just a normal 98-year old! 700ppm. Normal Normal... and old.', 
        8:'What is a name? Years of old old sounds and repeated symbols...',
        3:'I I I have repeated repeated my90 self my..'
    }

    corpus, vocabulary = preprocess(docs, min_frequency=2)

    print()
    print('Corpus')
    print(corpus)
    document = 0
    print(f'Document {document}')
    print(corpus.docs[document])
    print('Vocabulary')
    print(vocabulary)
    print()

    model = HeirarchicalClusteringModel(corpus=corpus, vocabulary=vocabulary)

    model.cluster()

    print(model.levels)

    return

if __name__ == '__main__':
    main()
