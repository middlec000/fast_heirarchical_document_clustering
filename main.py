from preprocessing import preprocess
from heirarchical_clustering_model import HeirarchicalClusteringModel


if __name__ == '__main__':

    docs = {
        0: 'Hello my name is Colin.', 
        2:"I'm fun and    cool, hello!", 
        7:'Just a normal 98-year old! 700ppm', 
        8:'What is a name? Years of old old sounds and repeated symbols...',
        3:'I I I have repeated repeated my90 self my..'
    }

    corpus, vocabulary = preprocess(docs, min_frequency=2)

    model = HeirarchicalClusteringModel(corpus=corpus, vocabulary=vocabulary)

    print()
    print('Corpus')
    print(corpus)
    print('Document 0')
    print(corpus.docs[0])
    print('Vocabulary')
    print(vocabulary)
    print()

