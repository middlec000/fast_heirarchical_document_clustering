import sys
sys.path.insert(1, 'C:/Users/colin/Documents/GitHub/fhdc/src/fhdc')

from fhdc_model import FHDC_Model

def main():
    
    docs = {
        "doc1": "one one one two", # one: 3, two: 1 -> one: 3
        "doc0": "one one one three three", # one: 3, three: 2 -> one: 3, three: 2
        "doc2": "two two three three" # two: 2, three: 2 -> two: 2, three: 2
    }
    # Vocabulary: {one: 6, two: 2, three: 3}
    # doc1 -> one: 3/6 = 0.5
    # doc0 -> one: 3/6 = 0.5, three: 2/4 = 0.5
    # doc2 -> two: 2/1 = 1, three: 2/4 = 0.5

    model = FHDC_Model()

    corpus, vocabulary = model.preprocess(docs, doc_min_frequency=2, corpus_min_frequency=2, return_processed=True)

    print()
    print('Corpus')
    print(corpus.__str__(vocabulary=vocabulary, verbosity=2))
    print()
    print('Vocabulary')
    print(vocabulary)
    print()

    #model.cluster()

    #verbosity = 2
    #print(f'Clustering Summary (Verbosity = {verbosity})')
    #print(model.summary(verbosity=verbosity))
    
    return

if __name__ == '__main__':
    main()
