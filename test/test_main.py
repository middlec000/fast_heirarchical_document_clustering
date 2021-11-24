import sys
sys.path.insert(1, 'C:/Users/colin/Documents/GitHub/fhdc/src/fhdc')
from fhdc_model import FHDC_Model

def main():
    
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

    model = FHDC_Model()

    corpus, vocabulary = model.preprocess(docs=docs, return_processed=True)

    # print(f'Corpus:\n{corpus}')

    # print(f'Vocabulary:\n{vocabulary}')

    model.cluster()

    print(model.summary(verbosity=2))
    
    return

if __name__ == '__main__':
    main()
