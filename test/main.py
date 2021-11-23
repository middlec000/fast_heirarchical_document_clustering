import sys
sys.path.insert(1, 'C:/Users/colin/Documents/GitHub/fhdc/src/fhdc')
import pickle
from time import time
import itertools

from fhdc_model import FHDC_Model

def main():
    
    #docs = {
    #    "doc1": "one one one two", # one: 3, two: 1 -> one: 3
    #    "doc0": "one one one three three", # one: 3, three: 2 -> one: 3, three: 2
    #    "doc2": "two two three three" # two: 2, three: 2 -> two: 2, three: 2
    #}
    # Vocabulary: {one: 6, two: 2, three: 3}
    # doc1 -> one: 3/6 = 0.5
    # doc0 -> one: 3/6 = 0.5, three: 2/4 = 0.5
    # doc2 -> two: 2/1 = 1, three: 2/4 = 0.5

    filename = "C:\\Users\\colin\\Documents\\GitHub\\fhdc\\data\\dictionary_version\\docs_dict.pickle"

    time0 = time()
    docs = pickle.load(open(filename,'rb'))
    time1 = time()

    # num_docs = 5
    # docs = dict(itertools.islice(docs.items(), num_docs)) 

    print(f"Loading Corpus (s): {time1-time0}")

    model = FHDC_Model()

    time2 = time()
    corpus, vocabulary = model.preprocess(docs, doc_min_frequency=2, corpus_min_frequency=5, return_processed=True)
    time3 = time()

    outfile_name = "C:\\Users\\colin\\Documents\\GitHub\\fhdc\\data\\preprocessed_version\\corpus.pickle"

    outfile = open(outfile_name, 'wb')
    pickle.dump(corpus, outfile)
    outfile.close()

    print(f"Processing Corpus (s): {time3 - time2}")
    
    return

if __name__ == '__main__':
    main()
