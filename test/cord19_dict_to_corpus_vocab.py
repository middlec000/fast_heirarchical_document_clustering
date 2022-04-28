import sys
sys.path.insert(1, 'C:/Users/colin/Documents/GitHub/fhdc/src/fhdc')
import pickle
from time import time
# import itertools
from fhdc_model import FHDC_Model

if __name__=="__main__":
    filename = "C:\\Users\\colin\\Documents\\GitHub\\fhdc\\data\\dictionary_version\\docs_dict.pickle"

    time0 = time()
    docs = pickle.load(open(filename,'rb'))
    time1 = time()

    # num_docs = 5
    # docs = dict(itertools.islice(docs.items(), num_docs)) 

    print(f"Load Corpus Dictionary (s): {time1-time0}")

    model = FHDC_Model()

    time2 = time()
    corpus, vocabulary = model.preprocess(
        docs=docs, 
        doc_min_frequency=2, 
        corpus_min_frequency=5, 
        return_processed=True
    )
    time3 = time()

    # Save corpus
    outfile_name = "C:\\Users\\colin\\Documents\\GitHub\\fhdc\\data\\preprocessed_version\\corpus.pickle"
    outfile = open(outfile_name, 'wb')
    pickle.dump(corpus, outfile)
    outfile.close()

    # Save vocabulary
    outfile_name = "C:\\Users\\colin\\Documents\\GitHub\\fhdc\\data\\preprocessed_version\\vocabulary.pickle"
    outfile = open(outfile_name, 'wb')
    pickle.dump(vocabulary, outfile)
    outfile.close()

    print(f"Processing Corpus (s): {time3 - time2}")