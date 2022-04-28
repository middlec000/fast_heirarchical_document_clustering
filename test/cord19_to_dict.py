import json
import pickle
from langdetect import detect
import os
from time import time

def detect_english(text):
    # Default langauge is English (most docs are)
    # All we care about are the 'en' ones
    language = 'en'
    threshold = 100
    try:
        if len(text) > threshold:
            language = detect(text[:threshold])
        elif len(text) > 0:
            language = detect(text)
    except Exception as e:
        # If cannot detect language, drop doc
        return False
    return language == 'en'

def doc_to_dict(folder: str, num_docs: int=None) -> dict:
    files = os.listdir(folder)
    if num_docs is not None:
        files = files[:num_docs]
    corpus = {}
    for file in files:
        with open(folder + "\\" + file) as opened_file:
            doc = json.load(opened_file)
            text = ' '.join([chunk['text'].lower() for chunk in doc['body_text']])
            if detect_english(text):
                corpus[file] = text
    return corpus

if __name__=="__main__":
    folder = "C:\\Users\\colin\\Documents\\GitHub\\fhdc\\data\\archive\\document_parses\\pdf_json"

    outfile_name = "C:\\Users\\colin\\Documents\\GitHub\\fhdc\\data\\dictionary_version\\docs_dict.pickle"
    stats_record = "C:\\Users\\colin\\Documents\\GitHub\\fhdc\\data\\dictionary_version\\stats.txt"

    # total number of docs: 278,523
    # Est. time = 2:15
    num_docs = None

    time0 = time()

    corpus = doc_to_dict(folder=folder, num_docs=num_docs)

    time1 = time()

    outfile = open(outfile_name, 'wb')
    pickle.dump(corpus, outfile)
    outfile.close()

    time2 = time()

    record_string = f"Number of documents: {num_docs}\nCreating Corpus (s): {time1-time0}\nSaving Corpus (s): {time2 - time1}"
    with open(stats_record, 'w') as record_file:
        record_file.write(record_string)

    print(record_string)