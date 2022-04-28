from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# Note: Must download stuff for stopwords:
# showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
import re
import string
from typing import Dict
from data_classes import *

def preprocess(docs: Dict[str,str], **kwargs):
    """
    Takes as input a dictionary of the documents for clustering and transforms them into the compact data structures of corpus and Vocabulary in preperation for clustering. 
    This process turns each document from a string of words into a dictionary with (k, v) of (word id, TF-IDF score).

    Args:
        docs (Dict[str,str]): Documents to preprocess. Must be in format {doc_id (str): doc_text (str)}
        **kwargs:
            corpus_min_frequency (optional): Minimum corpus wide frequency each word needs to meet in order to be retained in clustering. Defaults to 2.
            doc_min_frequency (optional): Minimum document frequency each word needs to meet in order to be retained in each document. Defaults to 2.
            tfidf_decimals (optional): Number of decimal places to round TF-IDF scores. Defaults to 4.
            stop_words (optional): Words to remove from corpus. If none is provided, the default list of nltk english stopwords is used by default.
            lemmatizer (optional): Lemmatizer to be used. Must have a .lematize(word) function. If none is provided, nltk's WordNetLemmatizer is used by default.

    Returns:
        corpus (Dict[int, ClusterContents]): The document corpus where each document is represented as a dictionary of word_id's and the corresponding TF-IDF scores.
        doc_name_map (Dict[str, int]): Mapping of passed document name to cluster (document) id.
        vocabulary (Vocabulary): The vocabulary for the given corpus.
    """
    # Establish parameter values
    params = {'corpus_min_frequency':2, 'doc_min_frequency':2, 'tfidf_digits':4, 'stop_words': set(stopwords.words('english')), 'lemmatizer': WordNetLemmatizer()}
    if kwargs is not None:
        for k,v in kwargs.items():
            params[k] = v
    
    # print(params)

    for doc in docs:
        # Lowercase
        current_doc = docs[doc].lower()
        # Remove punctuation and symbols
        regex = re.compile(f"[{re.escape(string.punctuation)}]")
        current_doc = regex.sub('', current_doc)
        # Remove numbers
        current_doc = re.sub('\d', '', current_doc)
        # Tokenize
        current_doc = current_doc.split(' ')
        # Remove stopwords and empty strings
        current_doc = [word for word in current_doc if word not in params['stop_words'] and word]
        # Lemmatize
        current_doc = [params['lemmatizer'].lemmatize(word) for word in current_doc]
        # Transform to vector format {word: frequency}
        transformed_doc = {}
        for word in current_doc:
            if word not in transformed_doc:
                transformed_doc[word] = 1
            else:
                transformed_doc[word] += 1
        # Remove low frequency words from doc
        transformed_doc = {k:v for (k,v) in transformed_doc.items() if v >= params['doc_min_frequency']}
        # Replace the original doc with transformed_doc
        docs[doc] = transformed_doc

    # Create vocabulary
    vocabulary = Vocabulary({}, {}, {})
    current_word_id = 0
    for doc in docs:
        for word in docs[doc]:
            if word in vocabulary.word_id:
                existing_id = vocabulary.word_id[word]
                vocabulary.id_freq[existing_id] += docs[doc][word]
            else:
                vocabulary.word_id[word] = current_word_id
                vocabulary.id_word[current_word_id] = word
                vocabulary.id_freq[current_word_id] = docs[doc][word]
                current_word_id += 1

    # Find corpus-wide low-frequency words
    infrequent_corpus_word_ids = []
    for word_id in vocabulary.id_freq:
        if vocabulary.id_freq[word_id] < params['corpus_min_frequency']:
            infrequent_corpus_word_ids.append(word_id)

    # Remove corpus-wide low-frequency words from vocabulary
    for word_id_to_drop in infrequent_corpus_word_ids:
            vocabulary.id_freq.pop(word_id_to_drop)
            word_to_drop = vocabulary.id_word[word_id_to_drop]
            vocabulary.id_word.pop(word_id_to_drop)
            vocabulary.word_id.pop(word_to_drop)

    # Remove corpus-wide low-frequency words from corpus
    # Change words to word_ids
    # Transform word frequencies to TF-IDF scores
    # Create clusters, cluster_ids
    doc_name_map = DocumentNameMap({}, {})
    cluster_id = 0
    new_docs = {}
    for doc in docs:
        cluster_contents = {}
        for word in docs[doc]:
            if word in vocabulary.word_id:
                word_id = vocabulary.word_id[word]
                word_tfidf = float(docs[doc][word]) / float(vocabulary.id_freq[word_id])
                cluster_contents[word_id] = round(word_tfidf, ndigits=params['tfidf_digits'])
        new_docs[cluster_id] = ClusterContents(cluster_id=cluster_id, contents=cluster_contents)
        doc_name_map.name_id[doc] = cluster_id
        doc_name_map.id_name[cluster_id] = doc
        cluster_id += 1
    return new_docs, doc_name_map, vocabulary