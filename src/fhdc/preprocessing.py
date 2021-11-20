from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# Note: Must download stuff for stopwords:
# showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
import re
import string
from data_classes import Vocabulary, Corpus, Cluster

def preprocess(docs: dict, corpus_min_frequency: int=2, doc_min_frequency: int=2, tfidf_decimals: int=4):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for doc in docs:
        # Lowercase
        current_doc = docs[doc].lower()
        # Remove Punctuation + Symbols
        regex = re.compile(f"[{re.escape(string.punctuation)}]")
        current_doc = regex.sub('', current_doc)
        # Remove Numbers
        current_doc = re.sub('\d', '', current_doc)
        # Tokenize
        current_doc = current_doc.split(' ')
        # Remove Stopwords and Empty Strings
        current_doc = [word for word in current_doc if word not in stop_words and word]
        # Lemmatize
        current_doc = [lemmatizer.lemmatize(word) for word in current_doc]

        # Transform to New Format {word: frequency}
        transformed_doc = {}
        for word in current_doc:
            if word not in transformed_doc:
                transformed_doc[word] = 1
            else:
                transformed_doc[word] += 1

        # Remove low frequency words from doc
        transformed_doc = {k:v for (k,v) in transformed_doc.items() if v >= doc_min_frequency}

        # Replace the original doc with transformed_doc
        docs[doc] = transformed_doc

    # Create Vocabulary
    vocabulary = Vocabulary({}, {}, {})
    current_word_id = 0
    for doc in docs:
        for word in docs[doc]:
            if word in vocabulary.word_id:
                existing_id = vocabulary.word_id[word]
                vocabulary.id_count[existing_id] += docs[doc][word]
            else:
                vocabulary.word_id[word] = current_word_id
                vocabulary.id_word[current_word_id] = word
                vocabulary.id_count[current_word_id] = docs[doc][word]
                current_word_id += 1

    # Find Corpus-Wide Low-Frequency Words
    infrequent_corpus_word_ids = []
    for word_id in vocabulary.id_count:
        if vocabulary.id_count[word_id] < corpus_min_frequency:
            infrequent_corpus_word_ids.append(word_id)

    # Remove Corpus-Wide Low-Frequency Words From Vocabulary
    for word_id_to_drop in infrequent_corpus_word_ids:
            vocabulary.id_count.pop(word_id_to_drop)
            word_to_drop = vocabulary.id_word[word_id_to_drop]
            vocabulary.id_word.pop(word_id_to_drop)
            vocabulary.word_id.pop(word_to_drop)

    # Remove these Corpus-Wide Low-Frequency Words from the Corpus
    # Change Words to word_ids
    # Transform to TF-IDF scores
    # Create Clusters, cluster_ids
    cluster_id = 0
    new_docs = {}
    for doc in docs:
        cluster_contents = {}
        for word in docs[doc]:
            if word in vocabulary.word_id:
                word_id = vocabulary.word_id[word]
                word_tfidf = float(docs[doc][word]) / float(vocabulary.id_count[word_id])
                cluster_contents[word_id] = round(word_tfidf, ndigits=tfidf_decimals)
        new_docs[cluster_id] = Cluster(cluster_id=cluster_id, docs=[doc], contents=cluster_contents)
        cluster_id += 1

    corpus = Corpus(new_docs)

    return corpus, vocabulary
