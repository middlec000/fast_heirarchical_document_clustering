from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# Note: Must download stuff for stopwords:
# showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
import re
import string
from data_classes import Vocabulary, Doc, Corpus

def preprocess(docs: dict, min_frequency: int=2):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for doc in docs.keys():
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

        # Transform to New Format + Count Words
        transformed_doc = {}
        for word in current_doc:
            if word not in transformed_doc.keys():
                transformed_doc[word] = 1
            else:
                transformed_doc[word] += 1

        # Replace the original doc with transformed_doc
        docs[doc] = transformed_doc
    
    # Create Vocabulary
    vocabulary = Vocabulary({}, {}, {})
    for doc in docs:
        for word in docs[doc].keys():
            if word not in vocabulary.word_id.keys():
                new_id = len(vocabulary.word_id)
                vocabulary.word_id[word] = new_id
                vocabulary.id_word[new_id] = word
                vocabulary.id_count[new_id] = 1
            else:
                existing_id = vocabulary.word_id[word]
                vocabulary.id_count[existing_id] += docs[doc][word]

    # Find Corpus Wide Low-Frequency Words
    word_ids_to_drop = []
    for word_id in vocabulary.id_count.keys():
        if vocabulary.id_count[word_id] < min_frequency:
            word_ids_to_drop.append(word_id)

    # Remove Corpus Wide Low-Frequency Words From Vocabulary
    for word_id_to_drop in word_ids_to_drop:
            del vocabulary.id_count[word_id_to_drop]
            word_to_drop = vocabulary.id_word[word_id_to_drop]
            del vocabulary.id_word[word_id_to_drop]
            del vocabulary.word_id[word_to_drop]

    # Remove these Words from the Corpus and Calculate TF-IDF Scores
    for doc in docs:
        tfidf_doc = {}
        for word in docs[doc]:
            if word in vocabulary.word_id.keys():
                word_id = vocabulary.word_id[word]
                tfidf_doc[word_id] = docs[doc][word] / vocabulary.id_count[word_id]
        docs[doc] = Doc(doc_id=doc, contents=tfidf_doc)

    corpus = Corpus(docs)

    return corpus, vocabulary
