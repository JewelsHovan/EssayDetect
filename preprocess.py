import re
import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')

# Load spaCy model and GloVe model at initialization
nlp = spacy.load('en_core_web_sm')

def load_glove_model(glove_file):
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            model[word] = embedding
    return model

glove_model = load_glove_model("glove.6B.100d.txt")

def clean_text(text):
    text = re.sub("(\\d|\\W)+", " ", text)
    return text.lower()

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize(tokens):
    return " ".join([token.lemma_ for token in nlp(" ".join(tokens))])

def vectorize_tfidf(text, max_features=None):
    vectorizer = TfidfVectorizer(max_features=max_features)
    return vectorizer.fit_transform([text]).toarray()

def vectorize_glove(tokens):
    token_embeddings = [glove_model[token] for token in tokens if token in glove_model]
    if not token_embeddings:
        return np.zeros(100)  # Size of GloVe vectors
    return np.mean(token_embeddings, axis=0)

def preprocess_text(text, use_tfidf=True, max_features=None):
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    lemmatized_text = lemmatize(tokens)
    if use_tfidf:
        return vectorize_tfidf(lemmatized_text, max_features)
    return lemmatized_text

def ner_features(text):
    doc = nlp(text)
    entity_counts = {}
    for ent in doc.ents:
        entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
    return entity_counts

def pos_features(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    pos_counts = nltk.FreqDist(tag for (word, tag) in pos_tags)
    return pos_counts

def flatten_pos_counts(pos_counts):
    pos_tags = ['NN', 'JJ', 'VB', 'VBP', 'MD', 'RB', 'IN', 'NNS', 'VBG', 'VBD', ...] 
    return np.array([pos_counts.get(tag, 0) for tag in pos_tags])

def flatten_ner_counts(ner_counts):
    ner_labels = ['DATE', 'GPE', 'NORP', 'ORG', 'CARDINAL', ...]
    return np.array([ner_counts.get(label, 0) for label in ner_labels])

def feature_engineering_pipeline(text):
    preprocessed_text = preprocess_text(text, use_tfidf=False)
    
    pos_counts = pos_features(preprocessed_text)
    pos_features_vector = flatten_pos_counts(pos_counts)

    ner_counts = ner_features(preprocessed_text)
    ner_features_vector = flatten_ner_counts(ner_counts)

    glove_vector = vectorize_glove(preprocessed_text.split())

    additional_features = [len(preprocessed_text.split()), 
                           len(set(preprocessed_text.split())),
                           len(preprocessed_text),
                           np.mean([len(word) for word in preprocessed_text.split()])]
    
    features = np.hstack([pos_features_vector, ner_features_vector, glove_vector, additional_features])
    return features

def normalize_features(features_array):
    scaler = StandardScaler()
    return scaler.fit_transform(features_array)


if __name__ == "__main__":
    # Example usage
    text = "Your example text here."
    features = feature_engineering_pipeline(text)
