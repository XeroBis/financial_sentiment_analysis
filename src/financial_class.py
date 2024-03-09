from .utils import get_ngrams
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import evaluate
import sklearn
import spacy
import nltk
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import LatentDirichletAllocation, NMF

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer

from bertopic import BERTopic

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          RobertaTokenizer, RobertaForSequenceClassification,
                          AlbertTokenizer, AlbertForSequenceClassification,
                          TrainingArguments, Trainer, pipeline)

from wordcloud import WordCloud


class FinancialSentimentAnalysis:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        pio.renderers.default = "colab"
        self.dataset = None
        self.tmp_df = None
        self.model = None
        self.tokenizer = None
        self.load_dataset()
        self.preprocess_text()

    def load_dataset(self):
        self.dataset = load_dataset(
            'financial_phrasebank', 'sentences_allagree')
        self.train_df = self.dataset['train'].to_pandas()

    def preprocess_text(self):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def clean_text(row):
            data = row.lower()
            data = re.sub(r'[^a-zA-Z0-9\s.]', ' ', data)
            data = re.sub(r'( ){2,}', ' ', data)
            data = word_tokenize(data)
            data = [word for word in data if word not in stop_words]
            data = [lemmatizer.lemmatize(word) for word in data]
            return " ".join(data)

        self.train_df['cleaned_text'] = self.train_df["sentence"].apply(
            clean_text)

    # Create FIGURES
    def create_png_of_EDA(self):
        self.train_df['length_text'] = self.train_df['sentence'].apply(
            lambda x: len(str(x)))

        fig = px.pie(self.train_df, names='label', labels='label')
        fig.write_image("images/pie.png")

        fig = px.box(self.train_df, x='label', y='length_text')
        fig.write_image("images/box.png")

        phrase_length = pd.DataFrame(data=self.dataset["train"]).apply(
            lambda x: len(x["sentence"]), axis=1)
        sns.histplot(phrase_length, bins=30, kde=True)
        plt.title('Distribution of Phrase Lengths')
        plt.xlabel('Phrase Length')
        plt.savefig("images/hist_phrase_length.png")

        number_words = pd.DataFrame(data=self.dataset["train"]).apply(
            lambda x: len(x["sentence"].split(" ")), axis=1)
        sns.histplot(number_words, bins=30, kde=True)
        plt.title('Distribution Number of words in Phrases')
        plt.xlabel('Phrase Length')
        plt.savefig("images/hist_number_words.png")

        text = " ".join(self.dataset["train"]["sentence"])
        wordcloud = WordCloud(max_font_size=None).generate(text)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig("images/wordcloud.png")

    def create_bigrams(self):
        bigrams = get_ngrams(
            self.train_df['cleaned_text'], ngram_from=2, ngram_to=2, n=150)
        bigrams_df = pd.DataFrame(bigrams)
        bigrams_df.columns = ["Bigram", "Frequency"]
        fig = px.bar(bigrams_df, x='Bigram', y='Frequency')
        fig.write_image("images/bigrams.png")

    # Sentiment Analysis
    def train_one_sentiment_analysis_model(self, vectorizer, model):

        X = self.train_df['cleaned_text'].values
        X = vectorizer.fit_transform(X)
        y = self.train_df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        clf = model.fit(X_train, y_train)

        y_predictions = clf.predict(X_test)

        return clf, accuracy_score(y_test, y_predictions)

    def train_all_sentiment_analysis(self):
        vectorizers = {
            "BoW": CountVectorizer(max_features=1000, min_df=0.1, stop_words='english'),
            "BoW+Ngrams": CountVectorizer(max_features=1000, min_df=0.05, stop_words='english', ngram_range=(1, 3)),
            "Tf-IDF": TfidfVectorizer(max_features=10000, stop_words='english'),
            "TF-IDF+Ngrams": TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 4))
        }
        classifiers = {
            'Logistic Regression OVR': LogisticRegression(multi_class='ovr'),
            'Logistic Regression Multinomial': LogisticRegression(multi_class='multinomial', solver='lbfgs'),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC()
        }
        results = []
        for vectorizer_name, vectorizer in vectorizers.items():
            for classifier_name, classifier in classifiers.items():
                model, accuracy = self.train_one_sentiment_analysis_model(
                    vectorizer, classifier)
                results.append([vectorizer_name, classifier_name, accuracy])

        return results

    def train_n_grams_sentiment_analysis(self):
        classifiers = {
            'Logistic Regression OVR': LogisticRegression(multi_class='ovr'),
            'Logistic Regression Multinomial': LogisticRegression(multi_class='multinomial', solver='lbfgs'),
            'Random Forest': RandomForestClassifier(),
            'SVM': SVC()
        }
        vectorizers = {
            "BoW": CountVectorizer,
            "Tf-IDF": TfidfVectorizer
        }
        # Example: testing up to 5-grams
        n_gram_ranges = [(1, n) for n in range(2, 6)]
        results = []
        for n_gram_range in n_gram_ranges:
            for vectorizer_name, vectorizer in vectorizers.items():
                for classifier_name, classifier in classifiers.items():
                    vect = vectorizer(
                        max_features=10000, stop_words='english', ngram_range=n_gram_range)
                    model, accuracy = self.train_one_sentiment_analysis_model(
                        vect, classifier)
                    results.append(
                        [n_gram_range, vectorizer_name, classifier_name, accuracy])

        return results

    def train_word2vec(self):
        texts = [i.split() for i in self.train_df['cleaned_text']]
        model = Word2Vec(sentences=texts, vector_size=300,
                         window=10, min_count=2, workers=4)

        def document_vector(doc):
            word_vectors = [model.wv[word] for word in doc if word in model.wv]
            return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

        self.train_df['word2vec_avg'] = self.train_df['cleaned_text'].apply(
            lambda x: document_vector(x.split()))

        X_train, X_test, y_train, y_test = train_test_split(
            self.train_df['word2vec_avg'].tolist(), self.train_df['label'], test_size=0.2, random_state=42)

        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)

        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)

        return accuracy_score(y_test, predictions)

    # Topic Modelling & Visualization of Topics
    def get_topics_list_lda(self, nb_topics=5):
        tokenizer = RegexpTokenizer(r'\w+')

        tfidf = TfidfVectorizer(lowercase=True,
                                stop_words='english',
                                ngram_range=(1, 1), min_df=0.1,
                                tokenizer=tokenizer.tokenize)
        train_data = tfidf.fit_transform(
            self.train_df['cleaned_text'].tolist())
        model = LatentDirichletAllocation(
            n_components=nb_topics)  # Number of topics
        model.fit_transform(train_data)
        lda_components = model.components_

        terms = tfidf.get_feature_names_out()
        topics = []
        for _, component in enumerate(lda_components):
            zipped = zip(terms, component)
            top_terms_key = sorted(
                zipped, key=lambda t: t[1], reverse=True)[:7]
            top_terms_list = list(dict(top_terms_key).keys())
            topics.append(top_terms_list)
        return topics

    def get_coherence_score(self):

        def sent_to_words(sentences):
            for sentence in sentences:
                # deacc=True removes punctuations
                yield (simple_preprocess(str(sentence), deacc=True))

        docs = list(sent_to_words(self.train_df['cleaned_text'].tolist()))

        dictionary = Dictionary(docs)
        dictionary.filter_extremes(no_below=10, no_above=0.2)
        # Create dictionary and corpus required for Topic Modeling
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        # Set parameters.
        num_topics = 10
        chunksize = 500
        passes = 20
        iterations = 400
        eval_every = 1
        temp = dictionary[0]  # only to "load" the dictionary.
        id2word = dictionary.id2token

        lda_model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize,
                             alpha='auto', eta='auto',
                             iterations=iterations, num_topics=num_topics,
                             passes=passes, eval_every=eval_every)

        coherence_model_lda = CoherenceModel(
            model=lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
        coherence_ldas = coherence_model_lda.get_coherence_per_topic()

        return lda_model.print_topics(), coherence_ldas

    def visualize_pydalvis(self):

        documents = [sample['sentence'] for sample in self.dataset['train']]

        stop_words = set(stopwords.words('english'))
        texts = []

        for document in documents:
            # Tokenization et mise en minuscules
            tokens = word_tokenize(document.lower())
            filtered_tokens = [word for word in tokens if word.isalnum(
            ) and word not in stop_words]  # Suppression des stopwords et de la ponctuation
            texts.append(filtered_tokens)

        # Création du modèle LDA avec Gensim
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda_model = LdaModel(corpus, num_topics=3,
                             id2word=dictionary, passes=15)

        # Visualisation avec pyLDAvis
        lda_vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(lda_vis_data, 'html/doc_name.html')

    def find_topics_NMF(self):

        documents = [sample['sentence'] for sample in self.dataset['train']]

        # Prétraitement des documents : tokenization, suppression des stopwords
        stop_words = set(stopwords.words('english'))
        texts = []

        for document in documents:
            # Tokenization et mise en minuscules
            tokens = word_tokenize(document.lower())
            filtered_tokens = [word for word in tokens if word.isalnum(
            ) and word not in stop_words]  # Suppression des stopwords et de la ponctuation
            texts.append(' '.join(filtered_tokens))

        # Création de la matrice TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # Entraînement du modèle NMF
        num_topics = 3
        nmf_model = NMF(n_components=num_topics, random_state=42)
        nmf_model.fit(tfidf_matrix)

        # Visualisation des thèmes découverts
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Création de sous-graphiques pour chaque thème
        _, axes = plt.subplots(1, num_topics, figsize=(15, 5), sharey=True)
        plt.subplots_adjust(wspace=0.4)

        for topic_idx, ax in enumerate(axes):
            top_words_idx = nmf_model.components_[topic_idx].argsort(
            )[-10:][::-1]  # Top 10 mots-clés pour chaque thème
            top_words = [feature_names[i] for i in top_words_idx]
            top_word_weights = nmf_model.components_[topic_idx][top_words_idx]

            ax.barh(top_words, top_word_weights, color='skyblue')
            ax.invert_yaxis()
            ax.set_title(f"Topic {topic_idx + 1}")

        plt.suptitle("Topics découverts avec NMF")
        plt.savefig("images/NMF_topics.png")

    def bert_topics(self):
        docs = self.dataset['train']['sentence']
        topic_model = BERTopic()
        _, _ = topic_model.fit_transform(docs)

        fig = topic_model.visualize_topics()
        fig.write_html("html/bert_viz_topics.html")
        fig = topic_model.visualize_documents(docs)
        fig.write_html("html/bert_docs.html")
        fig = topic_model.visualize_heatmap()
        fig.write_html("html/bert_heatmap.html")

    # Fine tuning LMs
    def fine_tune_LM(self, name_model, tokenizer_class, sequence_classification):
        def tokenize_function(examples):
            return tokenizer(examples["sentence"], padding="max_length", truncation=True)

        tokenizer = tokenizer_class.from_pretrained(name_model)

        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)

        small_train_dataset = tokenized_datasets["train"].select(range(1812))
        small_eval_dataset = tokenized_datasets["train"].select(
            range(1812, 2264))

        model = sequence_classification.from_pretrained(
            name_model, num_labels=3)

        metric = evaluate.load("accuracy")

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
        )
        trainer.train()
        trainer.evaluate()
        trainer.save_model(f"./{name_model.upper()}")
        tokenizer.save_pretrained(f"./{name_model.upper()}")

    def fine_tune_bert(self):
        self.fine_tune_LM(
            "bert-base-uncased", AutoTokenizer, AutoModelForSequenceClassification)

    def fine_tune_roberta(self):
        self.fine_tune_LM(
            "roberta-base", RobertaTokenizer, RobertaForSequenceClassification)

    def fine_tune_Albert(self):
        self.fine_tune_LM(
            "albert-base-v2", AlbertTokenizer, AlbertForSequenceClassification)

    def pipeline_test_lms(self):
        def check_predict(predictions, results):
            res = []
            mean_acc = sum([x[0]["score"]
                           for x in predictions]) / len(predictions) * 100

            labels = [x[0]['label'] for x in predictions]
            for i in range(len(labels)-1):
                if labels[i] == "LABEL_" + str(results[i]):
                    res.append(1)
            return f"{sum(res)/len(results) * 100}% de précision, {mean_acc} de certitude"

        BERT_results = []
        RoBERTa_results = []
        AlBERT_results = []

        indices = np.random.randint(0, len(self.dataset["train"])-1, 10)
        phrases = [self.dataset["train"]["sentence"][i] for i in indices]
        results = [self.dataset["train"]["label"][i] for i in indices]

        BERT_sentiment_pipeline = pipeline(
            "sentiment-analysis", model="./bert-base-uncased".upper(), tokenizer="./bert-base-uncased".upper())
        RoBERTa_sentiment_pipeline = pipeline(
            "sentiment-analysis", model="./roberta-base".upper(), tokenizer="./roberta-base".upper())
        AlBERT_sentiment_pipeline = pipeline(
            "sentiment-analysis", model="./albert-base-v2".upper(), tokenizer="./albert-base-v2")

        for phrase in phrases:

            # Now you can use the pipeline for sentiment analysis
            BERT_results.append(BERT_sentiment_pipeline(phrase))
            RoBERTa_results.append(RoBERTa_sentiment_pipeline(phrase))
            AlBERT_results.append(AlBERT_sentiment_pipeline(phrase))

        print(f'BERT : {check_predict(BERT_results, results)}', f'RoBERTa : {check_predict(RoBERTa_results, results)}',
              f'AlBERT : {check_predict(AlBERT_results, results)}', sep="\n")
