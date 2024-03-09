from sklearn.feature_extraction.text import CountVectorizer


def get_ngrams(text, ngram_from=2, ngram_to=2, n=None, max_features=20000):
    vec = CountVectorizer(ngram_range = (ngram_from, ngram_to),
                        max_features = max_features,
                        stop_words='english').fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]

    
