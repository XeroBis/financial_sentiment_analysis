from src import FinancialSentimentAnalysis



my_data = FinancialSentimentAnalysis()
my_data.create_png_of_EDA()
# print(my_data.train_all_sentiment_analysis())
# print(my_data.train_n_grams_sentiment_analysis())
# print(my_data.train_word2vec())

my_data.bert_topics()