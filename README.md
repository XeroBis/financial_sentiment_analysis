# Financial Sentiment Analysis

Clone the repository and install the required dependencies:

````bash
git clone https://github.com/yourusername/financial-sentiment-analysis.git
cd financial-sentiment-analysis
pip install -r requirements.txt
````

This notebook is organized into different sections:

- Data Collection: 
    - Financial PhraseBank dataset is loaded using the Hugging Face Datasets library.

- Exploratory Data Analysis (EDA): 
    - Analyzing the dataset, visualizing the distribution of labels, and exploring the length of phrases.

- Pre-processing & Text Cleaning: 
    - Cleaning the text data by removing stopwords, lemmatizing, and tokenizing.

- Train Sentiment Analysis ML Models using TF-IDF / BoW: 
    - Training various machine learning models using Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) as features.

- Train Word2Vec for Sentiment Analysis: 
    - Training a Word2Vec model for sentiment analysis.

- Topic Modelling (LDA / NMF) & Visualization of Topics: 
    - Applying topic modeling techniques (Latent Dirichlet Allocation - LDA, Non-Negative Matrix Factorization - NMF) to identify topics within the dataset.

- Fine-tune LMs for Sentiment Analysis (BERT, RoBERTa, AlBERT, DistilBERT...): 
    - Fine-tuning pre-trained language models for sentiment analysis using transformers library.

- Model Evaluation & Explainability: 
    - Evaluating models and using SHAP for model interpretability.

- Pipeline: 
    - Using pre-trained models to perform sentiment analysis on a set of randomly selected financial phrases.
