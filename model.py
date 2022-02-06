import pandas as pd
import numpy as np
import nltk, spacy
nltk.download('all', quiet=True)
nlp = spacy.load("en_core_web_sm")
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Ignore the warnings
import warnings
warnings.filterwarnings(action='ignore')

def recommend_products (uname, data, ml_model_path, rec_sys_path):
    ml_model = pickle.load(open(ml_model_path, 'rb'))
    rec_system = pickle.load(open(rec_sys_path, 'rb'))

    # Recommending the top 20 products to the user.
    top_20 = rec_system.loc[uname].sort_values(ascending=False)[0:20]

    # Joining the train and top_20 datasets to create a new train dataset
    train_new = pd.merge(top_20, data, left_on='prodname', right_on='prodname', how='left')

    # Creating the vectorizer for the user review text for the new train dataset - 'train_new'
    rec_vector = TfidfVectorizer(min_df=2, analyzer="word", max_features=3948)
    rec_model = rec_vector.fit_transform(list(train_new['cleaned_reviews_text']))

    # Predicting the sentiments for all the reviews for the top 20 recommended products for a user
    X = pd.DataFrame(rec_model.toarray(), columns=vectorizer.get_feature_names())
    y_pred = ml_model.predict(X)  # Logistic Regression model is used to make predictions
    train_new['user_sentiment_pred'] = y_pred

    # Filtering the percentage of positive sentiments for all the reviews of each of the top 20 products
    train_final = train_new[['prodname', 'user_sentiment_pred']]
    top_products = train_final.groupby(['prodname']).sum() * 100 / train_final.groupby(['prodname']).count()

    # Filtering out the top 5 products with the highest percentage of positive reviews
    top_5 = top_products.sort_values(by=['user_sentiment_pred'], ascending=False)[0:5]
    top_5.rename(columns={'user_sentiment_pred': 'positive_sentiments(%)'}, inplace=True)
    top_5 = top_5.reset_index()

    return top_5['prodname']



