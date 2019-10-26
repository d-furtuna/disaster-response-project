"""
ETL Pipeline Preparation module

Script loads the data, cleans it and stores it into an sqlite database.
Output database contains two tables: 1) "messages" with cleaned messages and
2) "words" with top 10 words per category
Command-line arguments for input and output data files are passed to the script.

Args:
    arg1 (str): (filepath) to disaster response input messages.
    arg2 (str): (filepath) to categories to which the messages have been categoried.
    arg3 (str): (filepath) to the sqlite database where the clean dataset will be stored.
                If file exists it will be replaced.

"""


import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer


def load_data(messages_filepath, categories_filepath):
    '''
        Function to load the input datasets and merge them.
        Function returns the merged dataset as a DataFrame.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    '''
        Function cleans the input DataFrame, by:
        - splitting "categories" column in separate category columns,
        - converting values to binary, and
        - dropping duplicates.
        Function returns the cleaned dataset as a DataFrame.
    '''
    # split categories into separate category columns
    categories = df["categories"]
    categories = categories.str.split(';', expand=True)
    category_colnames = categories.iloc[0,:].str.split(pat = r"-\d")
    category_colnames = [name[0] for name in category_colnames]
    categories.columns = category_colnames
    # convert values to binary
    for colnames in categories.columns:
        categories[colnames] = categories[colnames].str[-1:].astype(int).astype(bool).astype(int) #remove some 2 in data
    df.drop(columns="categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1)

    # remove duplicates
    df.drop_duplicates(keep=False,inplace=True)

    return df


def tokenize(text):
    '''
        Function to clean the input string by: normalizing, tokenizing, 
        removing stopwords, lemmatizing for nouns and for verbs, and stemming.
        Output is a list of cleaned words.
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Lemmatize for nouns and for verbs, and stem
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    lemmed_stemmed = [PorterStemmer().stem(w) for w in lemmed]
    return lemmed_stemmed


def get_top_words(df):
    '''
        Function get the top 10 tokenized words and their number, present in each category.
        Output is a data frame with 3 columns (category, word, count)
    '''
    # get counts of each token (word) in text data
    vect = CountVectorizer(tokenizer = tokenize)
    X = vect.fit_transform(df["message"].values)
    # create category list
    y_cats = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products',\
       'search_and_rescue', 'security', 'military', 'child_alone', 'water',\
       'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',\
       'death', 'other_aid', 'infrastructure_related', 'transport',\
       'buildings', 'electricity', 'tools', 'hospitals', 'shops',\
       'aid_centers', 'other_infrastructure', 'weather_related', 'floods',\
       'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    # create "vocabulary" data frame of top 300 words in all messages
    sort_vocabulary = pd.DataFrame.from_dict(vect.vocabulary_, orient='index', columns=['ids'])
    sort_vocabulary.sort_values(by=['ids'], inplace=True)
    sort_vocabulary.reset_index(inplace=True)
    sort_vocabulary["count"] = np.transpose(X.sum(axis = 0))
    sort_vocabulary.rename(columns={"index": "word"}, inplace=True)
    sort_vocabulary.sort_values(by=['count'], inplace=True, ascending=False)
    df_word_count = pd.DataFrame(X[:,sort_vocabulary[0:300]["ids"].values].toarray(), columns = sort_vocabulary[0:300]["word"].values)
    # create the output data frame (df_plot) containing top 10 words per category
    df_plot=pd.DataFrame()
    for cat in y_cats:
        df_temp = df_word_count.copy()
        df_temp["category"] = df[cat]
        df_temp = df_temp[df_temp["category"]==1]
        sum_top_w = df_temp.loc[:, df_temp.columns != 'category'].values.sum(axis = 0)
        ind_top_w = np.argpartition(sum_top_w, -10)[-10:]
        df_top_w = pd.DataFrame(index = range(10))
        df_top_w["category"] = cat
        df_top_w["word"] = df_temp.iloc[:,ind_top_w].columns
        df_top_w["count"] = sum_top_w[ind_top_w]
        df_top_w.sort_values(["count"], ascending=False, inplace=True)
        df_plot = df_plot.append(df_top_w, ignore_index=True)

    return df_plot

def save_data(df, df_plot, database_filename):
    '''
        Function stores the clean data into an SQLite database of the specified file path.
        Table "messages"  contains the cleaned messages.
        Table "words" stores the top 10 words per category (used later for the web-plot).
    '''
    # save the clean messages dataset into an sqlite database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("messages", engine, index=False, if_exists="replace")
    df_plot.to_sql("words", engine, index=False, if_exists="replace")

    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Calculating top words per category...')
        df_plot = get_top_words(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, df_plot, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()